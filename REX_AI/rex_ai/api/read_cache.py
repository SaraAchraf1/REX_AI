"""Answers the API remembers, so pages do not wait on the database for data that did not change (owner: Member 3).

The shared database is far away (each query is a 250-450 ms round trip to Neon), but administrators' pages mostly
re-read the same lists. So:
  * a GET of shared administrator data (tenders, curricula, documents, campaigns, courses, users, tools) is answered
    from memory when the same request was answered less than MAX_AGE seconds ago;
  * any request that changes something (POST, PATCH, PUT, DELETE) empties the whole cache, so nobody ever sees their
    own change late;
  * keep_warm() re-runs, every REFRESH_AFTER seconds, the requests someone used in the last KEEP_WARM_FOR seconds:
    the next page view is instant, and changes made outside a request (the task orchestrator, the flow repairs, other
    servers) show up within about REFRESH_AFTER seconds.
Per-user data (chats, the login) and anything that changes on its own (tasks, jobs) is never cached.
Only used after the access check (main.access_control) has accepted the request. REX_API_CACHE=0 switches it off.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time

from starlette.requests import Request
from starlette.responses import Response

from rex_ai import config

log = logging.getLogger("rex.api.cache")
CACHEABLE = ("/rfp", "/curriculum", "/documents", "/campaigns", "/courses", "/auth/users", "/orchestrator/tools")
NEVER = ("/pdf", "/download", "/export.csv")
REFRESH_HEADER = "x-rex-cache-refresh"           # marks the cache's own refresh requests (secret value, in-process only)
REFRESH_SECRET = secrets.token_hex(16)
MAX_AGE, REFRESH_AFTER, KEEP_WARM_FOR = 120.0, 15.0, 600.0
READS = {"GET", "HEAD", "OPTIONS"}

_entries: dict[str, dict] = {}
_generation = 0                                  # bumped by every change: answers computed before it are dropped


def is_refresh(request: Request) -> bool:
    return request.headers.get(REFRESH_HEADER) == REFRESH_SECRET


def _key(request: Request) -> str:
    return request.url.path + (f"?{request.url.query}" if request.url.query else "")


def cacheable(request: Request, path: str) -> bool:
    return (config.API_CACHE and request.method == "GET" and path.startswith(CACHEABLE)
            and not path.endswith(NEVER))


def clear() -> None:
    global _generation
    _generation += 1
    _entries.clear()


async def handle(request: Request, call_next, path: str):
    """Pass an allowed request on, answering from / filling the cache, or emptying it for a change."""
    if request.method not in READS:
        clear()                                  # before: no refresh started now can store the old answer
        try:
            return await call_next(request)
        finally:
            clear()                              # after: the change is in the database
    if not cacheable(request, path):
        return await call_next(request)
    key, now = _key(request), time.monotonic()
    hit = _entries.get(key)
    if hit and now - hit["at"] < MAX_AGE:
        hit["used"] = now
        return Response(hit["body"], status_code=200, media_type=hit["type"], headers={"x-rex-cache": "hit"})
    generation = _generation
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200 or not content_type.startswith("application/json"):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])
    if generation == _generation:
        _entries[key] = {"body": body, "type": content_type, "at": now, "used": now}
    return Response(body, status_code=200, media_type=content_type, headers={"x-rex-cache": "miss"})


async def _refresh(app, key: str) -> None:
    import httpx
    generation = _generation
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://rex-cache",
                                     timeout=120) as client:
            r = await client.get(key, headers={REFRESH_HEADER: REFRESH_SECRET})
    except Exception as exc:                     # keep the old answer; it expires after MAX_AGE anyway
        log.warning("Cache refresh of %s failed: %s", key, exc)
        return
    entry = _entries.get(key)
    if r.status_code == 200 and entry is not None and generation == _generation:
        entry.update(body=r.content, at=time.monotonic())


async def keep_warm(app) -> None:
    """Background loop (started with the API): refresh what people are using, forget what they stopped using."""
    while True:
        await asyncio.sleep(5)
        now = time.monotonic()
        for key, entry in list(_entries.items()):
            if now - entry["used"] > KEEP_WARM_FOR:
                _entries.pop(key, None)
        due = [k for k, e in _entries.items() if now - e["at"] > REFRESH_AFTER]
        if due:
            await asyncio.gather(*(_refresh(app, k) for k in due))
