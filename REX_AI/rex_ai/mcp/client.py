"""How agents reach external services: MCP tool calls to the REX AI tool server (owner: Member 3).

    from rex_ai.mcp import client as tools
    tools.call("send_email", to=[...], subject="...", body="...")      # -> dict, or ToolCallError with the reason
    tools.chat("openai", messages, temperature=0.3, json_mode=True)    # -> the model's text

REX_MCP_URL set (e.g. http://127.0.0.1:8765/mcp): calls go to that standalone server (python -m rex_ai.mcp.server
--http). Empty: the server in rex_ai/mcp/server.py runs inside this process, spoken to over MCP all the same.
One connection per process is kept open on a background event loop (a call costs a few milliseconds), and it is
re-opened once if it drops.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional

from rex_ai import config

log = logging.getLogger("rex.mcp")


class ToolCallError(RuntimeError):
    """The tool ran and failed (the message is the reason), or the tool server could not be reached."""


_lock = threading.Lock()
_state: dict[str, Any] = {"pid": None, "loop": None, "client": None}


def target() -> str:
    return config.MCP_URL or "in-process"


def _connect():
    """The open MCP client for this process (created on first use)."""
    with _lock:
        if _state["client"] is not None and _state["pid"] == os.getpid():
            return _state["loop"], _state["client"]
        from mcp import Client
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="rex-mcp-client", daemon=True).start()
        if config.MCP_URL:
            server = config.MCP_URL
        else:
            from rex_ai.mcp.server import server
        client = Client(server, read_timeout_seconds=config.MCP_TIMEOUT)
        try:
            asyncio.run_coroutine_threadsafe(client.__aenter__(), loop).result(timeout=30)
        except Exception as exc:
            loop.call_soon_threadsafe(loop.stop)
            raise ToolCallError(f"The MCP tool server at {target()} could not be reached: {exc}") from exc
        _state.update(pid=os.getpid(), loop=loop, client=client)
        return loop, client


def reset() -> None:
    """Close the connection; the next call opens a new one."""
    with _lock:
        loop, client = _state["loop"], _state["client"]
        _state.update(pid=None, loop=None, client=None)
    if client is not None and loop is not None:
        try:
            asyncio.run_coroutine_threadsafe(client.__aexit__(None, None, None), loop).result(timeout=5)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)


def _run(make_coro, timeout: Optional[float] = None):
    for attempt in (1, 2):
        loop, client = _connect()
        try:
            return asyncio.run_coroutine_threadsafe(make_coro(client), loop).result(
                timeout=timeout or config.MCP_TIMEOUT + 5)
        except ToolCallError:
            raise
        except Exception as exc:                    # connection dropped (server restarted, network)
            reset()
            if attempt == 2:
                raise ToolCallError(f"The MCP tool server at {target()} could not be reached: {exc}") from exc
            log.warning("MCP connection to %s failed (%s) - reconnecting", target(), exc)


def call(name: str, **arguments) -> dict:
    """Call an MCP tool; returns its JSON result or raises ToolCallError with the tool's own error message."""
    result = _run(lambda c: c.call_tool(name, arguments))
    text = "".join(getattr(part, "text", "") for part in result.content)
    if result.is_error:
        raise ToolCallError(text.removeprefix(f"Error executing tool {name}: ") or f"tool {name} failed")
    if isinstance(result.structured_content, dict):
        data = result.structured_content
        return data["result"] if set(data) == {"result"} and isinstance(data["result"], dict) else data
    return json.loads(text) if text else {}


def list_tools() -> list[dict]:
    result = _run(lambda c: c.list_tools(), timeout=30)
    return [{"name": t.name, "description": (t.description or "").split("\n")[0]} for t in result.tools]


def chat(provider: str, messages: list[dict], temperature: float = 0.3, json_mode: bool = False) -> str:
    """LLM chat through the llm_chat tool; returns the model's text (raises ToolCallError on failure)."""
    return call("llm_chat", provider=provider, messages=messages, temperature=temperature,
                json_mode=json_mode)["content"]


class HttpSession:
    """A requests-like session whose GETs go through the http_get tool (used by the RFP web-portal source)."""

    def __init__(self):
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float = 20) -> "HttpResponse":
        try:
            data = call("http_get", url=url, timeout=timeout, headers=dict(self.headers))
        except ToolCallError as exc:
            raise ConnectionError(str(exc)) from exc
        return HttpResponse(data["status_code"], data["text"], data.get("url", url))


class HttpResponse:
    def __init__(self, status_code: int, text: str, url: str):
        self.status_code, self.text, self.url = status_code, text, url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ConnectionError(f"HTTP {self.status_code} for {self.url}")
