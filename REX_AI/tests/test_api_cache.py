"""The API's read cache (rex_ai/api/read_cache.py) and the non-blocking flow status. Run: pytest -q tests/test_api_cache.py"""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent
from rex_ai.api import read_cache
from rex_ai.api.main import app
from rex_ai.core.schemas import CurriculumRequest

COURSE = dict(course_title="Cache Testing 101", target_audience="Testers", skill_level="Beginner", total_hours=12)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "API_CACHE", True)
    read_cache.clear()
    yield TestClient(app)
    read_cache.clear()


def _ids(r):
    return {c["curriculum_id"] for c in r.json()}


def test_a_repeated_read_is_answered_from_memory(client):
    first = client.get("/curriculum")
    assert first.headers["x-rex-cache"] == "miss"
    made_outside = CurriculumAgent().generate(CurriculumRequest(**COURSE))      # no request: the cache cannot know
    again = client.get("/curriculum")
    assert again.headers["x-rex-cache"] == "hit" and made_outside.curriculum_id not in _ids(again)
    assert client.get("/curriculum", params={"q": "x"}).headers["x-rex-cache"] == "miss"   # other filters, own entry


def test_any_change_empties_the_cache(client):
    client.get("/curriculum")
    made = client.post("/curriculum", json=COURSE).json()["curriculum_id"]
    after = client.get("/curriculum")
    assert after.headers["x-rex-cache"] == "miss" and made in _ids(after)


def test_keep_warm_picks_up_changes_made_outside_requests(client):
    client.get("/curriculum")
    made_outside = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    asyncio.run(read_cache._refresh(app, "/curriculum"))
    assert made_outside.curriculum_id in _ids(client.get("/curriculum"))


def test_cache_never_skips_the_login_check(client, monkeypatch):
    client.get("/curriculum")                                                  # cached while logins are off
    monkeypatch.setattr(config, "API_AUTH", True)
    assert client.get("/curriculum").status_code == 401
    assert client.get("/curriculum", headers={read_cache.REFRESH_HEADER: "guess"}).status_code == 401


def test_per_user_and_live_data_is_never_cached(client):
    for path in ("/orchestrator/tasks", "/orchestrator/jobs", "/health"):
        client.get(path)
        assert "x-rex-cache" not in client.get(path).headers, path


def test_flow_status_answers_at_once_while_the_first_check_runs(monkeypatch):
    import rex_ai.api.main as main
    monkeypatch.setattr(main, "last_report", lambda: None)
    monkeypatch.setattr(main.Orchestrator, "check", lambda self, repair=True, deep=False: time.sleep(3))
    started = time.perf_counter()
    report = TestClient(app).get("/orchestrator/status").json()
    assert time.perf_counter() - started < 2.5
    assert report["pending"] is True and report["pipeline"] and report["checks"][0]["status"] == "warning"
