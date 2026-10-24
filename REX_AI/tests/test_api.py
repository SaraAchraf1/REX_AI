"""API contract tests (owner: Member 3). Run: pytest -q"""
from fastapi.testclient import TestClient

from rex_ai.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["courses"] >= 10


def test_course_list_and_filter():
    all_courses = client.get("/courses").json()
    beginners = client.get("/courses", params={"skill_level": "Beginner"}).json()
    assert len(beginners) < len(all_courses)
    assert all(c["skill_level"] == "Beginner" for c in beginners)


def test_unknown_course_404():
    assert client.get("/courses/NOPE-999").status_code == 404


def test_chat_returns_contract_fields():
    r = client.post("/chat", json={"session_id": "api-1", "message": "I want to work in cybersecurity"})
    assert r.status_code == 200
    body = r.json()
    for field in ("reply", "language", "stage", "intent", "profile", "sources"):
        assert field in body
    assert body["language"] == "en"


def test_chat_arabic_detected():
    body = client.post("/chat", json={"session_id": "api-2", "message": "عايز كورس في تحليل البيانات"}).json()
    assert body["language"] == "ar"


def test_chat_rejects_empty_message():
    r = client.post("/chat", json={"session_id": "api-3", "message": ""})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_lead_requires_consent():
    r = client.post("/leads", json={"session_id": "api-4", "name": "Test Lead",
                                    "email": "lead@example.com", "consent": False})
    assert r.status_code == 422


def test_lead_requires_contact():
    r = client.post("/leads", json={"session_id": "api-5", "name": "Test Lead", "consent": True})
    assert r.status_code == 422


def test_lead_saved_with_consent():
    r = client.post("/leads", json={"session_id": "api-6", "name": "Test Lead", "email": "lead@example.com",
                                    "interested_course_ids": ["DS-101"], "consent": True})
    assert r.status_code == 201
    assert r.json()["lead_id"]


def test_route_sends_tender_question_to_rfp_agent():
    body = client.post("/route", json={"session_id": "api-7", "message": "find training tenders in Egypt"}).json()
    assert body["agent"] == "rfp_research"
    assert "answer" in body["result"]                    # Sprint 2: answered by the RFP agent


def test_route_defaults_to_rex():
    body = client.post("/route", json={"session_id": "api-8", "message": "how much is the PMP course?"}).json()
    assert body["agent"] == "rex_assistant"
