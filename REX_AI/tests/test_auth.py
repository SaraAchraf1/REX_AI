"""Accounts and access (owner: Member 3). Run: pytest -q tests/test_auth.py

Sign up / log in as a user (REX Course Assistant only) or an administrator (every agent); each user's conversations
are stored under their account and hidden from other accounts; work done by administrators is shared by all of them.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.core import auth, db


def _email(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:6]}@academy.example"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH", True)
    from rex_ai.api.main import app
    return TestClient(app)


def _signup(api, role="user", code=None, name="Test"):
    r = api.post("/auth/signup", json={"name": name, "email": _email(role), "password": "correct-horse",
                                       "role": role, "admin_code": code})
    assert r.status_code == 201, r.text
    body = r.json()
    return body["user"], {"Authorization": f"Bearer {body['token']}"}


@pytest.fixture
def admin(api):
    return _signup(api, "admin", config.ADMIN_SIGNUP_CODE, name="Dean")


@pytest.fixture
def user(api):
    return _signup(api, "user", name="Sara")


# =========================================================================== passwords and accounts
def test_password_is_hashed_and_verified():
    h = auth.hash_password("correct-horse")
    assert "correct-horse" not in h and h.startswith("pbkdf2_sha256$")
    assert auth.verify_password("correct-horse", h) and not auth.verify_password("wrong", h)
    assert auth.hash_password("correct-horse") != h                     # random salt


def test_signup_validation():
    with pytest.raises(auth.AuthError, match="e-mail"):
        auth.signup("A", "not-an-email", "correct-horse")
    with pytest.raises(auth.AuthError, match="8 characters"):
        auth.signup("A", _email("short"), "short")
    email = _email("dup")
    auth.signup("A", email, "correct-horse")
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.signup("A", email.upper(), "correct-horse")                # e-mails are case-insensitive
    stored = db.get_user(email=email)
    assert stored["password_hash"] != "correct-horse" and "password_hash" not in auth.public(stored)


def test_admin_signup_needs_the_code_once_an_admin_exists(monkeypatch):
    if db.count_users("admin") == 0:
        first = auth.signup("First", _email("first"), "correct-horse", "admin")   # bootstrap: no code
        assert first["role"] == "admin"
    with pytest.raises(auth.AuthError, match="Wrong administrator code"):
        auth.signup("B", _email("admin"), "correct-horse", "admin", "guess")
    assert auth.signup("C", _email("admin"), "correct-horse", "admin", config.ADMIN_SIGNUP_CODE)["role"] == "admin"
    monkeypatch.setattr(config, "ADMIN_SIGNUP_CODE", "")
    with pytest.raises(auth.AuthError, match="closed"):
        auth.signup("D", _email("admin"), "correct-horse", "admin")
    assert auth.signup("E", _email("cli"), "correct-horse", "admin", trusted=True)["role"] == "admin"


def test_login_tokens(monkeypatch):
    email = _email("login")
    auth.signup("L", email, "correct-horse")
    with pytest.raises(auth.AuthError, match="Wrong e-mail or password"):
        auth.login(email, "nope")
    with pytest.raises(auth.AuthError, match="Wrong e-mail or password"):
        auth.login(_email("unknown"), "correct-horse")                  # same message: no account probing
    user, token = auth.login(email.upper(), "correct-horse")
    assert auth.user_for_token(token)["user_id"] == user["user_id"]
    with db.connect() as conn:                                          # only the hash of the token is stored
        assert conn.execute("SELECT count(*) AS n FROM auth_tokens WHERE token_hash=?", (token,)).fetchone()["n"] == 0
    auth.logout(token)
    assert auth.user_for_token(token) is None
    monkeypatch.setattr(config, "LOGIN_DAYS", -1)                       # expired login
    _, old = auth.login(email, "correct-horse")
    assert auth.user_for_token(old) is None


def test_roles_and_last_admin():
    a = auth.signup("Solo", _email("solo"), "correct-horse", "admin", trusted=True)
    u = auth.signup("U", _email("promote"), "correct-horse")
    assert auth.set_role(u["user_id"], "admin")["role"] == "admin"
    assert auth.set_role(u["user_id"], "user")["role"] == "user"
    with pytest.raises(LookupError):
        auth.set_role("USR-NOPE", "admin")
    admins = [x for x in db.list_users() if x["role"] == "admin"]
    for x in admins[:-1]:                                               # demote all but one ...
        auth.set_role(x["user_id"], "user")
    with pytest.raises(auth.AuthError, match="at least one administrator"):
        auth.set_role(admins[-1]["user_id"], "user")                    # ... the last one stays
    for x in admins[:-1]:
        auth.set_role(x["user_id"], "admin")
    assert a["role"] == "admin"


# =========================================================================== API access rules
def test_api_needs_a_login(api):
    assert api.get("/health").status_code == 200                        # public
    assert api.get("/courses").status_code == 200                       # public catalogue
    r = api.post("/chat", json={"session_id": "anon-1", "message": "hi"})
    assert r.status_code == 401 and r.json()["code"] == "not_authenticated"
    assert api.get("/curriculum").status_code == 401
    assert api.get("/curriculum", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_user_reaches_only_the_rex_assistant(api, user):
    _, h = user
    assert api.get("/auth/me", headers=h).json()["role"] == "user"
    assert api.post("/chat", headers=h, json={"session_id": f"u-{uuid.uuid4().hex[:6]}",
                                              "message": "I want a data course"}).status_code == 200
    for method, path in (("GET", "/curriculum"), ("GET", "/documents"), ("GET", "/campaigns"), ("GET", "/rfp"),
                         ("POST", "/rfp/run"), ("POST", "/route"), ("GET", "/auth/users"), ("POST", "/data/reload")):
        r = api.request(method, path, headers=h, json={})
        assert r.status_code == 403 and r.json()["code"] == "forbidden", path


def test_admin_reaches_every_agent(api, admin):
    _, h = admin
    for path in ("/curriculum", "/documents", "/campaigns", "/rfp", "/auth/users", "/documents/types"):
        assert api.get(path, headers=h).status_code == 200, path
    assert api.post("/chat", headers=h, json={"session_id": f"a-{uuid.uuid4().hex[:6]}",
                                              "message": "hello"}).status_code == 200


def test_each_user_has_their_own_history(api):
    sara, hs = _signup(api, name="Sara")
    omar, ho = _signup(api, name="Omar")
    sid = f"sara-{uuid.uuid4().hex[:6]}"
    api.post("/chat", headers=hs, json={"session_id": sid, "message": "I want to learn Python"})
    api.post("/chat", headers=hs, json={"session_id": sid, "message": "I am a beginner"})
    mine = api.get("/chat/sessions", headers=hs).json()
    assert [c["session_id"] for c in mine] == [sid] and mine[0]["title"] == "I want to learn Python"
    assert len(api.get(f"/chat/{sid}/history", headers=hs).json()["messages"]) == 4
    # Omar can neither see nor continue Sara's conversation
    assert api.get("/chat/sessions", headers=ho).json() == []
    assert api.get(f"/chat/{sid}/history", headers=ho).json() == {
        "session_id": sid, "found": False, "language": None, "stage": None, "messages": []}
    r = api.post("/chat", headers=ho, json={"session_id": sid, "message": "show me her profile"})
    assert r.status_code == 403
    assert len(db.list_messages(sid)) == 4
    assert db.session_owner(sid) == sara["user_id"] != omar["user_id"]


def test_admins_share_their_work(api):
    _, h1 = _signup(api, "admin", config.ADMIN_SIGNUP_CODE, name="Dean")
    _, h2 = _signup(api, "admin", config.ADMIN_SIGNUP_CODE, name="Mona")
    cur = api.post("/curriculum", headers=h1, json={"course_title": "Cloud Security Basics", "skill_level": "Beginner",
                                                    "target_audience": "IT staff", "total_hours": 24}).json()
    seen = api.get(f"/curriculum/{cur['curriculum_id']}", headers=h2)
    assert seen.status_code == 200 and seen.json()["course_title"] == "Cloud Security Basics"
    assert api.post(f"/curriculum/{cur['curriculum_id']}/review", headers=h2,
                    json={"action": "approved", "reviewer": "Mona"}).status_code == 200
    assert api.get(f"/curriculum/{cur['curriculum_id']}", headers=h1).json()["status"] == "approved"


def test_lead_is_linked_to_the_account(api, user):
    u, h = user
    r = api.post("/leads", headers=h, json={"session_id": "lead-1", "name": "Sara", "email": "sara@example.com",
                                            "consent": True,
                                            "interested_course_ids": ["DS-101"]})
    assert r.status_code == 201
    assert any(x["user_id"] == u["user_id"] for x in db.list_leads())


def test_signup_and_login_endpoints(api):
    email = _email("ep")
    assert api.post("/auth/signup", json={"name": "X", "email": email, "password": "short"}).status_code == 422
    assert api.post("/auth/signup", json={"name": "X", "email": email, "password": "correct-horse",
                                          "role": "admin", "admin_code": "wrong"}).json()["code"] == "signup_failed"
    assert api.post("/auth/signup", json={"name": "X", "email": email, "password": "correct-horse"}).status_code == 201
    assert api.post("/auth/login", json={"email": email, "password": "bad"}).status_code == 401
    r = api.post("/auth/login", json={"email": email, "password": "correct-horse"})
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    assert api.post("/auth/logout", headers=h).status_code == 200
    assert api.get("/auth/me", headers=h).status_code == 401


def test_admin_changes_roles(api, admin, user):
    _, ha = admin
    u, hu = user
    assert api.patch(f"/auth/users/{u['user_id']}", headers=hu, json={"role": "admin"}).status_code == 403
    r = api.patch(f"/auth/users/{u['user_id']}", headers=ha, json={"role": "admin"})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert api.get("/curriculum", headers=hu).status_code == 200        # the same login now reaches the agents
    assert all("password_hash" not in x for x in api.get("/auth/users", headers=ha).json())
