"""Shared helpers for the Streamlit pages (owner: Member 3; used by every page).

    from frontend.ui import api, no_fog, require
    no_fog()                                   # right after st.set_page_config
    user = require("admin")                    # stops the page unless an administrator is logged in
    api().get(f"{config.API_URL}/health")      # instead of httpx.get(...); sends the login token

Accounts (rex_ai/core/auth.py): the login is kept in the browser session and in a cookie (rex_token, valid for
REX_LOGIN_DAYS), so reloading the page keeps you logged in. Works with the API (POST /auth/login...) or, when the
API is not running, directly on the database.
"""
from __future__ import annotations

from typing import Optional

import httpx
import streamlit as st

from rex_ai import config

COOKIE = "rex_token"
ADMIN_REVIEWER_KEYS = ("reviewer_decision", "reviewer_modify", "doc_reviewer_edit", "doc_reviewer_instr",
                       "doc_reviewer_decision", "mkt_reviewer_edit", "mkt_reviewer_instr", "mkt_reviewer_decision",
                       "rfp_reviewer")


@st.cache_resource
def _http() -> httpx.Client:
    """One HTTP client for the whole app. httpx.get() builds a new client - and its SSL context - on every call
    (~350 ms each on Windows); a shared client answers local API calls in a few milliseconds."""
    return httpx.Client(timeout=60)


class _Client:
    """The shared client, plus this browser session's login token (never stored in the shared client)."""

    def request(self, method: str, url: str, **kw) -> httpx.Response:
        headers = dict(kw.pop("headers", None) or {})
        if token := st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {token}"
        return _http().request(method, url, headers=headers, **kw)

    def get(self, url: str, **kw) -> httpx.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> httpx.Response:
        return self.request("POST", url, **kw)

    def patch(self, url: str, **kw) -> httpx.Response:
        return self.request("PATCH", url, **kw)

    def delete(self, url: str, **kw) -> httpx.Response:
        return self.request("DELETE", url, **kw)


def api() -> _Client:
    return _Client()


def no_fog() -> None:
    """While a page reruns, Streamlit dims ("fogs") everything already on screen. Keep it fully visible: actions
    that take time show their own spinner instead."""
    st.markdown("<style>[data-stale='true'], .stale-element {opacity: 1 !important; transition: none !important;}"
                "</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ accounts
def api_up() -> bool:
    if "rex_api_up" not in st.session_state:
        try:
            st.session_state.rex_api_up = _http().get(f"{config.API_URL}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            st.session_state.rex_api_up = False
    return st.session_state.rex_api_up


@st.cache_resource
def _local_db() -> bool:
    from rex_ai.core import db
    db.init_db(seed=False)
    return True


def _api_error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text
    if body.get("details", {}).get("errors"):
        return "; ".join(e.get("msg", "") for e in body["details"]["errors"])
    return body.get("message") or body.get("detail") or r.text


def _login_call(email: str, password: str) -> tuple[dict, str]:
    if api_up():
        r = _http().post(f"{config.API_URL}/auth/login", json={"email": email, "password": password}, timeout=20)
        if r.status_code != 200:
            raise ValueError(_api_error(r))
        return r.json()["user"], r.json()["token"]
    from rex_ai.core import auth
    _local_db()
    return auth.login(email, password)


def _signup_call(name: str, email: str, password: str, role: str, code: str) -> tuple[dict, str]:
    if api_up():
        r = _http().post(f"{config.API_URL}/auth/signup", timeout=20, json={
            "name": name, "email": email, "password": password, "role": role, "admin_code": code or None})
        if r.status_code != 201:
            raise ValueError(_api_error(r))
        return r.json()["user"], r.json()["token"]
    from rex_ai.core import auth
    _local_db()
    auth.signup(name, email, password, role, code)
    return auth.login(email, password)


def _user_for(token: str) -> Optional[dict]:
    try:
        if api_up():
            r = _http().get(f"{config.API_URL}/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
            return r.json() if r.status_code == 200 else None
        from rex_ai.core import auth
        _local_db()
        return auth.user_for_token(token)
    except (httpx.HTTPError, ValueError):
        return None


def _start(user: dict, token: str) -> None:
    st.session_state.auth_user, st.session_state.auth_token = user, token
    st.session_state.cookie_todo = token                   # written to the browser on the next run
    st.session_state.pop("logged_out", None)
    if user["role"] == "admin":                            # reviewer fields default to the administrator's name
        for key in ADMIN_REVIEWER_KEYS:
            st.session_state.setdefault(key, user["name"])


def current_user() -> Optional[dict]:
    """The logged-in account, restored from the browser cookie after a page reload."""
    if st.session_state.get("auth_user"):
        return st.session_state.auth_user
    if st.session_state.get("logged_out"):
        return None
    try:
        token = st.context.cookies.get(COOKIE)
    except Exception:                                      # no browser context (scripts, tests)
        token = None
    if isinstance(token, str) and token and (user := _user_for(token)):
        _start(user, token)
        st.session_state.pop("cookie_todo", None)          # the browser already has it
        return user
    return None


def sync_cookie() -> None:
    """Store or clear the login cookie in the browser (a zero-height component; runs once after login/logout)."""
    todo = st.session_state.pop("cookie_todo", None)
    if todo is None:
        return
    age = int(config.LOGIN_DAYS * 86400) if todo else 0
    # our own token (URL-safe characters only), never user input
    st.iframe(f"<script>parent.document.cookie = '{COOKIE}={todo}; path=/; max-age={age}; SameSite=Strict';"
              "</script>", height=1)


def logout() -> None:
    token = st.session_state.get("auth_token")
    try:
        if api_up():
            api().post(f"{config.API_URL}/auth/logout", timeout=10)
        else:
            from rex_ai.core import auth
            auth.logout(token)
    except httpx.HTTPError:
        pass
    # forget the account and its conversation (the chat page also reloads when another account logs in)
    for key in ("auth_user", "auth_token", "chat_owner", "session_id", "messages", "last"):
        st.session_state.pop(key, None)
    st.session_state.logged_out = True
    st.session_state.cookie_todo = ""                      # expire the browser cookie


def require(role: str = "user") -> dict:
    """The logged-in account; stops the page if nobody is logged in or the account lacks the role."""
    user = current_user()
    if not user:
        st.warning("Please log in first.")
        st.stop()
    if role == "admin" and user["role"] != "admin":
        st.error("This page is for administrators. Your account can use the REX Course Assistant.")
        st.stop()
    return user


def login_page() -> None:
    """Log in or create an account (user = REX Course Assistant; administrator = every agent)."""
    st.markdown("<h1 style='text-align:center'>🎓 REX AI</h1><p style='text-align:center'>Andalusia Academy · "
                "Smart Academy Multi-Agent System</p>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Log in", "Sign up"])
        with tab_in, st.form("login"):
            email = st.text_input("E-mail", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                try:
                    _start(*_login_call(email, password))
                    st.rerun()
                except (ValueError, LookupError, httpx.HTTPError) as exc:
                    st.error(str(exc))
        with tab_up, st.form("signup"):
            name = st.text_input("Full name", key="signup_name")
            email = st.text_input("E-mail", key="signup_email")
            password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
            confirm = st.text_input("Repeat the password", type="password", key="signup_confirm")
            role = st.radio("Account type", ["user", "admin"], horizontal=True, key="signup_role",
                            format_func={"user": "User - REX Course Assistant",
                                         "admin": "Administrator - all agents"}.get)
            code = st.text_input("Administrator code (administrators only)", type="password", key="signup_code",
                                 help="Given by the academy. Not needed for the very first administrator.")
            if st.form_submit_button("Create account", type="primary", use_container_width=True):
                if password != confirm:
                    st.error("The two passwords are different.")
                else:
                    try:
                        _start(*_signup_call(name, email, password, role, code))
                        st.rerun()
                    except (ValueError, LookupError, httpx.HTTPError) as exc:
                        st.error(str(exc).splitlines()[-1])
        st.caption("Users chat with the REX Course Assistant and see only their own conversations. Administrators "
                   "use every agent and share their work with the other administrators.")
