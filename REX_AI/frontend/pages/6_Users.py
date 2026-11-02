"""Accounts (administrators only, owner: Member 3): who can use REX AI and with which role.

A user uses the REX Course Assistant; an administrator uses every agent. Promote a user here to make them an
administrator (the last administrator cannot be demoted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, api_up, no_fog, require  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX - Users", page_icon="👥", layout="wide")
no_fog()
ME = require("admin")


def users() -> list[dict]:
    if api_up():
        r = api().get(f"{config.API_URL}/auth/users", timeout=10)
        r.raise_for_status()
        return r.json()
    from rex_ai.core import db
    return db.list_users()


def set_role(user_id: str, role: str) -> dict:
    if api_up():
        r = api().patch(f"{config.API_URL}/auth/users/{user_id}", json={"role": role}, timeout=10)
        if r.status_code != 200:
            raise ValueError(r.json().get("message", r.text))
        return r.json()
    from rex_ai.core import auth
    return auth.set_role(user_id, role)


st.title("👥 Users")
st.caption("Users chat with the REX Course Assistant and see only their own conversations. Administrators use "
           "every agent and share curricula, documents, campaigns and tenders with each other.")
if note := st.session_state.pop("users_notice", None):
    st.success(note)

rows = users()
st.dataframe(pd.DataFrame([{"Name": u["name"], "E-mail": u["email"],
                            "Role": "Administrator" if u["role"] == "admin" else "User",
                            "Created": (u.get("created_at") or "")[:16].replace("T", " "),
                            "Last login": (u.get("last_login_at") or "")[:16].replace("T", " ")} for u in rows]),
             hide_index=True)

with st.form("role"):
    st.markdown("**Change a role**")
    labels = {u["user_id"]: f"{u['name']} · {u['email']} ({u['role']})" for u in rows}
    who = st.selectbox("Account", list(labels), format_func=labels.get)
    role = st.radio("Role", ["user", "admin"], horizontal=True,
                    format_func={"user": "User", "admin": "Administrator"}.get)
    if st.form_submit_button("Save", type="primary"):
        try:
            u = set_role(who, role)
            st.session_state.users_notice = f"{u['name']} is now {'an administrator' if u['role'] == 'admin' else 'a user'}."
            if who == ME["user_id"] and role != "admin":
                st.session_state.auth_user = {**ME, "role": role}
            st.rerun()
        except (ValueError, LookupError, httpx.HTTPError) as exc:
            st.error(str(exc))
