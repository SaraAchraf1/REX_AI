"""REX AI web app: log in, then the pages your account may use (owner: Member 3).

Run:  streamlit run frontend/streamlit_app.py

    not logged in   Log in / Sign up
    user            REX Course Assistant (own conversations)
    administrator   REX Course Assistant + RFP Research, Curriculum, Documents, Marketing, Orchestrator Tasks,
                    Orchestrator & Tools, Users
                    (everything administrators create is shared with the other administrators)

Pages an account may not use are not in its menu, and every page checks the role again (frontend/ui.require).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontend.ui import LOGO, current_user, login_page, logout, no_fog, sync_cookie  # noqa: E402

st.set_page_config(page_title="REX AI · Andalusia Academy", page_icon=LOGO, layout="wide")
no_fog()

# (file, menu title, icon, url path, menu group); the first page is the home page of every account
CHAT = ("pages/1_REX_Assistant.py", "REX Course Assistant", "🎓", "chat", "REX AI")
ADMIN_PAGES = [
    ("pages/2_RFP_Research.py", "RFP & Tender Research", "📑", "rfp", "Agents (shared by administrators)"),
    ("pages/3_Curriculum.py", "Curriculum & Accreditation", "📚", "curriculum", "Agents (shared by administrators)"),
    ("pages/4_Documents.py", "Documents", "📄", "documents", "Agents (shared by administrators)"),
    ("pages/5_Marketing.py", "Marketing", "📣", "marketing", "Agents (shared by administrators)"),
    ("pages/8_Tasks.py", "Orchestrator Tasks", "🤖", "tasks", "Agents (shared by administrators)"),
    ("pages/7_Orchestrator.py", "Orchestrator & Tools", "🧭", "orchestrator", "Administration"),
    ("pages/6_Users.py", "Users", "👥", "users", "Administration"),
]

@st.cache_resource
def _warm_imports() -> None:
    """Load the heavy libraries in the background once, while the first page is shown, instead of making the first
    visit to a page that needs them wait (pandas alone takes ~1 s)."""
    import importlib
    import threading

    def load():
        for name in ("pandas", "rex_ai.core.schemas", "rex_ai.core.db", "rex_ai.agents.rfp.taxonomy"):
            try:
                importlib.import_module(name)
            except Exception:
                pass
    threading.Thread(target=load, name="rex-warm-imports", daemon=True).start()


_warm_imports()
user = current_user()
sync_cookie()

if not user:
    # the login form answers on every page address, so a bookmarked page (or the page open when logging out)
    # shows it - and after logging in, that page opens if the account may use it
    nav = st.navigation([st.Page(login_page, title="Log in", icon="🔐", url_path="login", default=True)] +
                       [st.Page(login_page, title="Log in", icon="🔐", url_path=p[3]) for p in [CHAT, *ADMIN_PAGES]],
                       position="hidden")
else:
    pages: dict[str, list] = {}
    for file, title, icon, path, group in [CHAT] + (ADMIN_PAGES if user["role"] == "admin" else []):
        pages.setdefault(group, []).append(st.Page(file, title=title, icon=icon, url_path=path,
                                                   default=file == CHAT[0]))
    nav = st.navigation(pages)
    with st.sidebar:
        role = "Administrator" if user["role"] == "admin" else "User"
        st.markdown(f"👤 **{user['name']}**  \n{user['email']} · {role}")
        st.button("Log out", key="logout_button", on_click=logout)     # runs before the next page is drawn
        st.divider()

nav.run()
