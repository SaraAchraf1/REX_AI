"""REX Course Assistant - Streamlit chat UI (owner: Member 1; app shell later merged by Member 3).

Run:  streamlit run frontend/streamlit_app.py      (log in; this is the first page for every account)
Talks to the FastAPI backend at REX_API_URL. If the API is not running, it falls
back to calling the assistant in-process so the demo still works.

Every conversation is stored under the logged-in account: the sidebar lists that account's conversations, and
another account's conversation cannot be opened or continued.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, api_up, no_fog, require  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX Course Assistant", page_icon="🎓", layout="centered")
no_fog()
USER = require("user")                           # users and administrators

FORMAT = {"online_live": "Live online", "onsite": "On-site", "hybrid": "Hybrid"}


# ------------------------------------------------------------------ backend
def _local_assistant():
    # Cached inside data_sync and rebuilt whenever files in data/ are added or changed.
    from rex_ai.core import data_sync
    return data_sync.get_assistant()


def upload_data(name: str, content: bytes) -> dict:
    """Course catalogues are merged into the course data; other documents are indexed for answers."""
    if st.session_state.api_mode:
        r = api().post(f"{config.API_URL}/data/upload", files={"file": (name, content)}, timeout=600)
        if r.status_code != 200:
            raise ValueError(r.json().get("message", r.text))
        return r.json()
    from rex_ai.core import data_sync
    return data_sync.save_upload(name, content)


def send_chat(message: str, language: str | None) -> dict:
    payload = {"session_id": st.session_state.session_id, "message": message}
    if language:
        payload["language"] = language
    if st.session_state.api_mode:
        r = api().post(f"{config.API_URL}/chat", json=payload, timeout=60)
        if r.status_code == 403:
            raise PermissionError(r.json().get("message"))
        r.raise_for_status()
        return r.json()
    from rex_ai.core import db
    from rex_ai.core.schemas import ChatRequest
    assistant = _local_assistant()
    if not db.claim_session(payload["session_id"], USER["user_id"]):
        raise PermissionError("This conversation belongs to another account")
    return assistant.handle(ChatRequest(**payload)).model_dump(mode="json")


def save_lead(payload: dict) -> tuple[bool, str]:
    if st.session_state.api_mode:
        r = api().post(f"{config.API_URL}/leads", json=payload, timeout=10)
        if r.status_code == 201:
            return True, r.json()["lead_id"]
        detail = r.json().get("details", {}).get("errors", [{}])
        return False, "; ".join(e.get("msg", "") for e in detail) or r.text
    from rex_ai.core import db
    from rex_ai.core.schemas import LeadCreate
    try:
        lead = LeadCreate(**payload)
    except Exception as exc:  # pydantic ValidationError
        return False, str(exc).splitlines()[-1]
    return True, db.save_lead({**lead.model_dump(mode="json"), "user_id": USER["user_id"]})["lead_id"]


def load_history(session_id: str) -> list[dict]:
    """A conversation stored in the database (the shared one when REX_DATABASE_URL is set)."""
    if st.session_state.api_mode:
        r = api().get(f"{config.API_URL}/chat/{session_id}/history", timeout=10)
        r.raise_for_status()
        rows = r.json()["messages"]
    else:
        from rex_ai.core import db
        _local_assistant()                           # DB initialised
        owner = db.session_owner(session_id)
        rows = db.list_messages(session_id) if owner == USER["user_id"] else []
    return [{"role": m["role"], "content": m["content"], "sources": m.get("sources") or [],
             "lang": "ar" if re.search("[؀-ۿ]", m["content"]) else "en"} for m in rows]


def my_conversations() -> list[dict]:
    """This account's conversations, newest first."""
    if st.session_state.api_mode:
        r = api().get(f"{config.API_URL}/chat/sessions", timeout=10)
        return r.json() if r.status_code == 200 else []
    from rex_ai.core import db
    _local_assistant()
    return db.list_sessions(USER["user_id"])


def open_conversation(sid: str | None) -> None:
    """Show a stored conversation of this account (None = start a new one). Only the owner gets its messages."""
    st.session_state.last = None
    st.session_state.messages = []
    if sid and re.fullmatch(r"[\w-]{4,64}", sid):
        try:
            st.session_state.messages = load_history(sid)
        except (httpx.HTTPError, LookupError, ValueError):
            pass
        if not st.session_state.messages:              # not this account's (or empty): start fresh
            sid = None
    st.session_state.session_id = sid or uuid.uuid4().hex[:12]
    st.query_params["chat"] = st.session_state.session_id


# -------------------------------------------------------------------- state
# The conversation id is kept in the page address (?chat=...): reopening that link - on this or any other server
# using the same database - shows the conversation again, for the account that owns it.
if st.session_state.get("chat_owner") != USER["user_id"]:     # first run, or another account logged in
    st.session_state.chat_owner = USER["user_id"]
    st.session_state.api_mode = api_up()
    st.session_state.pop("session_id", None)
if "session_id" not in st.session_state:
    open_conversation(st.query_params.get("chat"))


def render_text(text: str, lang: str) -> None:
    # Keep the assistant's single line breaks (Markdown would merge them into one paragraph).
    text = re.sub(r"(?<!\n)\n(?!\n)", "  \n", text)
    if lang == "ar":
        st.markdown(f"<div dir='rtl' style='text-align:right'>\n\n{text}\n\n</div>", unsafe_allow_html=True)
    else:
        st.markdown(text)


def render_recs(recs: list[dict], lang: str) -> None:
    cols = st.columns(len(recs))
    for col, r in zip(cols, recs):
        with col.container(border=True):
            st.caption(r["course_id"])
            st.markdown(f"**{r['title']}**")
            st.markdown(f"{r['fee']:,.0f} {r['currency']}" if r.get("fee") else "")
            st.caption(" · ".join(FORMAT[f] for f in r["format_options"]))
            if r.get("next_start"):
                st.caption(("أقرب بداية: " if lang == "ar" else "Next start: ") + r["next_start"])
            st.progress(min(1.0, r["score"]), text=f"{r['score']:.0%} " + ("توافق" if lang == "ar" else "match"))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("REX AI")
    st.caption("Course Recommendation Assistant · Sprint 1 prototype")
    lang_choice = st.radio("Reply language", ["Auto", "English", "العربية"], horizontal=True)
    forced = {"Auto": None, "English": "en", "العربية": "ar"}[lang_choice]
    st.caption(("Connected to API " + config.API_URL) if st.session_state.api_mode else "API not running: in-process mode")
    if st.button("➕ New conversation", use_container_width=True):
        open_conversation(None)
        st.rerun()
    chats = my_conversations()
    if chats:
        st.markdown("**Your conversations**")
        for c in chats[:20]:
            title = (c.get("title") or "…").strip()
            label = (title[:38] + "…" if len(title) > 40 else title) + f"  ·  {c['updated_at'][:10]}"
            current = c["session_id"] == st.session_state.session_id
            if st.button(label, key=f"conv_{c['session_id']}", use_container_width=True,
                         type="primary" if current else "secondary", disabled=current):
                open_conversation(c["session_id"])
                st.rerun()
    if USER["role"] == "admin":                  # academy data is managed by administrators
        with st.expander("Add academy data"):
            st.caption("PDF, Word, Excel, CSV or TXT. Course catalogues are added to the courses; other documents "
                       "become answers. Nothing already loaded is removed.")
            uploads = st.file_uploader("Files", type=["pdf", "docx", "xlsx", "xlsm", "xls", "csv", "txt", "md"],
                                       accept_multiple_files=True, label_visibility="collapsed")
            if uploads and st.button("Load into REX"):
                for up in uploads:
                    try:
                        with st.spinner(f"Loading {up.name}..."):
                            res = upload_data(up.name, up.getvalue())
                    except (ValueError, httpx.HTTPError) as exc:
                        st.error(f"{up.name}: {exc}")
                        continue
                    if res["status"] == "duplicate":
                        st.info(f"{up.name}: already loaded as {res['file']}")
                    elif res["kind"] == "catalog":
                        st.success(f"{up.name}: {len(res['added'])} course(s) added, {len(res['updated'])} updated")
                    else:
                        st.success(f"{up.name}: saved as {res['file']}; REX can now answer from it")
                    for e in res.get("errors", []):
                        st.warning(e)
    if st.session_state.last:
        st.subheader("What REX understood")
        prof = st.session_state.last["profile"]
        shown = {k: v for k, v in prof.items() if v not in (None, [], "") and k != "language"}
        st.json(shown or {"profile": "nothing yet"})
        st.caption(f"Intent: {st.session_state.last['intent']} · Stage: {st.session_state.last['stage']}"
                   f" · LLM: {'on' if st.session_state.last['llm_used'] else 'off'}")

st.title("🎓 REX Course Assistant")
st.caption(f"Hello {USER['name']} - your conversations are saved to your account.")
st.info("Prototype running on SAMPLE course data. Prices, dates and instructors are placeholders until the "
        "academy's approved catalogue is loaded.", icon="ℹ️")

# ------------------------------------------------------------------ history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_text(m["content"], m.get("lang", "en"))
        if m.get("recs"):
            render_recs(m["recs"], m.get("lang", "en"))
        if m.get("sources"):
            with st.expander("Sources" if m.get("lang") != "ar" else "المصادر"):
                for s in m["sources"]:
                    st.caption(f"{s['source_id']} · {s['title']}" + (f" · {s['record_id']}" if s.get("record_id") else ""))

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hi! Say hello, or tell me what you'd like to learn.  \n"
                    "<div dir='rtl'>أهلًا! قل مرحبًا أو أخبرني ماذا تريد أن تتعلم.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------- input
if prompt := st.chat_input("Type your message / اكتب رسالتك"):
    st.session_state.messages.append({"role": "user", "content": prompt, "lang": "ar" if forced == "ar" else "en"})
    try:
        resp = send_chat(prompt, forced)
    except PermissionError as exc:
        st.error(str(exc))
        open_conversation(None)
        st.stop()
    except httpx.HTTPError as exc:
        st.error(f"Could not reach REX: {exc}")
        st.stop()
    st.session_state.last = resp
    st.session_state.messages.append({"role": "assistant", "content": resp["reply"], "lang": resp["language"],
                                      "recs": resp["recommendations"], "sources": resp["sources"]})
    st.rerun()

# -------------------------------------------------------------- lead capture
last = st.session_state.last
if last and last.get("offer_lead_capture"):
    ar = last["language"] == "ar"
    with st.expander("اترك بيانات التواصل (اختياري)" if ar else "Leave your contact details (optional)"):
        with st.form("lead_form", clear_on_submit=True):
            name = st.text_input("الاسم" if ar else "Name")
            email = st.text_input("البريد الإلكتروني" if ar else "Email")
            phone = st.text_input("رقم الهاتف" if ar else "Phone")
            course_ids = [r["course_id"] for r in last.get("recommendations", [])] or []
            interested = st.multiselect("الدورات المهتم بها" if ar else "Courses you're interested in",
                                        options=course_ids, default=course_ids[:1])
            consent = st.checkbox(
                "أوافق على حفظ بياناتي واستخدامها فقط للتواصل معي بخصوص الدورات" if ar else
                "I agree that the academy may save these details and contact me about courses only.")
            if st.form_submit_button("إرسال" if ar else "Send"):
                if not consent:
                    st.warning("لن نحفظ بياناتك بدون موافقتك." if ar else "We can't save your details without your consent.")
                else:
                    ok, msg = save_lead({"session_id": st.session_state.session_id, "name": name,
                                         "email": email or None, "phone": phone or None,
                                         "interested_course_ids": interested, "profile": last["profile"],
                                         "consent": True})
                    if ok:
                        st.success(("تم الحفظ. سيتواصل معك فريق القبول. رقم المرجع: " if ar else
                                    "Saved. Our admissions team will contact you. Reference: ") + msg)
                    else:
                        st.error(msg)
