"""REX Course Assistant - Streamlit chat UI (owner: Member 1; app shell later merged by Member 3).

Run:  streamlit run frontend/streamlit_app.py
Talks to the FastAPI backend at REX_API_URL. If the API is not running, it falls
back to calling the assistant in-process so the demo still works.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontend.ui import api, no_fog  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX Course Assistant", page_icon="🎓", layout="centered")
no_fog()

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


def api_available() -> bool:
    try:
        return api().get(f"{config.API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def send_chat(message: str, language: str | None) -> dict:
    payload = {"session_id": st.session_state.session_id, "message": message}
    if language:
        payload["language"] = language
    if st.session_state.api_mode:
        r = api().post(f"{config.API_URL}/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    from rex_ai.core.schemas import ChatRequest
    return _local_assistant().handle(ChatRequest(**payload)).model_dump(mode="json")


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
    return True, db.save_lead(lead.model_dump(mode="json"))["lead_id"]


# -------------------------------------------------------------------- state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []
    st.session_state.last = None
    st.session_state.api_mode = api_available()


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
    if st.button("New conversation"):
        for key in ("session_id", "messages", "last"):
            st.session_state.pop(key, None)
        st.rerun()

st.title("🎓 REX Course Assistant")
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
