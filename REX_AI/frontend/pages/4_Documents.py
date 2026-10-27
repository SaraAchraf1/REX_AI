"""Educational Content & Document Generation (owner: Member 3). Fourth page of the Streamlit app:

    streamlit run frontend/streamlit_app.py      -> sidebar: "Documents"

Talks to the API when it is running, otherwise runs the agent in-process (same as the other pages). Same speed
rules as the Curriculum page: inputs in forms, cached reads cleared by changed(), PDF fetched only on request, only
the visible tab drawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, no_fog  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX - Documents", page_icon="📄", layout="wide")
no_fog()

STATUS = {"draft": "📝 draft", "pending_approval": "⏳ pending approval", "changes_requested": "✏️ changes requested",
          "approved": "✅ approved", "rejected": "⛔ rejected"}
LIBRARY_TAB, VIEW_TAB = "Documents", "Selected document"
ERRORS = (LookupError, ValueError, httpx.HTTPError)


# ------------------------------------------------------------------ backend
def api_available() -> bool:
    try:
        return api().get(f"{config.API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if "doc_api" not in st.session_state:
    st.session_state.doc_api = api_available()
API = st.session_state.doc_api


@st.cache_resource
def local_agent():
    from rex_ai.agents.documents.agent import DocumentAgent
    from rex_ai.core import data_sync
    data_sync.get_assistant()                    # DB initialised, catalogue loaded
    return DocumentAgent()


def _call(method: str, path: str, **kw):
    r = api().request(method, f"{config.API_URL}{path}", timeout=600, **kw)
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        raise ValueError(body.get("message") or body.get("detail") or r.text)
    return r


@st.cache_data(ttl=600, show_spinner=False)
def doc_types() -> list[dict]:
    if API:
        return _call("GET", "/documents/types").json()
    from rex_ai.agents.documents.agent import DocumentAgent
    return [t.model_dump(mode="json") for t in DocumentAgent.types()]


@st.cache_data(ttl=600, show_spinner=False)
def courses() -> dict[str, str]:
    if API:
        rows = _call("GET", "/courses").json()
    else:
        local_agent()
        from rex_ai.agents.rex.knowledge_base import get_kb
        rows = [c.model_dump(mode="json") for c in get_kb().courses.values()]
    return {c["course_id"]: c["title_en"] for c in rows}


@st.cache_data(ttl=60, show_spinner=False)
def curricula() -> list[dict]:
    if API:
        rows = _call("GET", "/curriculum").json()
    else:
        local_agent()
        from rex_ai.agents.curriculum.agent import CurriculumAgent
        rows = [c.model_dump(mode="json") for c in CurriculumAgent().search()]
    return [{"id": c["curriculum_id"], "title": c["course_title"], "status": c["status"], "version": c["version"],
             "language": c.get("language")} for c in rows]


@st.cache_data(ttl=60, show_spinner=False)
def list_documents(params: tuple) -> list[dict]:
    params = {k: v for k, v in params if v}
    if API:
        return _call("GET", "/documents", params=params).json()
    return [d.model_dump(mode="json") for d in local_agent().search(**params)]


@st.cache_data(ttl=60, show_spinner=False)
def get_document(did: str) -> dict:
    if API:
        return _call("GET", f"/documents/{did}").json()
    return local_agent().get(did).model_dump(mode="json")


@st.cache_data(ttl=60, show_spinner=False)
def history(did: str) -> dict:
    if API:
        return _call("GET", f"/documents/{did}/history").json()
    return local_agent().history(did)


@st.cache_data(ttl=60, show_spinner=False)
def handoffs() -> list[dict]:
    if API:
        return _call("GET", "/documents/handoffs").json()
    return local_agent().pending_handoffs()


def changed() -> None:
    for fn in (list_documents, get_document, history, handoffs, curricula):
        fn.clear()
    st.session_state.pop("doc_pdf", None)


# writes
def generate(payload: dict) -> dict:
    if API:
        return _call("POST", "/documents", json=payload).json()
    from rex_ai.core.schemas import DocumentRequest
    return local_agent().generate(DocumentRequest(**payload)).model_dump(mode="json")


def edit(did: str, payload: dict) -> dict:
    if API:
        return _call("PATCH", f"/documents/{did}", json=payload).json()
    from rex_ai.core.schemas import DocumentEdit
    return local_agent().edit(did, DocumentEdit(**payload)).model_dump(mode="json")


def review(did: str, action: str, reviewer: str, comment: str) -> dict:
    payload = {"action": action, "reviewer": reviewer, "comment": comment or None}
    if API:
        return _call("POST", f"/documents/{did}/review", json=payload).json()
    from rex_ai.core.schemas import ReviewAction
    return local_agent().review(did, ReviewAction(**payload)).model_dump(mode="json")


def email(did: str, to: list[str], sender: str, message: str) -> dict:
    payload = {"to": to, "sender": sender, "message": message or None}
    if API:
        return _call("POST", f"/documents/{did}/email", json=payload).json()
    from rex_ai.core.schemas import DocumentEmail
    return local_agent().email(did, DocumentEmail(**payload)).model_dump(mode="json")


def pdf_bytes(did: str) -> tuple[str, bytes]:
    if API:
        r = _call("GET", f"/documents/{did}/download")
        name = r.headers.get("content-disposition", "").split("filename=")[-1].strip('"') or f"{did}.pdf"
        return name, r.content
    path = local_agent().pdf(did)
    return path.name, path.read_bytes()


def open_document(did: str) -> None:
    """Select a document and switch to its tab (must run before st.tabs is drawn - button callback or sidebar)."""
    st.session_state.doc_selected = did
    st.session_state.doc_tab = VIEW_TAB


def notice(kind: str, text: str, subtab: str | None = None) -> None:
    st.session_state.doc_notice = (kind, text)
    if subtab:
        st.session_state.doc_subtab_next = subtab


def create(types: list[str], base: dict) -> list[dict]:
    made = []
    for t in types:
        made.append(generate({**base, "doc_type": t}))
    return made


# ------------------------------------------------------------------ sidebar: discovery questions (brief §10)
TYPES = {t["doc_type"]: t for t in doc_types()}
catalogue = courses()

with st.sidebar:
    st.header("Create a document")
    st.caption(("Connected to API " + config.API_URL) if API else "API not running: in-process mode")
    source = st.radio("Which course or curriculum?", ["Curriculum", "Catalogue course"], horizontal=True)
    with st.form("create", border=False):
        payload: dict = {}
        if source == "Curriculum":
            curs = curricula()
            if curs:
                labels = {c["id"]: f"{STATUS.get(c['status'], c['status'])} · {c['title']} (v{c['version']})"
                          for c in curs}
                payload["curriculum_id"] = st.selectbox("Curriculum", list(labels), format_func=labels.get)
                st.caption("Approved curricula give approvable documents; others can be drafted and reviewed.")
            else:
                st.info("No curricula yet - create one on the Curriculum page, or use a catalogue course.")
        else:
            payload["course_id"] = st.selectbox("Course", list(catalogue), format_func=lambda i: f"{i} · {catalogue[i]}")
            st.caption("Uses the course's newest approved curriculum, else the catalogue data.")
        types = st.multiselect("Which document types?", list(TYPES), default=["course_description"],
                               format_func=lambda t: TYPES[t]["title"])
        audience = st.text_input("Who is it prepared for? (optional)", placeholder="Ministry of Health training unit")
        language = st.radio("Language", ["auto", "en", "ar"], horizontal=True,
                            format_func={"auto": "Same as source", "en": "English", "ar": "العربية"}.get)
        wishes = st.text_area("Extra wishes for the LLM (optional)", placeholder="Use banking examples")
        email_to = st.text_input("E-mail after approval (optional, comma separated)")
        go = st.form_submit_button("Generate", type="primary")

    if go:
        if not types:
            st.warning("Choose at least one document type.")
        elif not (payload.get("curriculum_id") or payload.get("course_id")):
            st.warning("Choose a curriculum or a course.")
        else:
            base = {**payload, "audience": audience.strip() or None, "instructions": wishes.strip() or None,
                    "email_to": [e.strip() for e in email_to.split(",") if e.strip()] or None}
            if language != "auto":
                base["language"] = language
            with st.spinner(f"Collecting course data, searching references and writing {len(types)} document(s)..."):
                try:
                    made = create(types, {k: v for k, v in base.items() if v is not None})
                    changed()
                    open_document(made[-1]["document_id"])
                    notice("success", "Created: " + ", ".join(f"{d['document_id']} ({TYPES[d['doc_type']]['title']})"
                                                              for d in made))
                except ERRORS as exc:
                    st.error(str(exc))

# ------------------------------------------------------------------ header
st.title("📄 Educational Content & Document Generation")
st.info("Documents are built from approved curricula and catalogue data with the academy templates. Nothing is "
        "e-mailed before a reviewer approves it, and no document claims accreditation. Templates and reference "
        "examples are SAMPLE data until the academy sends its own.", icon="ℹ️")

try:
    waiting = [h for h in handoffs() if h["curriculum_status"] == "approved"]
except ERRORS:
    waiting = []
if waiting:
    with st.expander(f"Approved curricula handed over by the Curriculum Agent ({len(waiting)})",
                     expanded=any(h["documents"] == 0 for h in waiting)):
        for h in waiting:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            made = ", ".join(TYPES[t]["title"] for t in h["doc_types"] if t in TYPES) or "no documents yet"
            c1.write(f"**{h['course_title']}** · `{h['curriculum_id']}` v{h['version']} · {made}")
            if c2.button("Create pack", key=f"pack_{h['curriculum_id']}"):
                with st.spinner("Creating course description, participant guide and question bank..."):
                    try:
                        made_docs = create(["course_description", "participant_guide", "question_bank"],
                                           {"curriculum_id": h["curriculum_id"]})
                        changed()
                        notice("success", f"{len(made_docs)} documents created for {h['course_title']}")
                        st.session_state.doc_selected = made_docs[0]["document_id"]
                        st.session_state.doc_tab_next = VIEW_TAB
                        st.rerun()
                    except ERRORS as exc:
                        st.error(str(exc))

if nxt := st.session_state.pop("doc_tab_next", None):
    st.session_state.doc_tab = nxt
tab_lib, tab_view = st.tabs([LIBRARY_TAB, VIEW_TAB], key="doc_tab", on_change="rerun")

# ------------------------------------------------------------------ library
if tab_lib.open is not False:
    with tab_lib:
        with st.form("doc_filters", border=False):
            f = st.columns([2, 1, 1, 1, 2, 1], vertical_alignment="bottom")
            flt = (("doc_type", f[0].selectbox("Type", [""] + list(TYPES),
                                               format_func=lambda x: TYPES[x]["title"] if x else "All")),
                   ("status", f[1].selectbox("Status", [""] + list(STATUS), format_func=lambda x: STATUS.get(x, "All"))),
                   ("language", f[2].selectbox("Language", ["", "en", "ar"], format_func=lambda x: x or "All")),
                   ("course_id", f[3].selectbox("Course", [""] + list(catalogue), format_func=lambda x: x or "All")),
                   ("q", f[4].text_input("Search", placeholder="title or content")))
            f[5].form_submit_button("Apply filters")
        try:
            rows = list_documents(flt)
        except ERRORS as exc:
            st.error(str(exc))
            rows = []
        if not rows:
            st.write("No documents yet - choose a source and document types in the sidebar and press **Generate**.")
        else:
            st.dataframe(pd.DataFrame([{
                "ID": d["document_id"], "Title": d["title"], "Type": TYPES.get(d["doc_type"], {}).get("title"),
                "Language": d["language"], "Source": d.get("curriculum_id") or d.get("course_id"),
                "Made by": d["metadata"].get("generated_by"), "Status": STATUS.get(d["status"], d["status"]),
                "v": d["version"], "Updated": (d.get("updated_at") or d["created_at"])[:16].replace("T", " ")}
                for d in rows]), hide_index=True)
            ids = [d["document_id"] for d in rows]
            titles = {d["document_id"]: d["title"] for d in rows}
            if st.session_state.get("doc_pick") not in ids:
                current = st.session_state.get("doc_selected")
                st.session_state.doc_pick = current if current in ids else ids[0]
            o1, o2 = st.columns([4, 1], vertical_alignment="bottom")
            o1.selectbox("Open", ids, key="doc_pick", format_func=lambda i: f"{i} · {titles[i]}")
            o2.button("Open selected", type="primary", on_click=lambda: open_document(st.session_state.doc_pick))


# ------------------------------------------------------------------ one document
def show_section(s: dict, rtl: bool) -> None:
    d = "rtl" if rtl else "ltr"
    tag = "h4" if s.get("level", 1) == 1 else "h5"
    lock = " 🔒" if s.get("locked") else ""
    st.markdown(f"<div dir='{d}'><{tag}>{s['heading']}{lock}</{tag}></div>", unsafe_allow_html=True)
    body = [f"<p>{p}</p>" for p in s.get("paragraphs") or []]
    if s.get("bullets"):
        lt = "ol" if s.get("numbered") else "ul"
        body.append(f"<{lt}>" + "".join(f"<li>{b}</li>" for b in s["bullets"]) + f"</{lt}>")
    if body:
        st.markdown(f"<div dir='{d}'>{''.join(body)}</div>", unsafe_allow_html=True)
    if s.get("table"):
        t = s["table"]
        st.dataframe(pd.DataFrame(t["rows"], columns=_unique(t["columns"])), hide_index=True)
    if s.get("notes"):
        st.caption("  \n".join(s["notes"]))


def _unique(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    return out


if tab_view.open is not False:
    with tab_view:
        did = st.session_state.get("doc_selected")
        if not did:
            st.write("Generate a document or open one from the **Documents** tab.")
            st.stop()
        try:
            doc = get_document(did)
        except ERRORS as exc:
            st.error(str(exc))
            st.stop()
        if note := st.session_state.pop("doc_notice", None):
            getattr(st, note[0])(note[1])
        meta = doc["metadata"]
        rtl = doc["language"] == "ar"
        st.markdown(f"<div dir='{'rtl' if rtl else 'ltr'}'><h3>{doc['title']}</h3></div>", unsafe_allow_html=True)
        how = {"llm": "the LLM (OpenAI) from the template", "rules": "rules from the template"}.get(
            meta.get("generated_by"), meta.get("generated_by"))
        src = doc.get("curriculum_id") and f"curriculum `{doc['curriculum_id']}` " \
            f"v{meta['facts'].get('curriculum_version')} ({meta['facts'].get('curriculum_status')})" \
            or f"catalogue course `{doc.get('course_id')}`"
        st.write(f"{STATUS.get(doc['status'], doc['status'])} · version {doc['version']} · `{did}` · "
                 f"{doc['template_id']} · generated by **{how}** from {src}")
        for n in meta.get("notes") or []:
            st.caption(f"ℹ️ {n}")

        pdf_key = (did, doc["version"], doc["status"])
        ready = st.session_state.get("doc_pdf")
        if ready and ready[0] == pdf_key:
            st.download_button("⬇️ Download PDF", ready[2], ready[1], "application/pdf", type="primary")
        elif st.button("📄 Get PDF"):
            with st.spinner("Fetching the PDF..."):
                try:
                    st.session_state.doc_pdf = (pdf_key, *pdf_bytes(did))
                    st.rerun()
                except ERRORS as exc:
                    st.warning(f"PDF unavailable: {exc}")

        if nxt := st.session_state.pop("doc_subtab_next", None):
            st.session_state.doc_subtab = nxt
        t_prev, t_edit, t_review, t_src = st.tabs(["Preview", "Edit", "Review & e-mail", "Sources & history"],
                                                  key="doc_subtab", on_change="rerun")
        with t_prev:
            for s in doc["sections"]:
                show_section(s, rtl)

        with t_edit:
            st.caption("Edit any section (🔒 = facts from approved data; edit only to correct them). Saving creates a "
                       "new version in draft, so the document goes through review again.")
            keys = {s["key"]: s for s in doc["sections"]}
            pick = st.selectbox("Section", list(keys), format_func=lambda k: f"{keys[k]['heading']}  ·  {k}"
                                + (" 🔒" if keys[k].get("locked") else ""))
            s = keys[pick]
            with st.form(f"edit_{did}_{doc['version']}_{pick}"):
                reviewer = st.text_input("Reviewer name", key="doc_reviewer_edit")
                heading = st.text_input("Heading", s["heading"])
                paras = st.text_area("Paragraphs (blank line between paragraphs)", "\n\n".join(s.get("paragraphs") or []),
                                     height=120)
                bullets = st.text_area("Bullets (one per line)", "\n".join(s.get("bullets") or []), height=150)
                table = None
                if s.get("table"):
                    table = st.data_editor(pd.DataFrame(s["table"]["rows"], columns=_unique(s["table"]["columns"])),
                                           num_rows="dynamic", hide_index=True, key=f"tbl_{did}_{pick}")
                notes = st.text_area("Notes (one per line)", "\n".join(s.get("notes") or []))
                c1, c2 = st.columns(2)
                save = c1.form_submit_button("Save section", type="primary")
                remove = c2.form_submit_button("Remove this section")
            if save or remove:
                if not reviewer.strip():
                    st.warning("Enter the reviewer's name.")
                else:
                    if remove:
                        body = {"reviewer": reviewer, "remove_sections": [pick]}
                    else:
                        new = {**s, "heading": heading.strip() or s["heading"],
                               "paragraphs": [p.strip() for p in paras.split("\n\n") if p.strip()],
                               "bullets": [b.strip() for b in bullets.splitlines() if b.strip()],
                               "notes": [n.strip() for n in notes.splitlines() if n.strip()]}
                        if table is not None:
                            new["table"] = {**s["table"], "rows": [["" if pd.isna(v) else str(v) for v in r]
                                                                   for r in table.itertuples(index=False)]}
                        body = {"reviewer": reviewer, "sections": [new]}
                    with st.spinner("Saving a new version..."):
                        try:
                            out = edit(did, body)
                            changed()
                            last = out["metadata"].get("last_edit", {})
                            notice("success", f"Saved as version {out['version']}: " +
                                   ("; ".join(last.get("applied", [])) or "no change"), "Edit")
                            st.rerun()
                        except ERRORS as exc:
                            st.error(str(exc))
            with st.form("doc_instructions"):
                st.markdown("**…or describe the change** (applied by the LLM when OPENAI_API_KEY is set)")
                reviewer_i = st.text_input("Reviewer name", key="doc_reviewer_instr")
                title = st.text_input("New title (optional)")
                instructions = st.text_area("Instructions", placeholder="Use banking examples in the scenario\n"
                                                                        "Make the welcome shorter")
                if st.form_submit_button("Apply"):
                    if not reviewer_i.strip() or not (title.strip() or instructions.strip()):
                        st.warning("Enter the reviewer's name and a title or instructions.")
                    else:
                        with st.spinner("Applying..."):
                            try:
                                out = edit(did, {"reviewer": reviewer_i, "title": title.strip() or None,
                                                 "instructions": instructions.strip() or None})
                                changed()
                                last = out["metadata"].get("last_edit", {})
                                msg = "Applied: " + ("; ".join(last.get("applied", [])) or "nothing")
                                if last.get("not_applied"):
                                    msg += "  \nNot applied: " + "; ".join(last["not_applied"])
                                notice("warning" if last.get("not_applied") else "success", msg, "Edit")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))

        with t_review:
            st.caption("Draft → submit → approved (handed to the Marketing Agent; any e-mail requested at creation is "
                       "sent) / changes requested / rejected. A document built from a curriculum can be approved only "
                       "when that curriculum is approved and unchanged.")
            with st.form("doc_decision"):
                reviewer = st.text_input("Reviewer name", key="doc_reviewer_decision")
                action = st.radio("Decision", ["pending_approval", "approved", "changes_requested", "rejected"],
                                  format_func=lambda x: {"pending_approval": "Submit for approval"}.get(x, STATUS[x]),
                                  horizontal=True)
                comment = st.text_input("Comment")
                if st.form_submit_button("Save decision"):
                    if not reviewer.strip():
                        st.warning("Enter the reviewer's name.")
                    else:
                        with st.spinner("Saving..."):
                            try:
                                review(did, action, reviewer, comment)
                                changed()
                                notice("success", f"Decision saved: {STATUS.get(action, action)}", "Review & e-mail")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))
            approved = doc["status"] == "approved"
            with st.form("doc_email"):
                st.markdown("**E-mail the PDF**" + ("" if approved else " - available once the document is approved"))
                to = st.text_input("To (comma separated)", disabled=not approved)
                sender = st.text_input("Sent by", disabled=not approved)
                message = st.text_area("Message (optional)", disabled=not approved)
                if st.form_submit_button("Send", disabled=not approved):
                    recipients = [x.strip() for x in to.split(",") if x.strip()]
                    if not (recipients and sender.strip()):
                        st.warning("Enter recipients and the sender's name.")
                    else:
                        with st.spinner("Sending..."):
                            try:
                                out = email(did, recipients, sender, message)
                                changed()
                                last = out["metadata"]["emails"][-1]
                                notice("success" if last["sent"] else "info",
                                       f"Sent to {', '.join(last['to'])}" if last["sent"] else
                                       f"SMTP is not configured - the e-mail was saved to {last.get('file')}",
                                       "Review & e-mail")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))
            for e in meta.get("emails") or []:
                st.caption(f"✉️ {e['at'][:16].replace('T', ' ')} · {e['by']} → {', '.join(e['to'])} via {e['channel']}")
            if meta.get("pending_email"):
                st.caption("Will be e-mailed on approval to: " + ", ".join(meta["pending_email"]))
            if doc.get("reviewer_comments"):
                st.markdown("**Reviewer comments**\n" + "\n".join(f"- {c}" for c in doc["reviewer_comments"]))

        with t_src:
            st.markdown("**Sources**")
            st.dataframe(pd.DataFrame([{"Source": s["source_id"], "Title": s["title"], "Record": s.get("record_id"),
                                        "URL": s.get("url")} for s in doc["sources"]]), hide_index=True)
            st.markdown("**Course facts used**")
            st.json(meta.get("facts") or {}, expanded=False)
            h = history(did)
            st.markdown("**Versions**")
            st.dataframe(pd.DataFrame(h["versions"]), hide_index=True)
            if h["reviews"]:
                st.markdown("**Reviews**")
                st.dataframe(pd.DataFrame(h["reviews"]), hide_index=True)
            if h["handoffs"]:
                st.markdown("**Handoffs**")
                st.dataframe(pd.DataFrame(h["handoffs"]), hide_index=True)
