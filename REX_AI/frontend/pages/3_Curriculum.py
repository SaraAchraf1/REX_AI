"""Curriculum Research & Accreditation Alignment (owner: Member 2). Third page of the Streamlit app:

    streamlit run frontend/streamlit_app.py      -> sidebar: "Curriculum"

Talks to the API when it is running, otherwise runs the agent in-process (same as the other pages).

Speed: inputs live in forms (nothing reruns while typing), reference lists are cached, the PDF is built only when
asked for, and only the visible tab is drawn. Every change goes through changed(), which clears the caches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, no_fog  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX - Curriculum", page_icon="📚", layout="wide")
no_fog()

STATUS = {"draft": "📝 draft", "pending_approval": "⏳ pending approval", "changes_requested": "✏️ changes requested",
          "approved": "✅ approved", "rejected": "⛔ rejected"}
RESULT = {"covered": "🟢 covered", "partial": "🟡 partial", "missing": "🔴 missing"}
COUNTRY = {"EG": "🇪🇬 Egypt", "SA": "🇸🇦 Saudi Arabia"}
DEPARTMENTS = ["AI", "Data", "Software", "Cloud", "Cybersecurity", "Business", "Professional Development", "Healthcare"]
LIBRARY_TAB, VIEW_TAB = "Curricula", "Selected curriculum"
ERRORS = (LookupError, ValueError, httpx.HTTPError)


# ------------------------------------------------------------------ backend
def api_available() -> bool:
    try:
        return api().get(f"{config.API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if "cur_api" not in st.session_state:
    st.session_state.cur_api = api_available()
API = st.session_state.cur_api


@st.cache_resource
def local_agent():
    from rex_ai.agents.curriculum.agent import CurriculumAgent
    from rex_ai.core import data_sync
    data_sync.get_assistant()                    # DB initialised, catalogue loaded
    return CurriculumAgent()


def _call(method: str, path: str, **kw):
    r = api().request(method, f"{config.API_URL}{path}", timeout=600, **kw)
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        raise ValueError(body.get("message") or body.get("detail") or r.text)
    return r


# reads (cached; changed() clears them after every write)
@st.cache_data(ttl=600, show_spinner=False)
def frameworks() -> list[dict]:
    if API:
        return _call("GET", "/curriculum/frameworks").json()
    from rex_ai.agents.curriculum.agent import CurriculumAgent
    return [f.model_dump() for f in CurriculumAgent.frameworks()]


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
def list_curricula(params: tuple) -> list[dict]:
    params = {k: v for k, v in params if v not in (None, "", 0)}
    if API:
        return _call("GET", "/curriculum", params=params).json()
    return [c.model_dump(mode="json") for c in local_agent().search(**params)]


@st.cache_data(ttl=60, show_spinner=False)
def get_curriculum(cid: str) -> dict:
    if API:
        return _call("GET", f"/curriculum/{cid}").json()
    return local_agent().get(cid).model_dump(mode="json")


@st.cache_data(ttl=60, show_spinner=False)
def gap_reports(cid: str) -> list[dict]:
    if API:
        return _call("GET", f"/curriculum/{cid}/gap-reports").json()
    return [r.model_dump(mode="json") for r in local_agent().gap_reports(cid)]


@st.cache_data(ttl=60, show_spinner=False)
def history(cid: str) -> dict:
    if API:
        return _call("GET", f"/curriculum/{cid}/history").json()
    return local_agent().history(cid)


def changed() -> None:
    for fn in (list_curricula, get_curriculum, gap_reports, history):
        fn.clear()
    st.session_state.pop("cur_pdf", None)


# writes
def generate(payload: dict, preview: bool = False) -> dict:
    if API:
        return _call("POST", "/curriculum/research" if preview else "/curriculum", json=payload).json()
    from rex_ai.core.schemas import CurriculumRequest
    req = CurriculumRequest(**payload)
    out = local_agent().research(req) if preview else local_agent().generate(req)
    return out.model_dump(mode="json")


def import_course(payload: dict) -> dict:
    if API:
        return _call("POST", "/curriculum/import", json=payload).json()
    from rex_ai.core.schemas import CurriculumImport
    return local_agent().import_curriculum(CurriculumImport(**payload)).model_dump(mode="json")


def run_gap(cid: str, framework: str) -> dict:
    if API:
        return _call("POST", f"/curriculum/{cid}/gap-report", json={"framework": framework}).json()
    return local_agent().gap_report(cid, framework).model_dump(mode="json")


def review(cid: str, action: str, reviewer: str, comment: str) -> dict:
    payload = {"action": action, "reviewer": reviewer, "comment": comment or None}
    if API:
        return _call("POST", f"/curriculum/{cid}/review", json=payload).json()
    from rex_ai.core.schemas import ReviewAction
    return local_agent().review(cid, ReviewAction(**payload)).model_dump(mode="json")


def revise(cid: str, reviewer: str, instructions: str | None, content: dict | None) -> dict:
    payload = {"reviewer": reviewer, "instructions": instructions or None, "content": content}
    if API:
        return _call("POST", f"/curriculum/{cid}/revise", json=payload).json()
    from rex_ai.core.schemas import CurriculumRevision
    return local_agent().revise(cid, CurriculumRevision(**payload)).model_dump(mode="json")


def pdf_bytes(cid: str) -> tuple[str, bytes]:
    if API:
        r = _call("GET", f"/curriculum/{cid}/pdf")
        name = r.headers.get("content-disposition", "").split("filename=")[-1].strip('"') or f"{cid}.pdf"
        return name, r.content
    path = local_agent().export_pdf(cid)
    return path.name, path.read_bytes()


def open_curriculum(cid: str) -> None:
    """Select a curriculum and switch to its tab. Must run before st.tabs is drawn (button callback, or the
    sidebar code above the tabs): a widget's state can't change after it was created in the same run."""
    st.session_state.cur_selected = cid
    st.session_state.cur_tab = VIEW_TAB


def notice(kind: str, text: str, subtab: str | None = None) -> None:
    """Show a message after the rerun that follows a write (and keep the sub-tab the user acted in open)."""
    st.session_state.cur_notice = (kind, text)
    if subtab:
        st.session_state.cur_subtab_next = subtab


# ------------------------------------------------------------------ sidebar: discovery questions
fw_names = {f["framework_id"]: f"{f['short_name']} ({f['country']})" for f in frameworks()}
catalogue = courses()

with st.sidebar:
    st.header("Design a curriculum")
    st.caption(("Connected to API " + config.API_URL) if API else "API not running: in-process mode")
    mode = st.radio("Start from", ["New course", "Existing catalogue course"], horizontal=True)
    with st.form("design", border=False):
        course_id = None
        if mode == "Existing catalogue course":
            course_id = st.selectbox("Course", list(catalogue), format_func=lambda i: f"{i} · {catalogue[i]}")
            st.caption("Title, audience, level, hours and outcomes come from the catalogue; fill in only what you "
                       "want to change.")
        title = st.text_input("What course would you like to create?",
                              placeholder="Data Analysis for Healthcare Quality")
        audience = st.text_input("Who is the target audience?", placeholder="Hospital quality officers and nurses")
        level = st.selectbox("Intended skill level", ["", "Beginner", "Intermediate", "Advanced"],
                             format_func=lambda x: x or ("(from catalogue)" if course_id else "choose..."))
        hours = st.number_input("Training hours available", min_value=0, max_value=1000,
                                value=0 if course_id else 40, help="0 = take it from the catalogue course")
        outcomes = st.text_area("Expected learning outcomes (one per line)",
                                placeholder="Build a KPI dashboard for patient safety indicators")
        industry = st.text_input("Industry", placeholder="Healthcare")
        department = st.selectbox("Department", ["(infer)"] + DEPARTMENTS)
        framework = st.selectbox("Accreditation / competency framework", ["(suggest)"] + list(fw_names),
                                 format_func=lambda x: fw_names.get(x, x))
        country = st.radio("Country", ["EG", "SA"], format_func=COUNTRY.get, horizontal=True)
        projects = st.text_area("Required practical projects (one per line; the first becomes the capstone)")
        language = st.radio("Language", ["en", "ar"], format_func={"en": "English", "ar": "العربية"}.get,
                            horizontal=True)
        c1, c2 = st.columns(2)
        preview = c1.form_submit_button("Preview research")
        make = c2.form_submit_button("Generate", type="primary")

    if preview or make:
        payload = {k: v for k, v in {
            "course_id": course_id, "course_title": title.strip() or None, "target_audience": audience.strip() or None,
            "skill_level": level or None, "total_hours": int(hours) or None,
            "expected_outcomes": [o.strip() for o in outcomes.splitlines() if o.strip()],
            "industry": industry.strip() or None, "department": None if department == "(infer)" else department,
            "accreditation_framework": None if framework == "(suggest)" else framework, "country": country,
            "required_projects": [p.strip() for p in projects.splitlines() if p.strip()], "language": language,
        }.items() if v not in (None, [], "")}
        with st.spinner("Researching references..." if preview else
                        "Researching references, building the curriculum and checking alignment..."):
            try:
                if preview:
                    st.session_state.cur_research = generate(payload, preview=True)
                else:
                    cur = generate(payload)
                    changed()
                    st.session_state.pop("cur_research", None)
                    open_curriculum(cur["curriculum_id"])
                    st.success(f"Draft {cur['curriculum_id']} created")
            except ERRORS as exc:
                st.error(str(exc))

    with st.expander("Check an existing catalogue course against a framework"):
        with st.form("import", border=False):
            imp_course = st.selectbox("Catalogue course", list(catalogue), format_func=lambda i: f"{i} · {catalogue[i]}")
            imp_fw = st.selectbox("Framework", ["(suggest)"] + list(fw_names), format_func=lambda x: fw_names.get(x, x))
            imp_country = st.radio("Country", ["EG", "SA"], format_func=COUNTRY.get, horizontal=True, key="imp_country")
            if st.form_submit_button("Import and compare"):
                with st.spinner("Importing and checking alignment..."):
                    try:
                        cur = import_course({"course_id": imp_course, "country": imp_country,
                                             **({} if imp_fw == "(suggest)" else {"framework": imp_fw})})
                        changed()
                        open_curriculum(cur["curriculum_id"])
                    except ERRORS as exc:
                        st.error(str(exc))

# ------------------------------------------------------------------ header
st.title("📚 Curriculum Research & Accreditation Alignment")
st.info("Alignment is not accreditation: REX compares curricula with framework summaries (status: to verify) and "
        "never states that a course is accredited. Reference syllabi and industry skills are SAMPLE data.", icon="ℹ️")

if res := st.session_state.get("cur_research"):
    with st.expander("Research preview", expanded=True):
        st.write(f"**Department:** {res['department'] or '-'} · **Framework:** {res['framework_id'] or '-'} "
                 f"({res.get('framework_reason') or ''}) · **Related courses:** {', '.join(res['related_courses']) or '-'}")
        if res["industry_skills"]:
            st.write("**Industry skills:** " + ", ".join(res["industry_skills"]))
        for m in res["reference_modules"]:
            st.markdown(f"- **{m['title']}**: {'; '.join(m['topics'])}  \n  <small>{m['citation']}</small>",
                        unsafe_allow_html=True)
        for n in res["notes"]:
            st.caption(n)

tab_lib, tab_view = st.tabs([LIBRARY_TAB, VIEW_TAB], key="cur_tab", on_change="rerun")

# ------------------------------------------------------------------ library (metadata filtering)
if tab_lib.open is not False:                    # draw only the visible tab
    with tab_lib:
        with st.form("filters", border=False):
            f = st.columns([1, 1, 1, 1, 1, 2, 1], vertical_alignment="bottom")
            flt = (("department", f[0].selectbox("Department", [""] + DEPARTMENTS, format_func=lambda x: x or "All")),
                   ("skill_level", f[1].selectbox("Level", ["", "Beginner", "Intermediate", "Advanced"],
                                                  format_func=lambda x: x or "All")),
                   ("country", f[2].selectbox("Country", ["", "EG", "SA"], format_func=lambda x: COUNTRY.get(x, "All"))),
                   ("status", f[3].selectbox("Status", [""] + list(STATUS), format_func=lambda x: STATUS.get(x, "All"))),
                   ("max_hours", f[4].number_input("Max hours", 0, 1000, 0)),
                   ("q", f[5].text_input("Search", placeholder="title, topic, outcome")))
            f[6].form_submit_button("Apply filters")
        try:
            rows = list_curricula(flt)
        except ERRORS as exc:
            st.error(str(exc))
            rows = []
        if not rows:
            st.write("No curricula yet - answer the questions in the sidebar and press **Generate**.")
        else:
            st.dataframe(pd.DataFrame([{
                "ID": c["curriculum_id"], "Title": c["course_title"], "Department": c["metadata"].get("department"),
                "Level": c["metadata"].get("skill_level"), "Hours": c["training_hours"],
                "Country": c["metadata"].get("country"), "Framework": c["metadata"].get("accreditation_framework"),
                "Coverage": " · ".join(f"{a['framework_id']} {a['coverage_pct']:g}%"
                                       for a in c["metadata"].get("alignment", [])),
                "Status": STATUS.get(c["status"], c["status"]), "v": c["version"]} for c in rows]), hide_index=True)
            ids = [c["curriculum_id"] for c in rows]
            titles = {c["curriculum_id"]: c["course_title"] for c in rows}
            if st.session_state.get("cur_pick") not in ids:          # default: the open curriculum, else the first
                current = st.session_state.get("cur_selected")
                st.session_state.cur_pick = current if current in ids else ids[0]
            o1, o2 = st.columns([4, 1], vertical_alignment="bottom")
            o1.selectbox("Open", ids, key="cur_pick", format_func=lambda i: f"{i} · {titles[i]}")
            # read the pick inside the callback: args bound at draw time would miss a pick made in the same click
            o2.button("Open selected", type="primary", on_click=lambda: open_curriculum(st.session_state.cur_pick))

# ------------------------------------------------------------------ one curriculum
if tab_view.open is not False:
    with tab_view:
        cid = st.session_state.get("cur_selected")
        if not cid:
            st.write("Generate a curriculum or open one from the **Curricula** tab.")
            st.stop()
        try:
            cur = get_curriculum(cid)
        except ERRORS as exc:
            st.error(str(exc))
            st.stop()
        if note := st.session_state.pop("cur_notice", None):
            getattr(st, note[0])(note[1])
        meta = cur["metadata"]
        direction = "rtl" if cur.get("language") == "ar" else "ltr"
        st.markdown(f"<div dir='{direction}'><h3>{cur['course_title']}</h3></div>", unsafe_allow_html=True)
        if meta.get("generated_by") == "imported":
            origin = f"imported from **{meta.get('origin', 'a proposal')}** (checked, not generated)"
        else:
            sources = {"catalogue": "the catalogue course", "references": "reference syllabi (RAG)",
                       "framework": "framework topics", "industry_skills": "industry skills",
                       "none": "a generic outline (no evidence found)"}
            how = {"llm": "the LLM (OpenAI)", "rules": "rules"}.get(meta.get("generated_by"), meta.get("generated_by"))
            origin = f"generated by **{how}** from " + ", ".join(sources.get(e, e) for e in meta.get("evidence", []))
        st.write(f"{STATUS.get(cur['status'], cur['status'])} · version {cur['version']} · `{cid}` · {origin}")
        m = st.columns(5)
        m[0].metric("Training hours", f"{cur['training_hours']:g}")
        m[1].metric("Modules", len(cur["modules"]))
        m[2].metric("Outcomes", len(cur["learning_outcomes"]))
        for i, a in enumerate(meta.get("alignment", [])[:2]):
            m[3 + i].metric(a["framework_id"], f"{a['coverage_pct']:g}%")

        # the PDF is built on request only (it used to be rebuilt on every rerun)
        pdf_key = (cid, cur["version"], cur["status"])
        ready = st.session_state.get("cur_pdf")
        if ready and ready[0] == pdf_key:
            st.download_button("⬇️ Download PDF", ready[2], ready[1], "application/pdf", type="primary")
        elif st.button("📄 Create PDF"):
            with st.spinner("Building the PDF..."):
                try:
                    st.session_state.cur_pdf = (pdf_key, *pdf_bytes(cid))
                    st.rerun()
                except ERRORS as exc:
                    st.warning(f"PDF unavailable: {exc}")

        # keyed so an action can keep its sub-tab open after the rerun (their data is cached, switching is fast)
        if nxt := st.session_state.pop("cur_subtab_next", None):
            st.session_state.cur_subtab = nxt
        t_cur, t_align, t_review, t_src = st.tabs(["Curriculum", "Alignment report", "Review & modify",
                                                   "Sources & history"], key="cur_subtab", on_change="rerun")
        with t_cur:
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Objectives**")
                st.markdown("\n".join(f"- {o}" for o in cur["objectives"]))
                st.markdown("**Learning outcomes**")
                st.markdown("\n".join(f"{i}. {o}" for i, o in enumerate(cur["learning_outcomes"], 1)))
            with right:
                st.markdown("**Course information**")
                st.dataframe(pd.DataFrame([
                    ("Department", meta.get("department")), ("Target audience", cur["target_audience"]),
                    ("Skill level", meta.get("skill_level")), ("Industry", meta.get("industry")),
                    ("Duration", f"{cur['training_hours']:g} h · {meta.get('duration_weeks')} weeks"),
                    ("Country", meta.get("country")), ("Framework (alignment only)", meta.get("accreditation_framework")),
                    ("Prerequisites", "; ".join(cur["prerequisites"])),
                    ("Data sources & update", f"{', '.join(meta.get('data_sources', []))} · {meta.get('update_date')}")],
                    columns=["Field", "Value"]), hide_index=True)
            st.markdown("**Modules**")
            st.dataframe(pd.DataFrame([{"#": i, "Module": mod["title"], "Topics": "; ".join(mod["topics"]),
                                        "Practical work": "; ".join(mod["exercises"]), "Hours": mod["hours"]}
                                       for i, mod in enumerate(cur["modules"], 1)]), hide_index=True)
            a, b = st.columns(2)
            a.markdown("**Assessment plan**")
            a.dataframe(pd.DataFrame(cur["assessments"]), hide_index=True)
            b.markdown("**Capstone project**")
            b.write(cur["capstone_project"])

        with t_align:
            with st.form("compare", border=False):
                g1, g2 = st.columns([4, 1], vertical_alignment="bottom")
                other = g1.selectbox("Compare with another framework", list(fw_names), format_func=fw_names.get)
                if g2.form_submit_button("Run gap analysis", type="primary"):
                    with st.spinner("Comparing..."):
                        try:
                            rep = run_gap(cid, other)
                            changed()
                            st.session_state.cur_report_next = rep["framework_id"]     # show it after the rerun
                            st.session_state.cur_subtab_next = "Alignment report"
                            st.session_state.cur_gap_notice = ("success", f"Gap analysis against {rep['framework']}: {rep['coverage_pct']:g}% "
                                              f"coverage, {len(rep['missing_topics']) + len(rep['missing_outcomes'])} "
                                              f"gaps, {len(rep['partial'])} to confirm")
                            st.rerun()
                        except ERRORS as exc:
                            st.error(str(exc))
            if note := st.session_state.pop("cur_gap_notice", None):
                getattr(st, note[0])(note[1])
            reports = {r["framework_id"]: r for r in gap_reports(cid)}
            if reports:
                if (nxt := st.session_state.pop("cur_report_next", None)) in reports:
                    st.session_state.cur_report = nxt          # set before the selectbox is drawn
                if st.session_state.get("cur_report") not in reports:
                    st.session_state.cur_report = next(iter(reports))
                fid = st.selectbox("Report", list(reports), key="cur_report",
                                   format_func=lambda k: f"{reports[k]['framework']} - {reports[k]['coverage_pct']:g}%")
                r = reports[fid]
                st.progress(min(r["coverage_pct"] / 100, 1.0), text=f"Coverage {r['coverage_pct']:g}% "
                            f"(curriculum v{r.get('curriculum_version')})")
                st.dataframe(pd.DataFrame([{"Competency": i["competency"], "Kind": i["kind"],
                                            "Result": RESULT.get(i["status"], i["status"]), "Evidence": i["evidence"]}
                                           for i in r["items"]]), hide_index=True)
                c1, c2 = st.columns(2)
                c1.markdown("**Missing topics**\n" + ("\n".join(f"- {x}" for x in r["missing_topics"]) or "\nNone"))
                c2.markdown("**Missing outcomes**\n" + ("\n".join(f"- {x}" for x in r["missing_outcomes"]) or "\nNone"))
                if r["recommendations"]:
                    st.markdown("**Recommendations**\n" + "\n".join(f"- {x}" for x in r["recommendations"]))
                st.caption(r["disclaimer"])

        with t_review:
            st.caption("Draft → submit → approved (handed to the Document Agent) / changes requested / rejected. "
                       "A modification creates a new version and re-checks alignment.")
            with st.form("decision"):
                reviewer = st.text_input("Reviewer name", key="reviewer_decision")
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
                                review(cid, action, reviewer, comment)
                                changed()
                                notice("success", f"Decision saved: {STATUS.get(action, action)}", "Review & modify")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))
            with st.form("modify"):
                st.markdown("**Request modifications**")
                reviewer_m = st.text_input("Reviewer name", key="reviewer_modify")
                instructions = st.text_area("One instruction per line", placeholder=(
                    "add module Power BI dashboards\nadd topic control charts to module 4\nremove topic hand hygiene\n"
                    "add outcome: Interpret control charts\nset hours to 48\ncapstone: KPI dashboard for an ICU"))
                edited = st.text_area("…or edit fields as JSON (advanced)", placeholder='{"capstone_project": "..."}')
                if st.form_submit_button("Apply modifications"):
                    try:
                        content = json.loads(edited) if edited.strip() else None
                        if not reviewer_m.strip():
                            st.warning("Enter the reviewer's name.")
                        else:
                            with st.spinner("Applying and re-checking alignment..."):
                                new = revise(cid, reviewer_m, instructions.strip(), content)
                            changed()
                            last = new["metadata"].get("last_revision", {})
                            msg = "Applied: " + ("; ".join(last.get("applied", [])) or "nothing")
                            if last.get("not_applied"):
                                msg += "  \nNot applied (kept as comments): " + "; ".join(last["not_applied"])
                            notice("warning" if last.get("not_applied") else "success", msg, "Review & modify")
                            st.rerun()
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid JSON: {exc}")
                    except ERRORS as exc:
                        st.error(str(exc))
            if cur.get("reviewer_comments"):
                st.markdown("**Reviewer comments**\n" + "\n".join(f"- {c}" for c in cur["reviewer_comments"]))

        with t_src:
            st.markdown("**Sources**")
            st.dataframe(pd.DataFrame([{"Source": s["source_id"], "Title": s["title"], "Record": s.get("record_id"),
                                        "URL": s.get("url")} for s in cur["sources"]]), hide_index=True)
            h = history(cid)
            st.markdown("**Versions**")
            st.dataframe(pd.DataFrame(h["versions"]), hide_index=True)
            if h["reviews"]:
                st.markdown("**Reviews**")
                st.dataframe(pd.DataFrame(h["reviews"]), hide_index=True)
            if h["handoffs"]:
                st.markdown("**Handoffs**")
                st.dataframe(pd.DataFrame(h["handoffs"]), hide_index=True)
