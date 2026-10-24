"""RFP & Tender Research dashboard (owner: Member 2). Opens as the second page of the Streamlit app:

    streamlit run frontend/streamlit_app.py      -> sidebar: "RFP Research"

Talks to the API when it is running, otherwise runs the agent in-process (same as the chat page).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rex_ai import config  # noqa: E402
from rex_ai.core.schemas import RFPCategory, RFPStatus  # noqa: E402

st.set_page_config(page_title="REX - RFP Research", page_icon="📑", layout="wide")

COUNTRY = {"EG": "🇪🇬 Egypt", "SA": "🇸🇦 Saudi Arabia"}
FIT = {"likely": "🟢 likely", "possible": "🟡 possible", "unlikely": "🔴 unlikely", "unknown": "⚪ unknown"}


# ------------------------------------------------------------------ backend
def api_available() -> bool:
    try:
        return httpx.get(f"{config.API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if "rfp_api" not in st.session_state:
    st.session_state.rfp_api = api_available()
API = st.session_state.rfp_api


@st.cache_resource
def _local_ready() -> bool:
    from rex_ai.core import data_sync
    data_sync.get_assistant()                    # DB + academy_kb ready (used for qualification)
    return True


def local_agent():
    from rex_ai.agents.rfp.agent import RFPResearchAgent
    _local_ready()
    return RFPResearchAgent()


def run_research(payload: dict) -> dict:
    if API:
        r = httpx.post(f"{config.API_URL}/rfp/run", json=payload, timeout=900)
        r.raise_for_status()
        return r.json()
    from rex_ai.core.schemas import RFPRunRequest
    return local_agent().run(RFPRunRequest(**payload)).model_dump(mode="json")


def list_rfps(params: dict) -> list[dict]:
    params = {k: v for k, v in params.items() if v not in (None, "", False)}
    if API:
        return httpx.get(f"{config.API_URL}/rfp", params=params, timeout=30).json()
    return [r.model_dump(mode="json") for r in local_agent().search(**params)]


def set_status(rfp_id: str, status: str, reviewer: str, comment: str) -> None:
    payload = {"status": status, "reviewer": reviewer, "comment": comment}
    if API:
        httpx.patch(f"{config.API_URL}/rfp/{rfp_id}/status", json=payload, timeout=10).raise_for_status()
        return
    from rex_ai.core import db
    db.update_rfp_status(rfp_id, status)
    db.log_event("INFO", "rfp_review", agent="rfp_research", detail=f"{rfp_id}:{status} by {reviewer}: {comment}")


def ask(question: str, country: str | None) -> dict:
    payload = {"question": question, **({"country": country} if country else {})}
    if API:
        return httpx.post(f"{config.API_URL}/rfp/ask", json=payload, timeout=120).json()
    from rex_ai.core.schemas import RFPAskRequest
    return local_agent().ask(RFPAskRequest(**payload)).model_dump(mode="json")


def alerts() -> list[dict]:
    if API:
        return httpx.get(f"{config.API_URL}/rfp/alerts", timeout=10).json()
    from rex_ai.core import db
    _local_ready()
    return db.list_rfp_alerts(20)


def upload(name: str, content: bytes) -> str:
    if API:
        r = httpx.post(f"{config.API_URL}/rfp/upload", files={"file": (name, content)}, timeout=60)
        if r.status_code != 200:
            raise ValueError(r.json().get("message", r.text))
        return r.json()["file"]
    from rex_ai.agents.rfp import sources
    from rex_ai.core import data_sync
    ext = Path(name).suffix.lower()
    if ext not in sources.DOC_EXTENSIONS | sources.FEED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {ext}")
    folder = sources.documents_dir() if ext in sources.DOC_EXTENSIONS else sources.feeds_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = data_sync._unique_path(folder, data_sync._safe_name(name))
    path.write_bytes(content)
    return path.name


# ------------------------------------------------------------------ sidebar: run the agent
with st.sidebar:
    st.header("Run tender research")
    st.caption(("Connected to API " + config.API_URL) if API else "API not running: in-process mode")
    countries = st.multiselect("Countries", ["EG", "SA"], default=["EG", "SA"], format_func=COUNTRY.get,
                               help="Only Egypt and Saudi Arabia are supported - everything else is rejected.")
    categories = st.multiselect("Training categories", [c.value for c in RFPCategory],
                                default=[c.value for c in RFPCategory])
    days = st.slider("Deadline within (days)", 7, 365, 365)
    keywords = st.text_input("Extra topics (comma separated)", placeholder="ISO 27001, nursing")
    priority = st.text_input("Organisations / sectors to rank first", placeholder="bank, ministry")
    live = st.toggle("Also read live tender portals", value=False,
                     help="Fetches the portals listed in data/data_sources.yaml (robots.txt respected). Slower.")
    if st.button("Run research", type="primary", disabled=not countries):
        today = date.fromisoformat(config.RFP_TODAY) if config.RFP_TODAY else date.today()
        payload = {"countries": countries, "categories": categories,
                   "deadline_to": (today + timedelta(days=days)).isoformat() if days < 365 else None,
                   "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
                   "priority_organizations": [p.strip() for p in priority.split(",") if p.strip()],
                   "include_live_sources": live}
        with st.spinner("Collecting, filtering and assessing tenders..."):
            try:
                st.session_state.last_run = run_research(payload)
            except (httpx.HTTPError, ValueError) as exc:
                st.error(f"Run failed: {exc}")

    with st.expander("Add tender documents or listings"):
        st.caption("PDF / Word / TXT tender documents, or JSON / CSV / Excel exports from a portal. "
                   "They are processed on the next run.")
        files = st.file_uploader("Files", type=["pdf", "docx", "txt", "md", "json", "csv", "xlsx", "xls"],
                                 accept_multiple_files=True, label_visibility="collapsed")
        if files and st.button("Save files"):
            for f in files:
                try:
                    st.success(f"Saved {upload(f.name, f.getvalue())}")
                except (ValueError, httpx.HTTPError) as exc:
                    st.error(f"{f.name}: {exc}")

# ------------------------------------------------------------------ header + last run
st.title("📑 Educational RFP & Tender Research")
st.info("Prototype running on SAMPLE tenders (data/rfp/). Organisations, links and dates are fictional until real "
        "portal exports or tender documents are added.", icon="ℹ️")

if run := st.session_state.get("last_run"):
    c = st.columns(6)
    c[0].metric("Notices found", run["found"])
    c[1].metric("Kept", run["kept"])
    c[2].metric("New 🇪🇬 flagged", len(run["new_flagged_egypt"]))
    c[3].metric("New 🇸🇦 monitored", len(run["new_monitoring_saudi"]))
    c[4].metric("Outside EG/SA", run["rejected_country"])
    c[5].metric("Duplicates", run["duplicates"])
    st.caption(f"Job {run['job_id']} · not training: {run['rejected_irrelevant']} · expired: {run['expired']} · "
               f"outside filters: {run['rejected_filters']} · already known: {run['already_known']} · "
               f"alerts sent: {run['alerts_sent']}")
    for e in run.get("source_errors", []):
        st.warning(f"Source {e['source_id']} failed: {e['error']}")
    if run.get("rejections"):
        with st.expander(f"Why {len(run['rejections'])} notices were excluded"):
            st.dataframe(pd.DataFrame(run["rejections"]), hide_index=True)

# ------------------------------------------------------------------ pipeline table
tab_pipeline, tab_ask, tab_alerts = st.tabs(["Pipeline", "Ask about tenders", "Alerts"])
with tab_pipeline:
    f = st.columns([1, 1, 1, 2, 1])
    country = f[0].selectbox("Country", ["", "EG", "SA"], format_func=lambda x: COUNTRY.get(x, "All"))
    status = f[1].selectbox("Status", [""] + [s.value for s in RFPStatus], format_func=lambda x: x or "All")
    fit = f[2].selectbox("Academy fit", ["", "likely", "possible", "unlikely", "unknown"],
                         format_func=lambda x: FIT.get(x, "All"))
    q = f[3].text_input("Search", placeholder="title, organisation, eligibility, summary")
    open_only = f[4].checkbox("Open only", value=True)
    try:
        rows = list_rfps({"country": country, "status": status, "qualification": fit, "q": q, "open_only": open_only})
    except httpx.HTTPError as exc:
        st.error(f"Could not load the pipeline: {exc}")
        rows = []

    if not rows:
        st.write("No tenders yet - press **Run research** in the sidebar.")
    else:
        df = pd.DataFrame([{
            "ID": r["rfp_id"], "Country": COUNTRY[r["country"]], "Title": r["title"],
            "Organisation": r["issuing_organization"], "Category": r["training_category"],
            "Deadline": r["deadline"], "Status": r["status"], "Fit": FIT[r["qualification"]],
            "Courses": ", ".join(r["matched_course_ids"]), "Link": r["submission_link"]} for r in rows])
        st.dataframe(df, hide_index=True,
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")})
        csv = "\ufeff" + df.to_csv(index=False)
        st.download_button("Download CSV (Excel / Google Sheets)", csv, "rex_rfp_pipeline.csv", "text/csv")

        st.subheader("Review a tender")
        by_id = {r["rfp_id"]: r for r in rows}
        chosen = st.selectbox("Tender", list(by_id), format_func=lambda i: f"{i} · {by_id[i]['title'][:90]}")
        r = by_id[chosen]
        left, right = st.columns([3, 2])
        with left:
            rtl = any("؀" <= ch <= "ۿ" for ch in r["title"])
            st.markdown(f"<div dir='{'rtl' if rtl else 'ltr'}'><h4>{r['title']}</h4></div>", unsafe_allow_html=True)
            st.write(f"**{r['issuing_organization']}** · {COUNTRY[r['country']]} · {r['training_category']}")
            st.write(f"Deadline: **{r['deadline'] or 'not stated'}** · Published: {r['publication_date'] or '-'}")
            st.write(r["summary"])
            if r.get("eligibility_requirements"):
                st.markdown("**Eligibility**")
                st.text(r["eligibility_requirements"])
            if r.get("available_documents"):
                st.markdown("**Documents:** " + "; ".join(r["available_documents"]))
            if r.get("submission_link"):
                st.markdown(f"[Submission link]({r['submission_link']})")
            src = r["source"]
            st.caption(f"Source: {src['source_id']} · {src['title']}" + (f" · {src['url']}" if src.get("url") else ""))
        with right:
            st.markdown(f"**Academy fit: {FIT[r['qualification']]}**")
            st.text(r.get("qualification_notes") or "")
            with st.form(f"review_{chosen}"):
                new_status = st.selectbox("Decision", ["qualified", "rejected", "flagged_review", "monitoring"],
                                          help="qualified = prepare a bid; rejected = no bid")
                reviewer = st.text_input("Reviewer")
                comment = st.text_input("Comment")
                if st.form_submit_button("Save decision"):
                    if not reviewer.strip():
                        st.warning("Enter the reviewer's name.")
                    else:
                        set_status(chosen, new_status, reviewer, comment)
                        st.success(f"{chosen} -> {new_status}")
                        st.rerun()

with tab_ask:
    st.caption("Semantic search (RAG) over the stored tenders and their documents, with citations.")
    question = st.text_input("Question", placeholder="Which tenders need cybersecurity training? / مناقصات تحليل البيانات")
    scope = st.radio("Country", ["All", "EG", "SA"], horizontal=True)
    if question:
        with st.spinner("Searching..."):
            res = ask(question, None if scope == "All" else scope)
        st.markdown(res["answer"].replace("\n", "  \n"))
        if res["sources"]:
            with st.expander("Sources"):
                for s in res["sources"]:
                    st.caption(f"{s['source_id']} · {s['title']}")

with tab_alerts:
    st.caption("Egypt tenders are alerted immediately (high priority); Saudi tenders in one digest per run. "
               "Set REX_RFP_WEBHOOK_URL or REX_RFP_ALERT_EMAILS to also send them to Slack/Teams/e-mail.")
    items = alerts()
    if not items:
        st.write("No alerts yet.")
    for a in items:
        icon = "🔴" if a["priority"] == "high" else "🔵"
        st.write(f"{icon} `{a['created_at']}` **{a['rfp_id']}** - {a['message']}  \n"
                 f"<small>channels: {', '.join(a['channels'])}</small>", unsafe_allow_html=True)
