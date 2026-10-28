"""AI Marketing Agent (owner: Member 1). Fifth page of the Streamlit app:

    streamlit run frontend/streamlit_app.py      -> sidebar: "Marketing"

Talks to the API when it is running, otherwise runs the agent in-process (same as the other pages). Same speed
rules as the Documents page: inputs in forms, cached reads cleared by changed(), PDF fetched only on request, only
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

st.set_page_config(page_title="REX - Marketing", page_icon="📣", layout="wide")
no_fog()

STATUS = {"draft": "📝 draft", "pending_approval": "⏳ pending approval", "changes_requested": "✏️ changes requested",
          "approved": "✅ approved", "rejected": "⛔ rejected"}
OBJECTIVES = ["registrations", "awareness", "leads", "webinar sign-ups"]
LIBRARY_TAB, VIEW_TAB = "Campaigns", "Selected campaign"
ERRORS = (LookupError, ValueError, httpx.HTTPError)


# ------------------------------------------------------------------ backend
def api_available() -> bool:
    try:
        return api().get(f"{config.API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if "mkt_api" not in st.session_state:
    st.session_state.mkt_api = api_available()
API = st.session_state.mkt_api


@st.cache_resource
def local_agent():
    from rex_ai.agents.marketing.agent import MarketingAgent
    from rex_ai.core import data_sync
    data_sync.get_assistant()                    # DB initialised, catalogue loaded, academy_kb indexed
    return MarketingAgent()


def _call(method: str, path: str, **kw):
    r = api().request(method, f"{config.API_URL}{path}", timeout=600, **kw)
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        raise ValueError(body.get("message") or body.get("detail") or r.text)
    return r


@st.cache_data(ttl=600, show_spinner=False)
def channels() -> dict[str, dict]:
    if API:
        rows = _call("GET", "/campaigns/channels").json()
    else:
        from rex_ai.agents.marketing.agent import MarketingAgent
        rows = [c.model_dump(mode="json") for c in MarketingAgent.channels()]
    return {c["channel"]: c for c in rows}


@st.cache_data(ttl=600, show_spinner=False)
def courses() -> dict[str, dict]:
    if API:
        rows = _call("GET", "/courses").json()
    else:
        local_agent()
        from rex_ai.agents.rex.knowledge_base import get_kb
        rows = [c.model_dump(mode="json") for c in get_kb().courses.values()]
    return {c["course_id"]: c for c in rows}


@st.cache_data(ttl=60, show_spinner=False)
def list_campaigns(params: tuple) -> list[dict]:
    params = {k: v for k, v in params if v}
    if API:
        return _call("GET", "/campaigns", params=params).json()
    return [c.model_dump(mode="json") for c in local_agent().search(**params)]


@st.cache_data(ttl=60, show_spinner=False)
def get_campaign(cid: str) -> dict:
    if API:
        return _call("GET", f"/campaigns/{cid}").json()
    return local_agent().get(cid).model_dump(mode="json")


@st.cache_data(ttl=60, show_spinner=False)
def history(cid: str) -> dict:
    if API:
        return _call("GET", f"/campaigns/{cid}/history").json()
    return local_agent().history(cid)


@st.cache_data(ttl=60, show_spinner=False)
def handoffs() -> list[dict]:
    if API:
        return _call("GET", "/campaigns/handoffs").json()
    return local_agent().pending_handoffs()


def changed() -> None:
    for fn in (list_campaigns, get_campaign, history, handoffs):
        fn.clear()
    st.session_state.pop("mkt_pdf", None)


# writes
def generate(payload: dict) -> dict:
    if API:
        return _call("POST", "/campaigns", json=payload).json()
    from rex_ai.core.schemas import CampaignRequest
    return local_agent().generate(CampaignRequest(**payload)).model_dump(mode="json")


def edit(cid: str, payload: dict) -> dict:
    if API:
        return _call("PATCH", f"/campaigns/{cid}", json=payload).json()
    from rex_ai.core.schemas import CampaignEdit
    return local_agent().edit(cid, CampaignEdit(**payload)).model_dump(mode="json")


def review(cid: str, action: str, reviewer: str, comment: str) -> dict:
    payload = {"action": action, "reviewer": reviewer, "comment": comment or None}
    if API:
        return _call("POST", f"/campaigns/{cid}/review", json=payload).json()
    from rex_ai.core.schemas import ReviewAction
    return local_agent().review(cid, ReviewAction(**payload)).model_dump(mode="json")


def publish(cid: str, payload: dict) -> dict:
    if API:
        return _call("POST", f"/campaigns/{cid}/publish", json=payload).json()
    from rex_ai.core.schemas import CampaignPublish
    return local_agent().publish(cid, CampaignPublish(**payload)).model_dump(mode="json")


def pdf_bytes(cid: str) -> tuple[str, bytes]:
    if API:
        r = _call("GET", f"/campaigns/{cid}/pdf")
        name = r.headers.get("content-disposition", "").split("filename=")[-1].strip('"') or f"{cid}.pdf"
        return name, r.content
    path = local_agent().pdf(cid)
    return path.name, path.read_bytes()


def open_campaign(cid: str) -> None:
    """Select a campaign and switch to its tab (must run before st.tabs is drawn - button callback or sidebar)."""
    st.session_state.mkt_selected = cid
    st.session_state.mkt_tab = VIEW_TAB


def notice(kind: str, text: str, subtab: str | None = None) -> None:
    st.session_state.mkt_notice = (kind, text)
    if subtab:
        st.session_state.mkt_subtab_next = subtab


def create(payload: dict) -> None:
    with st.spinner("Reading approved course data, writing the copy and planning the campaign..."):
        try:
            out = generate(payload)
            changed()
            open_campaign(out["campaign_id"])
            notice("success", f"Created {out['campaign_id']}: {len(out['variants'])} variants, waiting for approval")
        except ERRORS as exc:
            st.error(str(exc))


# ------------------------------------------------------------------ sidebar: discovery questions (brief §10)
CHANNELS = channels()
catalogue = courses()

with st.sidebar:
    st.header("New campaign")
    st.caption(("Connected to API " + config.API_URL) if API else "API not running: in-process mode")
    with st.form("campaign", border=False):
        course_id = st.selectbox("Which course are we promoting?", list(catalogue),
                                 format_func=lambda i: f"{i} · {catalogue[i]['title_en']}")
        audience = st.text_input("Who is the target audience? (optional)",
                                 placeholder="default: the course's audience from the catalogue")
        chosen = st.multiselect("Which platforms?", list(CHANNELS), default=list(CHANNELS),
                                format_func=lambda c: CHANNELS[c]["name"])
        objective = st.selectbox("Campaign objective", OBJECTIVES)
        budget = st.number_input("Budget (optional, course currency)", min_value=0.0, step=500.0, value=0.0)
        offer = st.text_input("Offer (optional)", placeholder="default: the current promotion")
        cta = st.text_input("Call to action (optional)", placeholder="default: from the objective")
        c1, c2 = st.columns(2)
        variations = c1.number_input("Variations", 1, 5, 2)
        language = c2.radio("Language", ["en", "ar"], format_func={"en": "English", "ar": "العربية"}.get)
        wishes = st.text_area("Extra wishes for the LLM (optional)", placeholder="Mention the weekend cohort")
        go = st.form_submit_button("Generate campaign", type="primary")
    if go:
        if not chosen:
            st.warning("Choose at least one platform.")
        else:
            payload = {"course_id": course_id, "channels": chosen, "objective": objective,
                       "variations_per_channel": int(variations), "language": language,
                       "target_audience": audience.strip() or None, "budget": budget or None,
                       "offer": offer.strip() or None, "cta": cta.strip() or None, "instructions": wishes.strip() or None}
            create({k: v for k, v in payload.items() if v is not None})

# ------------------------------------------------------------------ header
st.title("📣 AI Marketing Agent")
st.info("Campaigns are written only from approved course data (catalogue, approved curriculum, fees, offers and "
        "cohorts); every number is checked against it. Nothing is published before an administrator approves it. "
        "Course data and brand guides are SAMPLE data until the academy sends its own.", icon="ℹ️")

try:
    waiting = handoffs()
except ERRORS:
    waiting = []
if waiting:
    with st.expander(f"Handed over by the Curriculum and Document agents ({len(waiting)})",
                     expanded=any(h["campaigns"] == 0 for h in waiting)):
        for h in waiting:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            kind = "📄 approved document" if h["kind"] == "document" else "📚 approved curriculum"
            c1.write(f"{kind} **{h['title']}** · `{h['id']}` · course `{h['course_id']}` · "
                     f"{h['campaigns']} campaign(s)")
            if c2.button("Create campaign", key=f"camp_{h['id']}"):
                create({"course_id": h["course_id"], "curriculum_id": h.get("curriculum_id"),
                        **({"document_id": h["id"]} if h["kind"] == "document" else {})})
                st.rerun()

if nxt := st.session_state.pop("mkt_tab_next", None):
    st.session_state.mkt_tab = nxt
tab_lib, tab_view = st.tabs([LIBRARY_TAB, VIEW_TAB], key="mkt_tab", on_change="rerun")

# ------------------------------------------------------------------ library
if tab_lib.open is not False:
    with tab_lib:
        with st.form("mkt_filters", border=False):
            f = st.columns([1, 1, 1, 2, 1], vertical_alignment="bottom")
            flt = (("course_id", f[0].selectbox("Course", [""] + list(catalogue), format_func=lambda x: x or "All")),
                   ("status", f[1].selectbox("Status", [""] + list(STATUS), format_func=lambda x: STATUS.get(x, "All"))),
                   ("language", f[2].selectbox("Language", ["", "en", "ar"], format_func=lambda x: x or "All")),
                   ("q", f[3].text_input("Search", placeholder="words in the copy or strategy")))
            f[4].form_submit_button("Apply filters")
        try:
            rows = list_campaigns(flt)
        except ERRORS as exc:
            st.error(str(exc))
            rows = []
        if not rows:
            st.write("No campaigns yet - answer the questions in the sidebar and press **Generate campaign**.")
        else:
            st.dataframe(pd.DataFrame([{
                "ID": c["campaign_id"], "Title": c.get("title"), "Course": c["course_id"],
                "Audience": (c.get("strategy_plan") or {}).get("audience", {}).get("persona_en"),
                "Channels": len(c.get("channels") or []), "Variants": len(c["variants"]), "Language": c["language"],
                "Made by": c["metadata"].get("generated_by"), "Status": STATUS.get(c["status"], c["status"]),
                "v": c.get("version", 1), "Published": len(c["metadata"].get("publications") or []),
                "Updated": (c.get("updated_at") or c["created_at"])[:16].replace("T", " ")} for c in rows]),
                hide_index=True)
            ids = [c["campaign_id"] for c in rows]
            titles = {c["campaign_id"]: c.get("title") for c in rows}
            if st.session_state.get("mkt_pick") not in ids:
                current = st.session_state.get("mkt_selected")
                st.session_state.mkt_pick = current if current in ids else ids[0]
            o1, o2 = st.columns([4, 1], vertical_alignment="bottom")
            o1.selectbox("Open", ids, key="mkt_pick", format_func=lambda i: f"{i} · {titles[i]}")
            o2.button("Open selected", type="primary", on_click=lambda: open_campaign(st.session_state.mkt_pick))


# ------------------------------------------------------------------ one campaign
def show_variant(v: dict, rtl: bool) -> None:
    d = "rtl" if rtl else "ltr"
    with st.container(border=True):
        st.markdown(f"**{v['variant_no']}. {v.get('headline') or ''}**  ·  angle *{v.get('angle')}*  ·  "
                    f"{v.get('char_count')} characters  ·  by {v.get('generated_by')}")
        body = v["body"].replace("<", "&lt;").replace("\n", "<br>")
        st.markdown(f"<div dir='{d}'>{body}</div>", unsafe_allow_html=True)
        st.caption(f"CTA: {v['cta']}")
        for c in v.get("checks") or []:
            st.caption(f"⚠️ {c}")


if tab_view.open is not False:
    with tab_view:
        cid = st.session_state.get("mkt_selected")
        if not cid:
            st.write("Generate a campaign or open one from the **Campaigns** tab.")
            st.stop()
        try:
            camp = get_campaign(cid)
        except ERRORS as exc:
            st.error(str(exc))
            st.stop()
        if note := st.session_state.pop("mkt_notice", None):
            getattr(st, note[0])(note[1])
        meta, plan = camp["metadata"], camp.get("strategy_plan") or {}
        rtl = camp["language"] == "ar"
        st.subheader(camp.get("title") or cid)
        src = (f"curriculum `{camp['curriculum_id']}`" if camp.get("curriculum_id") else "the catalogue") + \
            (f" · document `{camp['document_id']}`" if camp.get("document_id") else "")
        st.write(f"{STATUS.get(camp['status'], camp['status'])} · version {camp.get('version', 1)} · `{cid}` · "
                 f"course `{camp['course_id']}` from {src} · written by **{meta.get('generated_by')}**")
        for n in meta.get("notes") or []:
            st.caption(f"ℹ️ {n}")
        for n in meta.get("llm_rejected") or []:
            st.caption(f"🛡️ {n}")

        pdf_key = (cid, camp.get("version"), camp["status"])
        ready = st.session_state.get("mkt_pdf")
        if ready and ready[0] == pdf_key:
            st.download_button("⬇️ Download campaign brief (PDF)", ready[2], ready[1], "application/pdf", type="primary")
        elif st.button("📄 Get campaign brief PDF"):
            with st.spinner("Fetching the PDF..."):
                try:
                    st.session_state.mkt_pdf = (pdf_key, *pdf_bytes(cid))
                    st.rerun()
                except ERRORS as exc:
                    st.warning(f"PDF unavailable: {exc}")

        if nxt := st.session_state.pop("mkt_subtab_next", None):
            st.session_state.mkt_subtab = nxt
        t_content, t_strategy, t_edit, t_review, t_src = st.tabs(
            ["Content", "Strategy", "Edit", "Review & publish", "Sources & history"], key="mkt_subtab",
            on_change="rerun")

        with t_content:
            if camp.get("headlines"):
                st.markdown("**Headline options:** " + " · ".join(camp["headlines"]))
                st.markdown("**Calls to action:** " + " · ".join(camp.get("ctas") or []))
            for ch in dict.fromkeys(v["channel"] for v in camp["variants"]):
                st.markdown(f"#### {CHANNELS.get(ch, {}).get('name', ch)}")
                for v in (x for x in camp["variants"] if x["channel"] == ch):
                    show_variant(v, rtl)

        with t_strategy:
            st.text(camp["strategy"])
            a = plan.get("audience") or {}
            if a:
                c1, c2 = st.columns(2)
                c1.markdown("**Pain points**\n" + "\n".join(f"- {x}" for x in a.get("pains") or []))
                c2.markdown("**Motivations**\n" + "\n".join(f"- {x}" for x in a.get("motivations") or []))
                st.caption("Targeting: " + "; ".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                                                     for k, v in (a.get("targeting") or {}).items()))
            if plan.get("channel_mix"):
                st.markdown("**Channel mix**")
                st.dataframe(pd.DataFrame([{k: m.get(k) for k in ("priority", "name", "role", "why", "paid", "budget")}
                                           for m in plan["channel_mix"]]), hide_index=True)
            if plan.get("timeline"):
                st.markdown("**Timeline**")
                st.dataframe(pd.DataFrame([{"Date": t.get("date") or f"week {t.get('week')}", "Phase": t["phase"],
                                            "Channels": ", ".join(t["channels"]), "Action": t["action"]}
                                           for t in plan["timeline"]]), hide_index=True)
            if plan.get("ab_tests"):
                st.markdown("**A/B tests**")
                st.dataframe(pd.DataFrame(plan["ab_tests"]), hide_index=True)
            if plan.get("objection_handling"):
                st.markdown("**Objection handling for the sales team** (academy FAQs, found through the RAG index)")
                st.dataframe(pd.DataFrame(plan["objection_handling"])[["question", "answer"]], hide_index=True)
            st.markdown("**KPIs**\n" + "\n".join(f"- {k}" for k in plan.get("kpis") or []))
            st.markdown("**Compliance**\n" + "\n".join(f"- {k}" for k in plan.get("compliance") or []))

        with t_edit:
            st.caption("Saving creates a new version that goes back to approval. Numbers that are not in the approved "
                       "data are accepted from a reviewer but flagged.")
            keys = {f"{v['channel']}:{v['variant_no']}": v for v in camp["variants"]}
            pick = st.selectbox("Variant", list(keys), format_func=lambda k: f"{k} · {keys[k].get('headline') or ''}")
            v = keys[pick]
            with st.form(f"mkt_edit_{cid}_{camp.get('version')}_{pick}"):
                reviewer = st.text_input("Reviewer name", key="mkt_reviewer_edit")
                headline = st.text_input("Headline", v.get("headline") or "")
                body = st.text_area("Body", v["body"], height=260)
                cta_new = st.text_input("CTA", v["cta"])
                c1, c2 = st.columns(2)
                save = c1.form_submit_button("Save variant", type="primary")
                remove = c2.form_submit_button("Remove this variant")
            if save or remove:
                if not reviewer.strip():
                    st.warning("Enter the reviewer's name.")
                else:
                    payload = {"reviewer": reviewer, "remove_variants": [pick]} if remove else \
                        {"reviewer": reviewer, "variants": [{**v, "headline": headline.strip() or None,
                                                             "body": body.strip(), "cta": cta_new.strip() or v["cta"]}]}
                    with st.spinner("Saving a new version..."):
                        try:
                            out = edit(cid, payload)
                            changed()
                            last = out["metadata"].get("last_edit", {})
                            notice("success", f"Saved as version {out['version']}: " +
                                   ("; ".join(last.get("applied", [])) or "no change"), "Edit")
                            st.rerun()
                        except ERRORS as exc:
                            st.error(str(exc))
            with st.form("mkt_instructions"):
                st.markdown("**…or describe the change** (applied by the LLM when OPENAI_API_KEY is set)")
                reviewer_i = st.text_input("Reviewer name", key="mkt_reviewer_instr")
                strategy_text = st.text_area("New strategy text (optional)")
                instructions = st.text_area("Instructions", placeholder="Make the WhatsApp messages shorter\n"
                                                                        "Mention the weekend cohort in the e-mails")
                if st.form_submit_button("Apply"):
                    if not reviewer_i.strip() or not (strategy_text.strip() or instructions.strip()):
                        st.warning("Enter the reviewer's name and a strategy or instructions.")
                    else:
                        with st.spinner("Applying..."):
                            try:
                                out = edit(cid, {"reviewer": reviewer_i, "strategy": strategy_text.strip() or None,
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
            st.caption("Pending approval → approved / changes requested / rejected. Approval is refused while the "
                       "curriculum is not approved or the fees, offers and dates in the copy have changed since it was "
                       "written. Only approved campaigns can be published.")
            with st.form("mkt_decision"):
                reviewer = st.text_input("Administrator name", key="mkt_reviewer_decision")
                action = st.radio("Decision", ["approved", "changes_requested", "rejected", "pending_approval"],
                                  format_func=lambda x: {"pending_approval": "Resubmit for approval"}.get(x, STATUS[x]),
                                  horizontal=True)
                comment = st.text_input("Comment")
                if st.form_submit_button("Save decision"):
                    if not reviewer.strip():
                        st.warning("Enter the administrator's name.")
                    else:
                        with st.spinner("Saving..."):
                            try:
                                review(cid, action, reviewer, comment)
                                changed()
                                notice("success", f"Decision saved: {STATUS.get(action, action)}", "Review & publish")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))
            approved = camp["status"] == "approved"
            with st.form("mkt_publish"):
                st.markdown("**Publish**" + ("" if approved else " - available once the campaign is approved"))
                keys = [f"{v['channel']}:{v['variant_no']}" for v in camp["variants"]]
                picked = st.multiselect("Variants to publish (A/B winners)", keys, default=keys, disabled=not approved)
                publisher = st.text_input("Published by", disabled=not approved)
                to = st.text_input("Send the e-mail variant to (consented leads, comma separated)",
                                   disabled=not approved)
                if st.form_submit_button("Publish", disabled=not approved):
                    if not (publisher.strip() and picked):
                        st.warning("Choose variants and enter who publishes.")
                    else:
                        with st.spinner("Writing the distribution package..."):
                            try:
                                out = publish(cid, {"publisher": publisher, "variant_keys": picked,
                                                    "email_to": [x.strip() for x in to.split(",") if x.strip()]})
                                changed()
                                last = out["metadata"]["publications"][-1]
                                msg = f"Published {len(last['variants'])} variants to {last['folder']}"
                                if last.get("email"):
                                    msg += ("  \nE-mail sent" if last["email"]["sent"] else
                                            f"  \nSMTP is not configured - e-mail saved to {last['email'].get('file')}")
                                notice("success", msg, "Review & publish")
                                st.rerun()
                            except ERRORS as exc:
                                st.error(str(exc))
            for p in meta.get("publications") or []:
                st.caption(f"🚀 {p['at'][:16].replace('T', ' ')} · v{p['version']} · {p['by']} · "
                           f"{', '.join(p['variants'])} → {p['folder']}")
            if camp.get("reviewer_comments"):
                st.markdown("**Reviewer comments**\n" + "\n".join(f"- {c}" for c in camp["reviewer_comments"]))

        with t_src:
            st.markdown("**Sources**")
            st.dataframe(pd.DataFrame([{"Source": s["source_id"], "Title": s["title"], "Record": s.get("record_id"),
                                        "URL": s.get("url")} for s in camp["sources"]]), hide_index=True)
            st.markdown("**Course facts the copy relies on**")
            st.json(camp.get("course_facts_used") or {}, expanded=False)
            h = history(cid)
            st.markdown("**Versions**")
            st.dataframe(pd.DataFrame(h["versions"]), hide_index=True)
            if h["reviews"]:
                st.markdown("**Reviews**")
                st.dataframe(pd.DataFrame(h["reviews"]), hide_index=True)
            if h["handoffs"]:
                st.markdown("**Handoffs**")
                st.dataframe(pd.DataFrame(h["handoffs"]), hide_index=True)
