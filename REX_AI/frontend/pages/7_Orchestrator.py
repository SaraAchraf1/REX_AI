"""Orchestrator & Tools (administrators only, owner: Member 3): is the flow between the agents working?

Shows the orchestrator's last flow report (every agent, every handoff, the MCP tool server and the external
services it reaches), lets an administrator run a check / repair now, and lists the agents' recent jobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, api_up, no_fog, prefetch, require  # noqa: E402
from rex_ai import config  # noqa: E402

st.set_page_config(page_title="REX - Orchestrator", page_icon="🧭", layout="wide")
no_fog()
ME = require("admin")
ICON = {"ok": "🟢", "warning": "🟠", "error": "🔴"}


def _get(path: str, **params):
    r = api().get(f"{config.API_URL}{path}", params=params, timeout=120)
    if r.status_code >= 400:
        raise ValueError(r.json().get("message", r.text))
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def status() -> dict:
    if api_up():
        return _get("/orchestrator/status")
    from rex_ai.agents.orchestrator.agent import Orchestrator, last_report
    return last_report() or Orchestrator().check(repair=False)


def run_check(repair: bool, deep: bool) -> dict:
    if api_up():
        r = api().post(f"{config.API_URL}/orchestrator/check", params={"repair": repair, "deep": deep}, timeout=300)
        r.raise_for_status()
        return r.json()
    from rex_ai.agents.orchestrator.agent import Orchestrator
    return Orchestrator().check(repair=repair, deep=deep)


@st.cache_data(ttl=300, show_spinner=False)
def tools_info() -> dict:
    if api_up():
        return _get("/orchestrator/tools")
    from rex_ai.mcp import client as tools
    return {"server": tools.target(), "tools": tools.list_tools(), "services": tools.call("service_status")}


@st.cache_data(ttl=30, show_spinner=False)
def jobs() -> list[dict]:
    if api_up():
        return _get("/orchestrator/jobs", limit=100)
    from rex_ai.core import db
    return db.list_jobs(limit=100)


prefetch((status,), (tools_info,), (jobs,))           # the three calls at once; read from the cache below
st.title("🧭 Orchestrator & Tools")
st.caption("The orchestrator checks every link of the flow - RFP → bid team, Curriculum → Documents → Marketing - "
           "repairs what is safe to repair (lost handoffs, expired tenders, stuck jobs, pending e-mails) and checks the "
           "MCP tool server through which every agent reaches LLMs, e-mail, webhooks and the web. It never approves or "
           f"publishes anything. Automatic check every {config.ORCH_INTERVAL:g} s while the API runs.")

c1, c2, c3 = st.columns([1, 1, 3])
repair = c2.toggle("Repair", value=True, help="Fix missing handoffs, expired tenders, stuck jobs, pending e-mails")
deep = c3.toggle("Deep check (log in to the mail server)", value=False)
try:
    if c1.button("Run check now", type="primary"):
        with st.spinner("Checking every agent, tool and handoff..."):
            report = run_check(repair, deep)
            status.clear()
            jobs.clear()
    else:
        report = status()
        if report.get("pending"):                # the first check is still running: ask again on the next rerun
            status.clear()
except (ValueError, httpx.HTTPError) as exc:
    st.error(f"The orchestrator could not be reached: {exc}")
    st.stop()

counts = report.get("summary", {})
st.subheader(f"{ICON[report['status']]} Flow is {report['status'].upper()}")
st.caption(f"Checked {report['checked_at'].replace('T', ' ')[:19]} UTC · " +
           " · ".join(f"{ICON[k]} {v} {k}" for k, v in counts.items()))

cols = st.columns(len(report["pipeline"]))
for col, stage in zip(cols, report["pipeline"]):
    with col:
        st.markdown(f"**{stage['stage']}**")
        st.caption("  \n".join(f"{k.replace('_', ' ')}: {v}" for k, v in stage["counts"].items()) or "nothing yet")

if report.get("repairs"):
    st.success("Repaired: " + "; ".join(f"{r['action']} - {r['target']}" for r in report["repairs"]))

tab_checks, tab_tools, tab_jobs = st.tabs(["Checks", "MCP tools & services", "Jobs"])
with tab_checks:
    for area in dict.fromkeys(c["area"] for c in report["checks"]):
        st.markdown(f"#### {area.capitalize()}")
        for c in (c for c in report["checks"] if c["area"] == area):
            st.markdown(f"{ICON[c['status']]} **{c['name']}** - {c['detail']}")
            if c["items"] and c["status"] != "ok":
                with st.expander(f"{len(c['items'])} item(s)"):
                    st.text("\n".join(c["items"]))

with tab_tools:
    try:
        info = tools_info()
    except (ValueError, httpx.HTTPError) as exc:
        st.error(str(exc))
    else:
        st.markdown(f"**MCP tool server:** `{info['server']}`  ·  run it standalone with "
                    "`python -m rex_ai.mcp.server --http` and set `REX_MCP_URL=http://127.0.0.1:8765/mcp`")
        st.dataframe(pd.DataFrame(info["tools"]), hide_index=True, width="stretch")
        rows = [{"Service": name, **{k: v for k, v in svc.items()}} for name, svc in info["services"].items()]
        st.dataframe(pd.DataFrame(rows).fillna(""), hide_index=True, width="stretch")

with tab_jobs:
    rows = jobs()
    if not rows:
        st.write("No jobs yet.")
    else:
        st.dataframe(pd.DataFrame([{"Job": j["job_id"], "Agent": j["agent"], "Status": j["status"],
                                    "Started": (j["started_at"] or "")[:19].replace("T", " "),
                                    "Result": j.get("output_ref") or "", "Error": (j.get("error") or "")[:200]}
                                   for j in rows]), hide_index=True, width="stretch")
