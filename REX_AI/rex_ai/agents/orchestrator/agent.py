"""Orchestrator Agent: keeps the flow between all agents working (owner: Member 3).

The academy's pipeline (brief):
    RFP Research  -> qualified tender e-mailed to the bid team
    REX Assistant -> leads (consented) -> Marketing audience insights
    Curriculum (approved) --handoff--> Document Agent (approved) --handoff--> Marketing Agent -> published campaign

The orchestrator
  * routes chat requests to the right agent (POST /route) and records each routed run as a job, so a failing agent
    gives a clear error instead of a crash;
  * checks every link of the flow (check()) and repairs what it safely can: missing handoff events, tenders past
    their deadline, jobs left "running" by a dead process, approved documents whose e-mail is still pending;
  * checks the MCP tool server and every external service the agents need (LLMs, e-mail, webhook), so a missing
    key or a refused mail login is reported before a reviewer hits it.
The checks and repairs never approve, generate or publish anything. An administrator's tasks (tasks.py: "create a
curriculum and a campaign for it") do create drafts through the other agents; approving, e-mailing and publishing
stay human decisions there too.

The API runs check() at start-up and every REX_ORCH_INTERVAL seconds (GET /orchestrator/status shows the last
report); the Orchestrator page in the Streamlit app shows it too.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from rex_ai import config
from rex_ai.core import db
from rex_ai.core.schemas import AgentName

log = logging.getLogger("rex.orchestrator")
AGENT = AgentName.orchestrator.value
OK, WARNING, ERROR = "ok", "warning", "error"
_RANK = {OK: 0, WARNING: 1, ERROR: 2}

_last_report: Optional[dict] = None
_report_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_hours(iso: Optional[str]) -> float:
    if not iso:
        return 0.0
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (_now() - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds() / 3600


def _check(area: str, name: str, status: str, detail: str, items: Optional[list[str]] = None) -> dict:
    return {"area": area, "name": name, "status": status, "detail": detail, "items": items or []}


def last_report() -> Optional[dict]:
    return _last_report


class Orchestrator:
    def __init__(self, stale_hours: Optional[float] = None):
        self.stale_hours = config.ORCH_STALE_HOURS if stale_hours is None else stale_hours

    # ================================================================== routing with tracking
    def dispatch(self, agent: AgentName, intent: str, message: str, fn: Callable[[], dict]) -> dict:
        """Run one routed request as a tracked job. A failure becomes {"status": "error", ...} with the reason."""
        job_id = db.start_job(AGENT, {"routed_to": agent.value, "intent": intent, "message": message[:500]})
        try:
            result = fn()
        except Exception as exc:
            log.exception("Routed request to %s failed", agent.value)
            db.finish_job(job_id, "failed", error=f"{agent.value}: {exc!r}"[:1000])
            db.log_event("ERROR", "route_failed", agent=AGENT, detail=f"{agent.value}/{intent}: {exc!r}"[:500])
            return {"status": "error", "job_id": job_id,
                    "message": f"The {agent.value.replace('_', ' ')} agent could not handle this request: {exc}"}
        ref = next((result.get(k) for k in ("curriculum_id", "document_id", "campaign_id") if result.get(k)), None)
        db.finish_job(job_id, "succeeded", output_ref=ref or result.get("status"))
        return result

    # ================================================================== flow check
    def check(self, repair: bool = True, deep: bool = False) -> dict:
        """Check every agent, tool and handoff. repair=True fixes what is safe to fix; deep=True also logs in to the
        mail server (no e-mail is sent)."""
        global _last_report
        checks: list[dict] = []
        repairs: list[dict] = []
        steps = (("database", self._database), ("tools", lambda r: self._tools(deep)), ("agents", self._agents),
                 ("jobs", self._jobs), ("rfp", self._rfp_flow), ("curriculum", self._curriculum_flow),
                 ("documents", self._document_flow), ("marketing", self._marketing_flow))
        for area, step in steps:
            try:
                checks.extend(step(repairs if repair else None))
            except Exception as exc:                        # a broken check is itself a finding
                log.exception("Orchestrator check %s failed", area)
                checks.append(_check(area, f"{area} check", ERROR, f"The check could not run: {exc!r}"[:500]))
        status = max((c["status"] for c in checks), key=_RANK.get, default=OK)
        report = {"status": status, "checked_at": _now().isoformat(timespec="seconds"), "repair": repair,
                  "deep": deep, "summary": dict(Counter(c["status"] for c in checks)), "checks": checks,
                  "repairs": repairs, "pipeline": self.pipeline()}
        for r in repairs:
            db.log_event("INFO", "orchestrator_repair", agent=AGENT, detail=f"{r['action']} {r['target']}: {r['detail']}"[:500])
        if status != OK:
            bad = "; ".join(f"{c['name']}: {c['detail']}" for c in checks if c["status"] != OK)
            db.log_event("WARNING" if status == WARNING else "ERROR", "orchestrator_check", agent=AGENT,
                         detail=bad[:1000])
        with _report_lock:
            _last_report = report
        return report

    def pipeline(self) -> list[dict]:
        """Counts per stage, in flow order (for the dashboard)."""
        rfps = Counter(r["status"] for r in db.list_rfps(limit=10000))
        curricula = Counter(r["status"] for r in db.list_curricula(limit=10000))
        documents = Counter(r["status"] for r in db.list_documents(limit=10000))
        campaigns = db.list_campaigns(limit=10000)
        published = sum(1 for r in campaigns if (r["content"] or {}).get("metadata", {}).get("publications"))
        return [
            {"stage": "RFP Research", "agent": AgentName.rfp_research.value, "counts": dict(rfps)},
            {"stage": "REX Assistant (leads)", "agent": AgentName.rex_assistant.value,
             "counts": {"leads": len(db.list_leads())}},
            {"stage": "Curriculum", "agent": AgentName.curriculum.value, "counts": dict(curricula)},
            {"stage": "Documents", "agent": AgentName.document.value, "counts": dict(documents)},
            {"stage": "Marketing", "agent": AgentName.marketing.value,
             "counts": {**Counter(r["status"] for r in campaigns), "published": published}},
        ]

    # ------------------------------------------------------------------ infrastructure
    @staticmethod
    def _database(repairs) -> list[dict]:
        counts = db.counts()
        return [_check("infrastructure", "Database", OK, f"{db.describe()} - {sum(counts.values())} rows")]

    @staticmethod
    def _tools(deep: bool) -> list[dict]:
        from rex_ai.mcp import client as tools
        try:
            names = [t["name"] for t in tools.list_tools()]
            status = tools.call("service_status")
        except tools.ToolCallError as exc:
            return [_check("tools", "MCP tool server", ERROR, str(exc))]
        out = [_check("tools", "MCP tool server", OK, f"{tools.target()} - {len(names)} tools", names)]
        o, g, smtp = status["openai"], status["gemini"], status["smtp"]
        for name, svc, key, without in (
                ("OpenAI (generation, assistant, RFP)", o, "OPENAI_API_KEY", "agents use rule-based output"),
                ("Gemini (change requests)", g, "GEMINI_API_KEY", "plain-language change requests cannot be applied")):
            why = ("switched off (REX_USE_LLM=0)" if svc["configured"] else f"{key} is not set") + f" - {without}"
            out.append(_check("tools", name, OK if svc["enabled"] else WARNING,
                              f"model {svc['model']}" if svc["enabled"] else why))
        if not smtp["configured"]:
            out.append(_check("tools", "E-mail (SMTP)", WARNING, "not configured - approved documents and qualified "
                              "tenders cannot be e-mailed (set REX_SMTP_* in .env)"))
        elif not smtp["login_set"]:
            out.append(_check("tools", "E-mail (SMTP)", WARNING, f"{smtp['host']}:{smtp['port']} without a login - set "
                              "REX_SMTP_USER and REX_SMTP_PASSWORD"))
        elif deep:
            try:
                tools.call("smtp_check")
                out.append(_check("tools", "E-mail (SMTP)", OK, f"logged in to {smtp['host']}:{smtp['port']}"))
            except tools.ToolCallError as exc:
                out.append(_check("tools", "E-mail (SMTP)", ERROR, str(exc)))
        else:
            out.append(_check("tools", "E-mail (SMTP)", OK, f"{smtp['host']}:{smtp['port']} (run a deep check to test "
                              "the login)"))
        if status["webhook"]["configured"]:
            out.append(_check("tools", "Alert webhook", OK, "configured"))
        return out

    @staticmethod
    def _agents(repairs) -> list[dict]:
        from rex_ai.agents.curriculum.agent import CurriculumAgent
        from rex_ai.agents.documents.agent import DocumentAgent
        from rex_ai.agents.marketing.agent import MarketingAgent
        from rex_ai.agents.rfp.agent import RFPResearchAgent
        from rex_ai.core import data_sync
        out = []
        kb = data_sync.get_assistant().kb
        out.append(_check("agents", "REX Assistant", OK if kb.courses else ERROR,
                          f"{len(kb.courses)} courses loaded" if kb.courses else "no courses in the catalogue"))
        for name, make in (("RFP Research", RFPResearchAgent), ("Curriculum", CurriculumAgent),
                           ("Document", lambda: DocumentAgent.types()), ("Marketing", MarketingAgent)):
            try:
                make()
                out.append(_check("agents", f"{name} Agent", OK, "ready"))
            except Exception as exc:
                out.append(_check("agents", f"{name} Agent", ERROR, f"cannot start: {exc!r}"[:300]))
        return out

    @staticmethod
    def _jobs(repairs) -> list[dict]:
        out = []
        cutoff = (_now() - timedelta(minutes=config.ORCH_JOB_TIMEOUT_MIN)).isoformat(timespec="seconds")
        stuck = [j["job_id"] for j in db.list_jobs(status="running", limit=1000) if j["started_at"] < cutoff]
        if stuck and repairs is not None:
            db.fail_stale_jobs(cutoff, f"still running after {config.ORCH_JOB_TIMEOUT_MIN:g} min - marked failed by "
                                       "the orchestrator (the process probably stopped)")
            repairs.extend({"action": "job marked failed", "target": j, "detail": "stuck in running"} for j in stuck)
        elif stuck:
            out.append(_check("jobs", "Stuck jobs", WARNING, f"{len(stuck)} job(s) running for over "
                              f"{config.ORCH_JOB_TIMEOUT_MIN:g} min", stuck))
        if repairs is not None:
            repairs.extend({"action": "task marked failed", "target": t, "detail": "no progress, its process stopped"}
                           for t in db.fail_stale_tasks(cutoff, f"No progress for {config.ORCH_JOB_TIMEOUT_MIN:g} min - "
                                                        "the process running this task probably stopped"))
        since = (_now() - timedelta(hours=24)).isoformat(timespec="seconds")
        failed = [j for j in db.list_jobs(status="failed", since=since, limit=50) if j["job_id"] not in stuck]
        out.append(_check("jobs", "Failed jobs (24 h)", WARNING if failed else OK,
                          f"{len(failed)} failed" if failed else "none",
                          [f"{j['job_id']} {j['agent']}: {(j['error'] or '')[:150]}" for j in failed[:10]]))
        return out

    # ------------------------------------------------------------------ the flow, link by link
    def _rfp_flow(self, repairs) -> list[dict]:
        from rex_ai.agents.rfp.agent import _today
        out = []
        if repairs is not None:
            n = db.expire_rfps(_today().isoformat())
            if n:
                repairs.append({"action": "tenders expired", "target": f"{n} tender(s)", "detail": "deadline passed"})
        rfps = db.list_rfps(limit=10000)
        waiting = [r for r in rfps if r["status"] == "flagged_review" and _age_hours(r.get("first_seen")) > self.stale_hours]
        out.append(_check("flow", "RFP: Egypt tenders reviewed", WARNING if waiting else OK,
                          f"{len(waiting)} flagged tender(s) waiting over {self.stale_hours:g} h for a decision"
                          if waiting else "no tender waiting too long",
                          [f"{r['rfp_id']} {r['title'][:80]} (deadline {r.get('deadline') or '-'})" for r in waiting[:10]]))
        emailed = {a["rfp_id"] for a in db.list_rfp_alerts(limit=10000) if a["priority"] == "qualified"}
        not_sent = [r for r in rfps if r["status"] == "qualified" and r["rfp_id"] not in emailed]
        out.append(_check("flow", "RFP: qualified tenders e-mailed", WARNING if not_sent else OK,
                          f"{len(not_sent)} qualified tender(s) not e-mailed to anyone - save the decision again with "
                          "an e-mail address" if not_sent else "every qualified tender was e-mailed",
                          [f"{r['rfp_id']} {r['title'][:80]}" for r in not_sent[:10]]))
        return out

    def _curriculum_flow(self, repairs) -> list[dict]:
        out = []
        handed = {e["payload_id"]: e for e in reversed(db.list_handoffs_to(AgentName.document.value))
                  if e["payload_type"] == "curriculum"}                  # newest handoff per curriculum
        missing = [r for r in db.list_curricula(status="approved", limit=10000) if r["curriculum_id"] not in handed]
        if missing and repairs is not None:
            for r in missing:
                ev = db.save_handoff(AgentName.curriculum.value, AgentName.document.value, "curriculum",
                                     r["curriculum_id"])
                handed[r["curriculum_id"]] = ev
                repairs.append({"action": "handoff re-sent", "target": r["curriculum_id"],
                                "detail": "approved curriculum had no handoff to the Document Agent"})
        elif missing:
            out.append(_check("flow", "Curriculum → Document handoffs", ERROR,
                              f"{len(missing)} approved curricula were never handed to the Document Agent",
                              [r["curriculum_id"] for r in missing]))
        broken, changed, waiting = [], [], []
        for cid, ev in handed.items():
            row = db.get_curriculum(cid)
            if not row:
                broken.append(f"{ev['event_id']}: curriculum {cid} no longer exists")
            elif row["status"] != "approved":
                changed.append(f"{cid} {row['course_title'][:60]} is now {row['status']} (v{row['version']})")
            elif not db.list_documents(curriculum_id=cid, limit=1) and _age_hours(ev["created_at"]) > self.stale_hours:
                waiting.append(f"{cid} {row['course_title'][:60]} - handed over {ev['created_at'][:10]}")
        if not (missing and repairs is None):
            out.append(_check("flow", "Curriculum → Document handoffs", ERROR if broken else OK,
                              f"{len(handed)} approved curricula handed over" + (f", {len(broken)} broken" if broken else ""),
                              broken))
        if changed:
            out.append(_check("flow", "Curricula changed after handoff", WARNING,
                              "these were revised after approval - re-approve them before making documents", changed))
        out.append(_check("flow", "Document Agent picks up curricula", WARNING if waiting else OK,
                          f"{len(waiting)} approved curricula without any document after {self.stale_hours:g} h"
                          if waiting else "no curriculum waiting too long", waiting[:10]))
        return out

    def _document_flow(self, repairs) -> list[dict]:
        from rex_ai.agents.documents.agent import DocumentAgent
        from rex_ai.core import mailer
        from rex_ai.core.schemas import DocumentEmail
        out = []
        handed = {e["payload_id"] for e in db.list_handoffs_to(AgentName.marketing.value)
                  if e["payload_type"] == "document"}
        approved = db.list_documents(status="approved", limit=10000)
        missing = [d for d in approved if d["document_id"] not in handed]
        if missing and repairs is not None:
            for d in missing:
                db.save_handoff(AgentName.document.value, AgentName.marketing.value, "document", d["document_id"])
                repairs.append({"action": "handoff re-sent", "target": d["document_id"],
                                "detail": "approved document had no handoff to the Marketing Agent"})
            missing = []
        out.append(_check("flow", "Document → Marketing handoffs", ERROR if missing else OK,
                          f"{len(missing)} approved documents were never handed to the Marketing Agent" if missing
                          else f"{len(approved)} approved documents handed over", [d["document_id"] for d in missing]))
        pending = [d for d in approved if (d["content"] or {}).get("metadata", {}).get("pending_email")]
        still = []
        for d in pending:
            to = d["content"]["metadata"]["pending_email"]
            if repairs is not None and mailer.configured():
                try:
                    DocumentAgent().email(d["document_id"], DocumentEmail(to=to, sender="REX orchestrator (retry)"))
                    repairs.append({"action": "e-mail re-sent", "target": d["document_id"], "detail": ", ".join(to)})
                    continue
                except Exception as exc:
                    still.append(f"{d['document_id']} → {', '.join(to)}: {exc}")
            else:
                still.append(f"{d['document_id']} → {', '.join(to)}")
        out.append(_check("flow", "Approved documents e-mailed", WARNING if still else OK,
                          f"{len(still)} approved document(s) still waiting for their e-mail" if still
                          else "no e-mail pending", still[:10]))
        return out

    def _marketing_flow(self, repairs) -> list[dict]:
        out = []
        waiting = []
        for ev in db.list_handoffs_to(AgentName.marketing.value):
            if ev["payload_type"] != "document" or _age_hours(ev["created_at"]) <= self.stale_hours:
                continue
            doc = db.get_document(ev["payload_id"])
            if doc and doc.get("course_id") and not any(r.get("document_id") == doc["document_id"]
                                                        for r in db.list_campaigns(course_id=doc["course_id"])):
                waiting.append(f"{doc['document_id']} {doc['title'][:60]} - handed over {ev['created_at'][:10]}")
        out.append(_check("flow", "Marketing Agent picks up documents", WARNING if waiting else OK,
                          f"{len(waiting)} approved document(s) without a campaign after {self.stale_hours:g} h"
                          if waiting else "no document waiting too long", waiting[:10]))
        campaigns = db.list_campaigns(limit=10000)
        unreviewed = [c for c in campaigns if c["status"] == "pending_approval"
                      and _age_hours(c.get("updated_at") or c.get("created_at")) > self.stale_hours]
        out.append(_check("flow", "Campaigns reviewed", WARNING if unreviewed else OK,
                          f"{len(unreviewed)} campaign(s) waiting over {self.stale_hours:g} h for an administrator"
                          if unreviewed else "no campaign waiting too long",
                          [c["campaign_id"] for c in unreviewed[:10]]))
        return out


# ====================================================================== background loop (API)
class FlowWatcher:
    """Runs Orchestrator().check(repair=True) every REX_ORCH_INTERVAL seconds in a background thread."""

    def __init__(self, interval: Optional[float] = None):
        self.interval = config.ORCH_INTERVAL if interval is None else interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.interval <= 0 or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="rex-orchestrator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        self._stop.wait(5)                          # let the API finish starting
        while not self._stop.is_set():
            try:
                report = Orchestrator().check(repair=True)
                log.info("Flow check: %s %s, %d repair(s)", report["status"], report["summary"],
                         len(report["repairs"]))
            except Exception:
                log.exception("Flow check failed")
            self._stop.wait(self.interval)


def summary_for(report: dict) -> dict[str, Any]:
    """Short form for /health."""
    return {"status": report["status"], "checked_at": report["checked_at"], "summary": report["summary"]}
