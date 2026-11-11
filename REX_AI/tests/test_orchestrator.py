"""Orchestrator Agent and MCP tool server. Run: pytest -q tests/test_orchestrator.py

The agents reach every external service through MCP tools; the orchestrator checks each handoff of the flow and
repairs what is safe to repair. No real LLM, mail server or web page is used.
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent
from rex_ai.agents.documents.agent import DocumentAgent
from rex_ai.agents.marketing.agent import MarketingAgent
from rex_ai.agents.orchestrator import planner, tools as tasks_tools
from rex_ai.agents.orchestrator.agent import Orchestrator
from rex_ai.agents.orchestrator.tasks import TaskConflict, TaskOrchestrator
from rex_ai.core import db
from rex_ai.core.schemas import AgentName, CampaignRequest, CurriculumRequest, DocumentRequest, ReviewAction
from rex_ai.mcp import client as tools

COURSE = dict(course_title="Excel for Accountants", target_audience="Accountants", skill_level="Beginner",
              total_hours=24)


def _by_name(report: dict, name: str) -> dict:
    return next(c for c in report["checks"] if c["name"] == name)


class FakeSMTP:
    sent: list = []

    def __init__(self, host, port, timeout):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, u, p):
        pass

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


# =========================================================================== MCP tool server
def test_every_external_service_is_an_mcp_tool():
    names = {t["name"] for t in tools.list_tools()}
    assert {"llm_chat", "embed_texts", "send_email", "post_webhook", "http_get", "smtp_check",
            "service_status"} <= names


def test_service_status_never_shows_secrets(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret-mail-password")
    status = tools.call("service_status")
    assert status["gemini"]["configured"] is True
    assert "secret" not in json.dumps(status)


def test_tool_errors_keep_their_reason(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(tools.ToolCallError, match="OPENAI_API_KEY"):
        tools.chat("openai", [{"role": "user", "content": "hi"}])
    with pytest.raises(tools.ToolCallError, match="Unknown LLM provider"):
        tools.chat("other", [{"role": "user", "content": "hi"}])


def test_email_goes_through_the_mcp_tool(monkeypatch, tmp_path):
    from rex_ai.core import mailer
    monkeypatch.setattr(config, "SMTP_HOST", "smtp.example")
    monkeypatch.setattr("rex_ai.core.mailer.smtplib.SMTP", FakeSMTP)
    FakeSMTP.sent = []
    pdf = tmp_path / "brief.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    assert mailer.send(["team@academy.example"], "Hello", "Body", [pdf])["sent"]
    [msg] = FakeSMTP.sent
    [att] = list(msg.iter_attachments())
    assert att.get_filename() == "brief.pdf" and att.get_content() == b"%PDF-1.4 test"   # survives base64


def test_web_pages_are_fetched_through_the_mcp_tool(monkeypatch):
    import requests

    class Resp:
        status_code, text, url, headers = 200, "<html>tenders</html>", "https://portal.example/list", {}

    monkeypatch.setattr(requests, "get", lambda url, timeout, headers: Resp())
    r = tools.HttpSession().get("https://portal.example/list")
    assert r.status_code == 200 and "tenders" in r.text


# =========================================================================== orchestrator: routing with tracking
def test_dispatch_turns_a_crash_into_a_clear_error():
    def broken():
        raise RuntimeError("template file missing")

    out = Orchestrator().dispatch(AgentName.document, "document_generation", "make a guide", broken)
    assert out["status"] == "error" and "template file missing" in out["message"]
    job = next(j for j in db.list_jobs(agent="orchestrator") if j["job_id"] == out["job_id"])
    assert job["status"] == "failed" and job["input"]["routed_to"] == "document"


# =========================================================================== orchestrator: flow checks and repairs
def test_missing_handoff_is_found_and_repaired():
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    CurriculumAgent().review(cur.curriculum_id, ReviewAction(action="approved", reviewer="Dean"))
    with db.connect() as conn:                                  # the handoff event got lost
        conn.execute("DELETE FROM handoff_events WHERE payload_id=?", (cur.curriculum_id,))
    report = Orchestrator().check(repair=False)
    broken = _by_name(report, "Curriculum → Document handoffs")
    assert broken["status"] == "error" and cur.curriculum_id in broken["items"]
    report = Orchestrator().check(repair=True)
    assert any(r["target"] == cur.curriculum_id and r["action"] == "handoff re-sent" for r in report["repairs"])
    assert _by_name(report, "Curriculum → Document handoffs")["status"] == "ok"
    assert any(e["to_agent"] == "document" for e in db.list_handoffs(cur.curriculum_id))


def test_stuck_job_is_marked_failed():
    job_id = db.start_job("curriculum", {"note": "process died"})
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE job_runs SET started_at=? WHERE job_id=?", (old, job_id))
    report = Orchestrator().check(repair=True)
    assert any(r["target"] == job_id for r in report["repairs"])
    assert next(j for j in db.list_jobs(limit=500) if j["job_id"] == job_id)["status"] == "failed"


def test_pending_document_email_is_retried(monkeypatch):
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    CurriculumAgent().review(cur.curriculum_id, ReviewAction(action="approved", reviewer="Dean"))
    doc = DocumentAgent().generate(DocumentRequest(doc_type="course_description", curriculum_id=cur.curriculum_id,
                                                   email_to=["coordinator@academy.example"]))
    monkeypatch.setattr(config, "SMTP_HOST", "")
    DocumentAgent().review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))   # e-mail fails
    assert _by_name(Orchestrator().check(repair=True), "Approved documents e-mailed")["status"] == "warning"
    monkeypatch.setattr(config, "SMTP_HOST", "smtp.example")                                 # mail fixed
    monkeypatch.setattr("rex_ai.core.mailer.smtplib.SMTP", FakeSMTP)
    FakeSMTP.sent = []
    report = Orchestrator().check(repair=True)
    assert any(r["target"] == doc.document_id and r["action"] == "e-mail re-sent" for r in report["repairs"])
    assert FakeSMTP.sent[0]["To"] == "coordinator@academy.example"
    assert DocumentAgent().get(doc.document_id).emailed_to == ["coordinator@academy.example"]


def test_report_covers_the_whole_pipeline():
    report = Orchestrator().check(repair=False)
    assert [s["stage"] for s in report["pipeline"]] == ["RFP Research", "REX Assistant (leads)", "Curriculum",
                                                         "Documents", "Marketing"]
    areas = {c["area"] for c in report["checks"]}
    assert {"infrastructure", "tools", "agents", "jobs", "flow"} <= areas
    assert _by_name(report, "MCP tool server")["status"] == "ok"


# =========================================================================== API
def test_orchestrator_api():
    from rex_ai.api.main import app
    c = TestClient(app)
    assert c.post("/orchestrator/check", params={"repair": False}).json()["checks"]
    assert c.get("/orchestrator/status").status_code == 200
    assert {t["name"] for t in c.get("/orchestrator/tools").json()["tools"]} >= {"llm_chat", "send_email"}
    assert isinstance(c.get("/orchestrator/jobs").json(), list)
    assert c.get("/health").json()["mcp_tools"] == "in-process"


# =========================================================================== orchestrator: administrator tasks
class FakePlanner:
    """Stands in for the LLM: gives the listed decisions in order and records what it was shown."""

    def __init__(self, monkeypatch, decisions):
        self.decisions, self.prompts = list(decisions), []
        monkeypatch.setattr(config, "USE_GEMINI", True)     # the agents only use Gemini for change requests
        monkeypatch.setattr(planner.mcp, "chat", self.chat)

    def chat(self, provider, messages, temperature=0.3, json_mode=False):
        self.prompts.append(messages[-1]["content"])
        next_one = self.decisions.pop(0)
        if isinstance(next_one, Exception):
            raise next_one
        return next_one if isinstance(next_one, str) else json.dumps(next_one)


def call(tool, **args):
    return {"action": "call", "thought": "next", "tool": tool, "args": args}


FINISH = {"action": "finish", "summary": "All done."}


def _run(message, created_by="Dean"):
    orch = TaskOrchestrator()
    return orch, orch.run(orch.submit(message, created_by)["task_id"])


def _tools(task):
    return [s["tool"] for s in task["steps"]]


def test_llm_chains_agents_using_real_ids(monkeypatch):
    class Script(FakePlanner):
        def chat(self, provider, messages, temperature=0.3, json_mode=False):
            self.prompts.append(messages[-1]["content"])
            if len(self.prompts) == 1:
                return json.dumps(call("create_curriculum", **COURSE))
            if len(self.prompts) == 2:
                cid = re.search(r"CUR-[0-9A-F]{8}", messages[-1]["content"]).group(0)   # read from the step result
                return json.dumps(call("create_document", doc_type="participant_guide", curriculum_id=cid))
            if len(self.prompts) == 3:
                return json.dumps(call("create_campaign", course_id="DS-101"))
            return json.dumps(FINISH)

    Script(monkeypatch, [])
    _, task = _run("Build a curriculum, a guide and a campaign")
    assert task["status"] == "completed" and task["summary"] == "All done."
    assert _tools(task) == ["create_curriculum", "create_document", "create_campaign"]
    assert sorted(a["kind"] for a in task["artifacts"].values()) == ["campaign", "curriculum", "document"]
    doc = next(a for a in task["artifacts"].values() if a["kind"] == "document")
    cur = next(a for a in task["artifacts"].values() if a["kind"] == "curriculum")
    assert DocumentAgent().get(doc["id"]).curriculum_id == cur["id"]
    assert cur["status"] != "approved"                                   # nobody approved it
    assert any(j["input"].get("routed_to") == "curriculum" for j in db.list_jobs(agent="orchestrator", limit=50))


def test_llm_can_correct_a_failed_step(monkeypatch):
    fake = FakePlanner(monkeypatch, [call("create_document", doc_type="not_a_type", course_id="DS-101"),
                                     call("no_such_tool"), "not json",
                                     call("get_curriculum", curriculum_id="CUR-00000000"), FINISH])
    _, task = _run("Make a document")
    assert task["status"] == "completed" and [s["status"] for s in task["steps"]] == ["failed"] * 4
    assert "doc_type" in task["steps"][0]["error"] and "not found" in task["steps"][3]["error"]
    assert "not found" in fake.prompts[-1]                               # the planner saw the error


def test_step_limit_stops_a_runaway_plan(monkeypatch):
    FakePlanner(monkeypatch, [call("search_courses", query="data")] * 5)
    orch = TaskOrchestrator(max_steps=3)
    task = orch.run(orch.submit("Loop", "Dean")["task_id"])
    assert task["status"] == "failed" and len(task["steps"]) == 3 and "3 steps" in task["error"]


def test_repeating_a_create_step_does_not_create_twice(monkeypatch):
    FakePlanner(monkeypatch, [call("create_campaign", course_id="DS-101"), call("create_campaign", course_id="DS-101"),
                              FINISH])
    before = len(db.list_campaigns(limit=10000))
    _, task = _run("Campaign")
    assert len(db.list_campaigns(limit=10000)) == before + 1 and "already succeeded" in task["steps"][1]["error"]


def test_questions_pause_the_task_and_answers_reach_the_planner(monkeypatch):
    fake = FakePlanner(monkeypatch, [{"action": "ask", "questions": ["Which course?"]}, FINISH])
    orch, task = _run("Promote our course")
    assert task["status"] == "needs_input" and task["questions"] == ["Which course?"]
    orch.answer(task["task_id"], ["DS-101"])
    assert orch.run(task["task_id"])["status"] == "completed"
    assert "DS-101" in fake.prompts[-1]
    with pytest.raises(TaskConflict):
        orch.answer(task["task_id"], ["again"])


def _approve_curriculum_task(monkeypatch):
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    FakePlanner(monkeypatch, [call("review_curriculum", curriculum_id=cur.curriculum_id, decision="approved"), FINISH])
    orch, task = _run("Approve it", created_by="Asked by Sam")
    return orch, task, cur


def test_approval_waits_for_a_person_and_records_who_confirmed(monkeypatch):
    orch, task, cur = _approve_curriculum_task(monkeypatch)
    assert task["status"] == "awaiting_approval" and cur.curriculum_id in task["pending"]["description"]
    assert CurriculumAgent().get(cur.curriculum_id).status.value != "approved"      # nothing ran yet
    orch.decide(task["task_id"], True, "Dean Nour")
    assert orch.run(task["task_id"])["status"] == "completed"
    done = CurriculumAgent().get(cur.curriculum_id)
    assert done.status.value == "approved" and done.metadata["approved_by"] == "Dean Nour"
    assert db.get_task(task["task_id"])["steps"][0]["approved_by"] == "Dean Nour"


def test_a_second_confirmation_cannot_run_the_action_twice(monkeypatch):
    orch, task, _ = _approve_curriculum_task(monkeypatch)
    orch.decide(task["task_id"], True, "Dean")
    with pytest.raises(TaskConflict):
        orch.decide(task["task_id"], True, "Dean again")


def test_declining_stops_the_task_and_changes_nothing(monkeypatch):
    orch, task, cur = _approve_curriculum_task(monkeypatch)
    done = orch.decide(task["task_id"], False, "Dean")
    assert done["status"] == "cancelled" and done["steps"][0]["status"] == "declined" and "declined" in done["summary"]
    assert CurriculumAgent().get(cur.curriculum_id).status.value != "approved"


def test_publishing_shows_the_recipients_before_anything_is_sent(monkeypatch):
    camp = MarketingAgent().generate(CampaignRequest(course_id="DS-101"))
    MarketingAgent().review(camp.campaign_id, ReviewAction(action="approved", reviewer="Dean"))
    FakePlanner(monkeypatch, [call("publish_campaign", campaign_id=camp.campaign_id, email_to=["all@academy.example"])])
    _, task = _run("Publish it")
    assert task["status"] == "awaiting_approval" and "all@academy.example" in task["pending"]["description"]
    assert not MarketingAgent().get(camp.campaign_id).metadata.get("publications")


def test_a_gated_action_on_a_wrong_id_never_reaches_the_administrator(monkeypatch):
    FakePlanner(monkeypatch, [call("review_campaign", campaign_id="CMP-00000000", decision="approved"), FINISH])
    _, task = _run("Approve it")
    assert task["status"] == "completed" and task["steps"][0]["status"] == "failed" and task["pending"] is None


class EmptyRun:
    found = kept = duplicates = expired = 0
    new_flagged_egypt, new_monitoring_saudi, records, source_errors = [], [], [], []


def test_planner_cannot_set_hidden_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks_tools.RFPResearchAgent, "run", lambda self, req: calls.append(req) or EmptyRun())
    FakePlanner(monkeypatch, [call("run_tender_research", notify=True),
                              call("create_document", doc_type="course_description", course_id="DS-101",
                                   email_to=["x@evil.example"]), FINISH])
    _, task = _run("Find tenders, write a description")
    assert calls[0].notify is False
    doc = next(a for a in task["artifacts"].values() if a["kind"] == "document")
    assert DocumentAgent().get(doc["id"]).metadata["pending_email"] == []


def test_every_read_tool_works():
    needs_id = {"get_curriculum", "get_document", "get_campaign", "get_tender"}
    args = {"ask_course_assistant": {"question": "How much is DS-101?"}, "ask_tenders": {"question": "any tenders?"}}
    ctx = tasks_tools.Context(task_id="TSK-READS", admin="Dean")
    for tool in (t for t in tasks_tools.TOOLS.values() if t.kind == tasks_tools.READ and t.name not in needs_id):
        assert isinstance(tool.run(tool.args.model_validate(args.get(tool.name, {})), ctx), dict), tool.name
    sig = tasks_tools.signature(tasks_tools.TOOLS["create_document"])
    assert sig.startswith("create_document(doc_type: course_description|") and "email_to" not in sig


def test_llm_failure_falls_back_to_rules_before_any_step(monkeypatch):
    FakePlanner(monkeypatch, [tools.ToolCallError("insufficient_quota")])
    _, task = _run("Create a campaign to promote DS-101")
    assert task["status"] == "completed" and task["mode"] == "rules" and _tools(task) == ["create_campaign"]
    assert "insufficient_quota" in task["notes"][0]


def test_gemini_plans_when_openai_fails(monkeypatch):
    FakePlanner(monkeypatch, [tools.ToolCallError("insufficient_quota"), "```json\n" + json.dumps(FINISH) + "\n```"])
    monkeypatch.setattr(config, "USE_LLM", True)
    _, task = _run("Anything")
    assert task["status"] == "completed" and task["mode"] == "llm" and task["summary"] == "All done."


def test_llm_failure_midway_keeps_the_work_done(monkeypatch):
    FakePlanner(monkeypatch, [call("create_campaign", course_id="DS-101"), tools.ToolCallError("insufficient_quota")])
    _, task = _run("Campaign and more")
    assert task["status"] == "failed" and "insufficient_quota" in task["error"] and len(task["artifacts"]) == 1


def test_rules_chain_curriculum_document_and_campaign():
    _, task = _run("Design a curriculum for DS-101, then a participant guide and a question bank and a campaign "
                   "to promote it")
    assert task["status"] == "completed", task
    assert _tools(task) == ["create_curriculum", "create_document", "create_document", "create_campaign"]
    cur = next(a["id"] for a in task["artifacts"].values() if a["kind"] == "curriculum")
    assert task["steps"][1]["args"]["curriculum_id"] == cur == task["steps"][3]["args"]["curriculum_id"]
    assert cur in task["summary"] and "none of them was approved" in task["summary"]


def test_rules_do_not_drop_a_curriculum_named_in_a_longer_request():
    orch, task = _run("Create a full package for DS-101: a curriculum, a participant guide and a campaign to promote "
                      "it. Then approve the curriculum.")
    assert _tools(task) == ["create_curriculum", "review_curriculum"] and task["status"] == "awaiting_approval"
    cur = next(a for a in task["artifacts"].values() if a["kind"] == "curriculum")
    assert cur["title"] != "DS-101" and cur["id"] in task["pending"]["description"]
    orch.decide(task["task_id"], True, "Dean")
    assert _tools(orch.run(task["task_id"])) == ["create_curriculum", "review_curriculum", "create_document",
                                                  "create_campaign"]
    # other test files expect DS-101 to have no approved curriculum
    CurriculumAgent().review(cur["id"], ReviewAction(action="rejected", reviewer="Test clean-up"))


def test_rules_ask_for_what_is_missing_and_use_bare_answers():
    orch, task = _run("Create a new curriculum please")
    assert task["status"] == "needs_input" and not task["steps"] and len(task["questions"]) == 4
    orch.answer(task["task_id"], ["Forklift Safety", "Warehouse staff", "beginner", "16"])
    done = orch.run(task["task_id"])
    assert done["status"] == "completed", done
    assert done["steps"][0]["args"]["course_title"] == "Forklift Safety"
    cur = next(a for a in done["artifacts"].values() if a["kind"] == "curriculum")
    assert CurriculumAgent().get(cur["id"]).training_hours == 16


def test_rules_document_request_names_its_curriculum_without_creating_one():
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    _, task = _run(f"Write a facilitator guide for curriculum {cur.curriculum_id}")
    assert _tools(task) == ["create_document"] and task["steps"][0]["args"]["curriculum_id"] == cur.curriculum_id


def test_rules_gate_decisions_named_by_id():
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    orch, task = _run(f"Approve {cur.curriculum_id}")
    assert task["status"] == "awaiting_approval"
    orch.decide(task["task_id"], True, "Dean")
    assert orch.run(task["task_id"])["status"] == "completed"
    assert CurriculumAgent().get(cur.curriculum_id).status.value == "approved"


def test_rules_stop_at_the_first_failure():
    _, task = _run("Approve CUR-00000000")
    assert task["status"] == "failed" and "not found" in task["error"]


def test_a_dollar_amount_is_not_taken_for_a_reference(monkeypatch):
    FakePlanner(monkeypatch, [call("create_campaign", course_id="DS-101", offer="$100 off"), FINISH])
    _, task = _run("Campaign")
    assert task["steps"][0]["args"]["offer"] == "$100 off"


def test_course_questions_go_to_the_assistant():
    _, task = _run("How much does DS-101 cost?")
    assert _tools(task) == ["ask_course_assistant"] and task["summary"]


def test_task_with_a_dead_process_is_marked_failed():
    orch = TaskOrchestrator()
    task = orch.submit("Never run", "Dean")
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE orchestrator_tasks SET updated_at=? WHERE task_id=?", (old, task["task_id"]))
    report = Orchestrator().check(repair=True)
    assert any(r["target"] == task["task_id"] for r in report["repairs"])
    assert orch.get(task["task_id"])["status"] == "failed"


def _wait(client, task_id):
    for _ in range(600):
        task = client.get(f"/orchestrator/tasks/{task_id}").json()
        if task["status"] != "running":
            return task
        time.sleep(0.1)
    raise AssertionError("the task did not finish")


def test_tasks_api_runs_pauses_and_resumes():
    from rex_ai.api.main import app
    c = TestClient(app)
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    r = c.post("/orchestrator/tasks", json={"message": f"Approve {cur.curriculum_id}"})
    assert r.status_code == 202
    task = _wait(c, r.json()["task_id"])
    assert task["status"] == "awaiting_approval"
    assert c.post(f"/orchestrator/tasks/{task['task_id']}/answers", json={"answers": ["x"]}).status_code == 409
    assert c.post(f"/orchestrator/tasks/{task['task_id']}/decision", json={"approve": True}).status_code == 200
    assert _wait(c, task["task_id"])["status"] == "completed"
    assert c.post(f"/orchestrator/tasks/{task['task_id']}/decision", json={"approve": True}).status_code == 409
    assert c.get("/orchestrator/tasks/TSK-MISSING").status_code == 404
    assert c.post("/orchestrator/tasks/TSK-MISSING/decision", json={"approve": True}).status_code == 404
    assert any(t["task_id"] == task["task_id"] for t in c.get("/orchestrator/tasks").json())
    assert CurriculumAgent().get(cur.curriculum_id).metadata["approved_by"] == "administrator"
