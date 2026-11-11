"""Task orchestrator: an administrator states a goal, the other agents carry it out (owner: Member 3).

    orch = TaskOrchestrator()
    task = orch.submit("Create a curriculum for Power BI for finance staff, beginner, 30 hours, and a participant guide",
                       created_by="Dean")
    orch.run(task["task_id"])       # the API runs it in a background thread: orch.start(task["task_id"])

A task is saved after every step, so its progress can be followed and it can pause and carry on from any server:

    running -> completed | failed
            -> needs_input        a required value is missing       answer()  -> running
            -> awaiting_approval  a gated tool was chosen           decide()  -> running | cancelled

Drafts are created without asking. Whatever leaves the academy or records a human decision (tools.GATED) runs only
after an administrator confirms that exact action, and is recorded under that administrator's name.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.orchestrator import planner
from rex_ai.agents.orchestrator.agent import Orchestrator
from rex_ai.agents.orchestrator.tools import GATED, READ, TOOLS, Context, Tool
from rex_ai.agents.rex.text_utils import detect_language
from rex_ai.core import db
from rex_ai.core.schemas import AgentName
from rex_ai.mcp.client import ToolCallError

log = logging.getLogger("rex.orchestrator.tasks")
AGENT = AgentName.orchestrator.value
RUNNING, NEEDS_INPUT, AWAITING, COMPLETED, FAILED, CANCELLED = ("running", "needs_input", "awaiting_approval",
                                                                "completed", "failed", "cancelled")
SUCCEEDED, STEP_FAILED, DECLINED = "succeeded", "failed", "declined"
# Most specific first: a document also names its curriculum, a campaign may name both
ARTIFACT_IDS = (("campaign", "campaign_id"), ("document", "document_id"), ("curriculum", "curriculum_id"))


class TaskConflict(ValueError):
    """The task is not waiting for this (already answered, confirmed by someone else, or finished)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors())


class TaskOrchestrator:
    def __init__(self, max_steps: Optional[int] = None):
        self.max_steps = config.ORCH_MAX_STEPS if max_steps is None else max_steps

    # ================================================================== what the API and the page call
    def submit(self, message: str, created_by: str) -> dict:
        now = _now()
        task = {"task_id": "TSK-" + uuid.uuid4().hex[:8].upper(), "status": RUNNING, "message": message,
                "created_by": created_by, "error": None, "created_at": now, "mode": "llm" if planner.enabled() else "rules",
                "steps": [], "questions": [], "answers": [], "pending": None, "summary": None, "artifacts": {},
                "notes": []}
        return db.save_task(task)

    def get(self, task_id: str) -> dict:
        task = db.get_task(task_id)
        if not task:
            raise LookupError(f"Task {task_id} not found")
        return task

    def answer(self, task_id: str, answers: list[str]) -> dict:
        """Take the answers to the task's questions; start() or run() it again afterwards."""
        self.get(task_id)
        if not db.claim_task(task_id, NEEDS_INPUT):
            raise TaskConflict(f"Task {task_id} is not waiting for an answer")
        task = self.get(task_id)
        task["answers"] += [{"question": q, "answer": a} for q, a in zip(task["questions"], answers)]
        for field, value in zip(task.get("question_fields", []), answers):
            if value.strip():
                task.setdefault("known", {})[field] = value.strip()
        task["questions"], task["question_fields"] = [], []
        task.pop("plan", None)
        return db.save_task(task)

    def decide(self, task_id: str, approve: bool, admin: str) -> dict:
        """Confirm or decline the action the task paused on. Confirmed: start() or run() it, and it executes first."""
        self.get(task_id)
        if not db.claim_task(task_id, AWAITING):             # atomic: a second click or server gets the error
            raise TaskConflict(f"Task {task_id} is not waiting for a confirmation")
        task = self.get(task_id)
        if approve:
            task["pending"]["approved_by"] = admin
            return db.save_task(task)
        step = task["steps"][task["pending"]["step"] - 1]
        step.update(status=DECLINED, finished_at=_now(), declined_by=admin)
        task["pending"] = None
        task["summary"] = self._closing(task, stopped=f"{admin} declined: {step['description']}")
        task["status"] = CANCELLED
        db.log_event("INFO", "task_declined", agent=AGENT, detail=f"{task_id} {step['tool']} by {admin}"[:500])
        return db.save_task(task)

    def start(self, task_id: str) -> None:
        threading.Thread(target=self.run, args=(task_id,), name=f"rex-task-{task_id}", daemon=True).start()

    def run(self, task_id: str) -> dict:
        """Work on the task until it finishes or needs a person. Never raises: a crash becomes a failed task."""
        task = self.get(task_id)
        try:
            if task["status"] == RUNNING:
                if task.get("pending"):
                    self._run_confirmed(task)
                self._loop(task)
        except Exception as exc:
            log.exception("Task %s failed", task_id)
            self._stop(task, FAILED, error=f"{type(exc).__name__}: {exc}"[:1000])
        return task

    # ================================================================== the loop
    def _loop(self, task: dict) -> None:
        while task["status"] == RUNNING:
            if len(task["steps"]) >= self.max_steps:
                self._stop(task, FAILED, error=f"Stopped after {self.max_steps} steps without finishing "
                                               "(REX_ORCH_MAX_STEPS). The work done so far is kept.")
                return
            action = self._decide(task)
            if action["action"] == "finish":
                summary = action.get("summary") or self._closing(task)
                self._stop(task, COMPLETED, summary=summary)
            elif action["action"] == "ask":
                task["questions"] = [str(q) for q in action["questions"]]
                task["question_fields"] = action.get("fields", [])
                task["status"] = NEEDS_INPUT
                db.save_task(task)
            elif action["action"] == "invalid":
                self._record(task, "(planner)", {}, None, status=STEP_FAILED, error=action["error"])
            else:
                self._call(task, action)
                self._stop_rules_on_failure(task)

    def _decide(self, task: dict) -> dict:
        if task["mode"] == "llm":
            try:
                return planner.llm_next(task)
            except ToolCallError as exc:
                if task["steps"]:                        # part of the work follows the LLM's plan: do not guess on
                    raise RuntimeError(f"The planning LLM stopped responding: {exc}") from exc
                task["mode"] = "rules"
                task["notes"].append(f"Planned with rules because the LLM failed: {exc}"[:300])
        return self._next_rule_action(task)

    def _next_rule_action(self, task: dict) -> dict:
        if "plan" not in task:
            built = planner.rule_plan(task["message"], [a["answer"] for a in task["answers"]], task.get("known", {}))
            if built.get("questions"):
                return {"action": "ask", "questions": built["questions"], "fields": built["fields"]}
            task["plan"] = built["plan"]
        if task["plan"]:
            step = task["plan"].pop(0)
            return {"action": "call", "tool": step["tool"], "args": step["args"]}
        return {"action": "finish"}

    def _stop_rules_on_failure(self, task: dict) -> None:
        last = task["steps"][-1]
        if task["status"] == RUNNING and task["mode"] == "rules" and last["status"] == STEP_FAILED:
            self._stop(task, FAILED, error=f"{last['tool']}: {last['error']}"[:1000])

    # ================================================================== one step
    def _record(self, task: dict, tool: str, args: dict, agent: Optional[str], thought: str = "", **fields) -> dict:
        step = {"n": len(task["steps"]) + 1, "tool": tool, "agent": agent, "args": args, "thought": thought,
                "started_at": _now(), **fields}
        task["steps"].append(step)
        db.save_task(task)
        return step

    def _call(self, task: dict, action: dict) -> None:
        name, thought = action["tool"], str(action.get("thought") or "")[:300]
        args = {k: task.get("last_curriculum") if v == planner.LAST_CURRICULUM else v
                for k, v in (action.get("args") or {}).items()}
        tool = TOOLS.get(name)
        if tool is None:
            self._record(task, name, args, None, thought, status=STEP_FAILED,
                         error=f"There is no tool named {name}. Tools: {', '.join(TOOLS)}")
            return
        step_args = {k: v for k, v in args.items() if k not in tool.fixed}
        try:
            parsed = tool.args.model_validate({**args, **tool.fixed})
        except ValidationError as exc:
            self._record(task, name, step_args, tool.agent.value, thought, status=STEP_FAILED,
                         error=f"Invalid arguments - {_errors(exc)}")
            return
        if tool.kind != READ and any(s["tool"] == name and s["args"] == step_args and s["status"] == SUCCEEDED
                                     for s in task["steps"]):
            self._record(task, name, step_args, tool.agent.value, thought, status=STEP_FAILED,
                         error="This exact step already succeeded in this task; use its result.")
            return
        if tool.kind == GATED:
            self._pause(task, tool, parsed, step_args, thought)
            return
        step = self._record(task, name, step_args, tool.agent.value, thought, status=RUNNING)
        self._execute(task, step, tool, parsed, task["created_by"])

    def _pause(self, task: dict, tool: Tool, parsed: Any, args: dict, thought: str) -> None:
        try:
            description = tool.describe(parsed)
        except Exception as exc:                         # e.g. a wrong id: nothing to ask the administrator about
            self._record(task, tool.name, args, tool.agent.value, thought, status=STEP_FAILED, error=str(exc))
            return
        step = self._record(task, tool.name, args, tool.agent.value, thought, status=AWAITING, description=description)
        task["pending"] = {"step": step["n"], "tool": tool.name, "description": description}
        task["status"] = AWAITING
        db.save_task(task)

    def _run_confirmed(self, task: dict) -> None:
        pending, task["pending"] = task["pending"], None
        step = task["steps"][pending["step"] - 1]
        tool = TOOLS[step["tool"]]
        step.update(status=RUNNING, approved_by=pending["approved_by"])
        db.save_task(task)
        self._execute(task, step, tool, tool.args.model_validate({**step["args"], **tool.fixed}), pending["approved_by"])
        self._stop_rules_on_failure(task)

    def _execute(self, task: dict, step: dict, tool: Tool, parsed: Any, admin: str) -> None:
        ctx = Context(task_id=task["task_id"], admin=admin)
        if tool.kind == READ:
            try:
                result = tool.run(parsed, ctx)
            except Exception as exc:
                log.warning("Task %s: %s failed: %s", task["task_id"], tool.name, exc)
                result = {"status": "error", "message": str(exc)}
        else:                                            # tracked as a job, like a routed request
            result = Orchestrator().dispatch(tool.agent, tool.name, f"{task['task_id']}: {task['message']}",
                                             lambda: tool.run(parsed, ctx))
        step["finished_at"] = _now()
        if result.get("status") == "error":
            step.update(status=STEP_FAILED, error=result.get("message") or "The agent failed")
        else:
            step.update(status=SUCCEEDED, result=result)
            if tool.kind != READ:
                self._collect(task, result)
        db.save_task(task)

    @staticmethod
    def _collect(task: dict, result: dict) -> None:
        """Remember what the task created or changed; a later step on the same record updates its status."""
        for item in [result, *result.get("documents", [])]:
            kind, key = next(((k, i) for k, i in ARTIFACT_IDS if item.get(i)), (None, None))
            if kind and "title" in item:                 # a gap report names a curriculum but is not one
                task["artifacts"][item[key]] = {"kind": kind, "id": item[key], "title": item["title"],
                                                "status": item.get("status"), "version": item.get("version"),
                                                "download_url": item.get("download_url")}
                if kind == "curriculum":
                    task["last_curriculum"] = item[key]

    def _stop(self, task: dict, status: str, summary: Optional[str] = None, error: Optional[str] = None) -> None:
        task.update(status=status, error=error, pending=None)
        task["summary"] = summary if summary is not None else self._closing(task)
        db.log_event("ERROR" if status == FAILED else "INFO", f"task_{status}", agent=AGENT,
                     detail=f"{task['task_id']}: {len(task['steps'])} step(s) {error or ''}"[:500])
        db.save_task(task)

    @staticmethod
    def _closing(task: dict, stopped: str = "") -> str:
        """The summary when no LLM wrote one."""
        ar = detect_language(task["message"]) == "ar"
        lines = [stopped] if stopped else []
        if task.get("error"):
            lines.append(task["error"])
        for s in task["steps"]:
            if s["status"] == SUCCEEDED and isinstance(s.get("result"), dict):
                if text := s["result"].get("answer"):
                    lines.append(text)
        if task["artifacts"]:
            lines.append("النتائج:" if ar else "Results:")
            lines += [f"- {a['kind']} {a['id']}: {a['title']} ({a['status']})" for a in task["artifacts"].values()]
            if any(a["status"] in ("draft", "pending_approval") for a in task["artifacts"].values()):
                lines.append("هذه مسودات تنتظر المراجعة، ولم يتم اعتماد أي منها أو نشره." if ar else
                             "The drafts are waiting for review; none of them was approved or published.")
        return "\n".join(lines) or ("تمت المهمة." if ar else "Done.")
