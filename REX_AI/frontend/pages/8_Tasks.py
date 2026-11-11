"""Orchestrator Tasks (administrators only, owner: Member 3): say what you need, the agents do it.

The page starts a task (rex_ai/agents/orchestrator/tasks.py), follows it step by step, and asks for what only a person
can give: a missing detail, or the confirmation of an approval, e-mail or publication.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.ui import api, api_up, no_fog, require  # noqa: E402
from rex_ai import config  # noqa: E402
from rex_ai.core import db  # noqa: E402

st.set_page_config(page_title="REX - Orchestrator Tasks", page_icon="🤖", layout="wide")
no_fog()
ME = require("admin")
BASE = f"{config.API_URL}/orchestrator/tasks"
TASK_ICON = {"running": "⏳", "needs_input": "❓", "awaiting_approval": "✋", "completed": "✅", "failed": "❌",
             "cancelled": "🚫"}
STEP_ICON = {"running": "⏳", "awaiting_approval": "✋", "succeeded": "✅", "failed": "❌", "declined": "🚫"}
PAGE_OF = {"curriculum": ("pages/3_Curriculum.py", "📚"), "document": ("pages/4_Documents.py", "📄"),
           "campaign": ("pages/5_Marketing.py", "📣")}
EXAMPLES = ["Create a curriculum for Power BI for finance staff, beginner, 30 hours",
            "Write a participant guide and a question bank for DS-101",
            "Create a LinkedIn and WhatsApp campaign to promote DS-101 for fresh graduates, budget 5000 EGP",
            "Which open tenders in Egypt need cybersecurity training? Design a curriculum for the best match."]


def _send(method: str, path: str, **kw) -> dict | list:
    r = api().request(method, BASE + path, timeout=30, **kw)
    if r.status_code >= 400:
        raise ValueError(r.json().get("message", r.text))
    return r.json()


def _local():
    """Without the API the same engine runs inside this app."""
    from rex_ai.agents.orchestrator.tasks import TaskOrchestrator
    return TaskOrchestrator()


def _continue_local(task: dict) -> dict:
    if task["status"] == "running":
        _local().start(task["task_id"])
    return task


def create(message: str) -> dict:
    if api_up():
        return _send("POST", "", json={"message": message})
    return _continue_local(_local().submit(message, ME["name"]))


FINAL = ("completed", "failed", "cancelled")


def fetch(task_id: str) -> dict:
    """A finished task never changes again, so it is fetched once per browser session."""
    done = st.session_state.setdefault("finished_tasks", {})
    if task_id in done:
        return done[task_id]
    task = _send("GET", f"/{task_id}") if api_up() else _local().get(task_id)
    if task["status"] in FINAL:
        done[task_id] = task
    return task


@st.cache_data(ttl=15, show_spinner=False)
def recent() -> list[dict]:
    return _send("GET", "", params={"limit": 20}) if api_up() else db.list_tasks(limit=20)


def answer(task_id: str, answers: list[str]) -> dict:
    if api_up():
        return _send("POST", f"/{task_id}/answers", json={"answers": answers})
    return _continue_local(_local().answer(task_id, answers))


def decide(task_id: str, approve: bool) -> dict:
    if api_up():
        return _send("POST", f"/{task_id}/decision", json={"approve": approve})
    return _continue_local(_local().decide(task_id, approve, ME["name"]))


def act(fn) -> None:
    """Run a button's action and redraw the page; if it was already handled, say so and show the new state."""
    try:
        fn()
    except (ValueError, httpx.HTTPError) as exc:
        st.session_state.task_error = str(exc)
    recent.clear()
    st.rerun()


def show_steps(task: dict) -> None:
    for s in task["steps"]:
        title = f"{STEP_ICON.get(s['status'], '•')} {s['n']}. {s['tool'].replace('_', ' ')}"
        if s.get("agent"):
            title += f" · {s['agent'].replace('_', ' ')}"
        with st.expander(title, expanded=s["status"] == "failed"):
            if s.get("thought"):
                st.caption(s["thought"])
            if s.get("description"):
                st.write(s["description"])
            for who, label in (("approved_by", "Confirmed by"), ("declined_by", "Declined by")):
                if s.get(who):
                    st.caption(f"{label} {s[who]}")
            if s.get("error"):
                st.error(s["error"])
            st.json({k: s[k] for k in ("args", "result") if s.get(k)}, expanded=False)


def show_waiting(task: dict) -> None:
    task_id = task["task_id"]
    if task["status"] == "awaiting_approval":
        st.warning(f"**Your confirmation is needed**\n\n{task['pending']['description']}", icon="✋")
        yes, no, _ = st.columns([1, 1, 4])
        if yes.button("Confirm", type="primary", key=f"yes_{task_id}"):
            act(lambda: decide(task_id, True))
        if no.button("Decline", key=f"no_{task_id}"):
            act(lambda: decide(task_id, False))
    elif task["status"] == "needs_input":
        with st.form(f"answers_{task_id}_{len(task['answers'])}"):
            st.markdown("**The orchestrator needs a few details**")
            replies = [st.text_input(q, key=f"a_{task_id}_{len(task['answers'])}_{i}")
                       for i, q in enumerate(task["questions"])]
            if st.form_submit_button("Continue", type="primary"):
                if all(r.strip() for r in replies):
                    act(lambda: answer(task_id, replies))
                else:
                    st.error("Please answer every question.")


def show_results(task: dict) -> None:
    if task.get("error"):
        st.error(task["error"])
    if task.get("summary"):
        st.markdown(task["summary"])
    for note in task.get("notes", []):
        st.caption(note)
    for a in task.get("artifacts", {}).values():
        page, icon = PAGE_OF[a["kind"]]
        left, right = st.columns([3, 1])
        left.markdown(f"{icon} **{a['title']}**  \n`{a['id']}` · {a['status'].replace('_', ' ')}")
        right.page_link(page, label=f"Open {a['kind']} page")


def show_task(task_id: str) -> None:
    first = {"task": fetch(task_id)}
    live = first["task"]["status"] == "running"

    @st.fragment(run_every=2 if live else None)
    def body() -> None:
        task = first.pop("task", None) or fetch(task_id)   # fetched once per page run; again on each refresh
        if (task["status"] == "running") != live:
            recent.clear()                              # its icon in the task list changes too
            st.rerun()                                  # redraw the whole page once, with or without the timer
        st.subheader(f"{TASK_ICON[task['status']]} {task['status'].replace('_', ' ').capitalize()}")
        st.markdown(f"> {task['message']}")
        show_results(task)
        show_waiting(task)
        show_steps(task)
        if live:
            st.caption("Working... this page updates by itself.")

    body()


st.title("🤖 Orchestrator Tasks")
st.caption("Tell the orchestrator what you need: it picks the agents and runs them in order. Drafts are created right "
           "away; approving, e-mailing and publishing wait for your confirmation.")
if not (config.USE_LLM or config.USE_GEMINI):
    st.info("No LLM is set up, so tasks are planned by rules: creating curricula, documents and campaigns, tender "
            "questions, and approving or publishing by id. Set OPENAI_API_KEY or GEMINI_API_KEY to allow any request.", icon="ℹ️")

with st.form("new_task", clear_on_submit=True):
    message = st.text_area("Your task", height=100, max_chars=4000, placeholder=EXAMPLES[0])
    if st.form_submit_button("Run task", type="primary") and message.strip():
        try:
            st.session_state.task_id = create(message.strip())["task_id"]
            recent.clear()
        except (ValueError, httpx.HTTPError) as exc:
            st.error(f"The task could not be started: {exc}")
with st.expander("Examples"):
    st.markdown("\n".join(f"- {e}" for e in EXAMPLES))

if error := st.session_state.pop("task_error", None):
    st.error(error)

try:
    history = recent()
    if history:
        labels = {t["task_id"]: f"{TASK_ICON[t['status']]} {t['task_id']} - {t['message'][:70]}" for t in history}
        if st.session_state.get("task_id") not in labels:
            st.session_state.task_id = history[0]["task_id"]
        st.selectbox("Task", list(labels), format_func=labels.get, key="task_id")
        show_task(st.session_state.task_id)
    else:
        st.write("No tasks yet.")
except (ValueError, httpx.HTTPError) as exc:
    st.error(f"The orchestrator could not be reached: {exc}")
