"""Optional LLM calls for the Curriculum Agent. Every function returns None when no key is set or the call fails,
and the agent keeps its rule-based curriculum - the workflow never depends on the LLM. Whatever the LLM returns
goes through generator.finalize() (hours, weights, requested outcomes, accreditation-claim guard)."""
from __future__ import annotations

import json
import logging
from typing import Optional

from rex_ai import config

log = logging.getLogger("rex.curriculum.llm")

SYSTEM = (
    "You are the curriculum designer of a training academy in Egypt and Saudi Arabia. Ground every module, topic "
    "and outcome in the DRAFT and the numbered CONTEXT passages (reference syllabi, industry skills, framework "
    "topics); do not invent tools, standards or certifications that are not there. Learning outcomes start with "
    "a measurable action verb. NEVER state or imply that the course is accredited, certified, endorsed or approved "
    "by any body: you may only say it is aligned with a framework. Return JSON only."
)
CURRICULUM_KEYS = ("objectives", "prerequisites", "learning_outcomes", "modules", "practical_exercises",
                   "assessments", "capstone_project")


def enabled() -> bool:
    return config.USE_LLM


def _chat(messages: list[dict]) -> Optional[dict]:
    if not config.USE_LLM:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=0.3,
                                              response_format={"type": "json_object"})
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        log.warning("Curriculum LLM call failed, using the rule-based curriculum: %s", exc)
        return None


def _valid(data: Optional[dict]) -> Optional[dict]:
    if not isinstance(data, dict) or not all(k in data for k in CURRICULUM_KEYS):
        return None
    mods = data.get("modules")
    if not isinstance(mods, list) or not mods or not all(isinstance(m, dict) and m.get("title") for m in mods):
        return None
    return {k: data[k] for k in CURRICULUM_KEYS}


def refine(request: dict, draft: dict, context: str, language: str) -> Optional[dict]:
    """Improve the rule-based draft: richer topics, better exercises and outcomes, same structure."""
    lang = "Arabic" if language == "ar" else "English"
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"REQUEST:\n{json.dumps(request, ensure_ascii=False, default=str)}\n\n"
            f"DRAFT:\n{json.dumps({k: draft[k] for k in CURRICULUM_KEYS}, ensure_ascii=False)}\n\n"
            f"CONTEXT:\n{context or '(no reference passages)'}\n\n"
            f"Improve the draft for the request. Keep the capstone module last and the total hours "
            f"{request.get('total_hours')}. modules = [{{title, topics (3-6), hours, exercises}}]; assessments = "
            f"[{{name, type (quiz|assignment|project|exam), weight_pct}}] totalling 100. Keep every requested outcome. "
            f"Write in {lang}. Return JSON with keys: {', '.join(CURRICULUM_KEYS)}."}])
    return _valid(data)


def revise(curriculum: dict, instructions: str, language: str) -> Optional[dict]:
    """Apply a reviewer's free-text modification request."""
    lang = "Arabic" if language == "ar" else "English"
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"CURRICULUM:\n{json.dumps({k: curriculum.get(k) for k in CURRICULUM_KEYS}, ensure_ascii=False)}\n\n"
            f"TRAINING HOURS: {curriculum.get('training_hours')}\n\nREVIEWER REQUEST:\n{instructions}\n\n"
            f"Apply the reviewer's request and change nothing else. Write in {lang}. "
            f"Return JSON with keys: {', '.join(CURRICULUM_KEYS)}."}])
    return _valid(data)
