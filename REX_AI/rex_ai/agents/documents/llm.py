"""Optional LLM calls for the Document Agent. Each function returns None when no key is set or the call fails, and
the agent keeps the rule-based sections - the workflow never depends on the LLM.

Safety: only UNLOCKED sections are sent and accepted back (locked ones hold fees, hours, weights and schedules from
approved data); every returned section must keep its key and table shape (merge_sections), and the agent runs the
accreditation-claim guard on the result.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.core.schemas import DocSection

log = logging.getLogger("rex.documents.llm")

SYSTEM = (
    "You write training documents for an academy in Egypt and Saudi Arabia. Use ONLY the facts in FACTS, the "
    "sections given, and the numbered CONTEXT passages (reference syllabi and the academy's reference examples); do "
    "not invent fees, dates, hours, tools, instructors, certifications or statistics. Follow the house style of the "
    "reference examples. NEVER state or imply that the course is accredited, certified, endorsed or approved by any "
    "body - you may only say it is aligned with a framework. Return JSON only."
)
QUESTION_RULES = (
    "For question-bank tables write real content questions about the topics (not about the course structure): "
    "multiple-choice items have one correct answer and three plausible distractors, options written as "
    "'A. ...  B. ...  C. ...  D. ...', and the answer column gives the letter; short-answer and practical items give "
    "marking points. Keep the same columns."
)


def enabled() -> bool:
    return config.USE_LLM


def _chat(messages: list[dict], temperature: float = 0.3) -> Optional[dict]:
    if not config.USE_LLM:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=temperature,
                                              response_format={"type": "json_object"})
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        log.warning("Document LLM call failed, keeping the rule-based content: %s", exc)
        return None


def merge_sections(original: list[DocSection], returned) -> tuple[list[DocSection], list[str]]:
    """Accept returned sections only where they replace an unlocked section with the same key and keep its table
    columns. Returns (merged sections, keys changed)."""
    if not isinstance(returned, list):
        return original, []
    by_key = {s.key: s for s in original}
    accepted: dict[str, DocSection] = {}
    for item in returned:
        if not isinstance(item, dict) or item.get("key") not in by_key:
            continue
        old = by_key[item["key"]]
        if old.locked:
            continue
        try:
            new = DocSection(**{**old.model_dump(), **{k: v for k, v in item.items()
                                                      if k in ("heading", "paragraphs", "bullets", "table", "notes")}})
        except (ValidationError, TypeError) as exc:
            log.warning("LLM section %s rejected: %s", item.get("key"), exc)
            continue
        if (old.table is None) != (new.table is None) or (old.table and old.table.columns != new.table.columns):
            log.warning("LLM section %s changed the table shape - rejected", old.key)
            continue
        new.locked, new.level, new.key = old.locked, old.level, old.key
        if new != old:
            accepted[old.key] = new
    return [accepted.get(s.key, s) for s in original], list(accepted)


def _payload(sections: list[DocSection]) -> list[dict]:
    return [s.model_dump(exclude={"locked", "level"}, exclude_none=True) for s in sections if not s.locked]


def refine(doc_type: str, title: str, sections: list[DocSection], facts: dict, context: str, language: str,
           wishes: Optional[str] = None) -> Optional[list]:
    """Improve the wording of the rule-based sections; returns the list of rewritten sections (unvalidated)."""
    lang = "Arabic" if language == "ar" else "English"
    extra = QUESTION_RULES if doc_type == "question_bank" else ""
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"DOCUMENT TYPE: {doc_type}\nTITLE: {title}\n\nFACTS:\n{json.dumps(facts, ensure_ascii=False, default=str)}"
            f"\n\nSECTIONS:\n{json.dumps(_payload(sections), ensure_ascii=False)}\n\n"
            f"CONTEXT:\n{context or '(no reference passages)'}\n\n"
            f"Rewrite the sections into polished, specific, professional {lang} following the reference examples. "
            f"Keep every key, keep the number of sections, keep table columns. {extra} "
            + (f"The requester also asked: {wishes}. " if wishes else "")
            + "Return JSON {\"sections\": [{key, heading, paragraphs, bullets, table, notes}]}."}])
    return data.get("sections") if isinstance(data, dict) else None


def revise(doc_type: str, sections: list[DocSection], instructions: str, language: str) -> Optional[list]:
    """Apply a reviewer's plain-language change request to the unlocked sections."""
    lang = "Arabic" if language == "ar" else "English"
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"DOCUMENT TYPE: {doc_type}\n\nSECTIONS:\n{json.dumps(_payload(sections), ensure_ascii=False)}\n\n"
            f"REVIEWER REQUEST:\n{instructions}\n\nApply the reviewer's request and change nothing else. Write in "
            f"{lang}. Return JSON {{\"sections\": [only the sections you changed, with their key]}}."}], temperature=0.2)
    return data.get("sections") if isinstance(data, dict) else None
