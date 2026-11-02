"""Optional LLM calls for the Document Agent. Each function returns None when no key is set or the call fails, and
the agent keeps the rule-based sections - the workflow never depends on the LLM.

refine() (generation) uses OpenAI: only UNLOCKED sections are sent and accepted back (locked ones hold fees, hours,
weights and schedules from approved data) and every returned section must keep its key and table shape
(merge_sections). revise() (a reviewer's change request) uses Gemini and may change anything the reviewer asks for -
any section including locked ones, table columns, new sections, removed sections, the title (merge_revision). The
agent runs the accreditation-claim guard on the result either way.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.core import gemini, llm_status
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


def edit_enabled() -> bool:
    """Change requests are applied by Gemini."""
    return gemini.enabled()


def _chat(messages: list[dict], temperature: float = 0.3) -> Optional[dict]:
    llm_status.clear()
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
        llm_status.record(exc)
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


def merge_revision(original: list[DocSection], answer) -> tuple[list[DocSection], list[str], Optional[str]]:
    """Apply a change request's answer {sections, remove, title}: a returned key replaces that section (locked ones
    too - the reviewer asked), a new key is added after the section named in "after" (or at the end), keys in
    "remove" are dropped. Returns (sections, what changed, new title or None)."""
    if not isinstance(answer, dict):
        return original, [], None
    sections, changed = list(original), []
    remove = answer.get("remove") if isinstance(answer.get("remove"), list) else []
    if remove:
        keep = [s for s in sections if s.key not in remove]
        changed += [f"removed {s.key}" for s in sections if s.key in remove]
        sections = keep
    for item in answer.get("sections") if isinstance(answer.get("sections"), list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str) or not item["key"].strip():
            continue
        idx = next((i for i, s in enumerate(sections) if s.key == item["key"]), None)
        fields = {k: v for k, v in item.items() if k in ("heading", "paragraphs", "bullets", "numbered", "table",
                                                        "notes", "level")}
        try:
            if idx is None:
                if not fields.get("heading"):
                    continue
                new = DocSection(key=item["key"], **fields)
                after = next((i for i, s in enumerate(sections) if s.key == item.get("after")), None)
                sections.insert(len(sections) if after is None else after + 1, new)
                changed.append(f"added {new.key}")
            else:
                old = sections[idx]
                new = DocSection(**{**old.model_dump(), **fields})
                new.locked, new.key = old.locked, old.key
                if new != old:
                    sections[idx] = new
                    changed.append(old.key)
        except (ValidationError, TypeError) as exc:
            log.warning("LLM section %s rejected: %s", item.get("key"), exc)
    title = answer.get("title") if isinstance(answer.get("title"), str) and answer["title"].strip() else None
    if title:
        changed.append("title")
    return sections, changed, title


def revise(doc_type: str, title: str, sections: list[DocSection], instructions: str, language: str
           ) -> Optional[dict]:
    """Apply a reviewer's plain-language change request (Gemini); any part of the document may change."""
    lang = "Arabic" if language == "ar" else "English"
    payload = [s.model_dump(exclude_none=True) for s in sections]
    data = gemini.chat_json([
        {"role": "system", "content": SYSTEM.replace(
            "do not invent fees, dates, hours,", "do not invent fees, dates, hours (unless the reviewer gives them),")},
        {"role": "user", "content":
            f"DOCUMENT TYPE: {doc_type}\nTITLE: {title}\n\nSECTIONS:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"REVIEWER REQUEST:\n{instructions}\n\nApply the reviewer's request and change nothing else. Anything "
            "may change if the reviewer asks: wording, headings, bullets, tables (rows and columns - every row needs one "
            "cell per column), notes, locked sections (fees, hours, weights, schedules), the title; sections may be "
            "added or removed. "
            + (QUESTION_RULES + " " if doc_type == "question_bank" else "")
            + f"Write in {lang}. Return JSON {{\"sections\": [only the sections you changed or added, each with its "
            "full content {key, heading, paragraphs, bullets, table: {columns, rows} or null, notes}; a new section "
            "gets a new key and \"after\": the key it follows], \"remove\": [keys of sections to delete], "
            "\"title\": new title or null}."}], temperature=0.2)
    return data if isinstance(data, dict) else None
