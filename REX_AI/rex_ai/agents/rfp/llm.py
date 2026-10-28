"""Optional LLM calls for the RFP agent. Every function returns None / {} when no key is set or the call
fails, and the agent falls back to its rule-based output - the pipeline never depends on the LLM."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rex_ai import config

log = logging.getLogger("rex.rfp.llm")

SYSTEM = (
    "You are the tender research analyst of a training academy in Egypt. Use ONLY the numbered context "
    "passages and the record fields you are given. Never invent dates, budgets, organisations or requirements; "
    "if something is not in the context, say it is not stated. Cite passages as [n]. Never state that the "
    "academy is accredited or certainly qualified."
)

FIELD_QUERIES = {
    "organization": "issuing organization client entity الجهة",
    "country": "country location place of delivery الدولة مكان التنفيذ",
    "deadline": "submission deadline closing date آخر موعد لتقديم العروض",
    "publication_date": "publication date issue date تاريخ النشر",
    "eligibility": "eligibility qualification requirements for bidders شروط التأهيل",
    "description": "scope of work training required نطاق العمل",
}


def enabled() -> bool:
    return config.USE_LLM


def _chat(messages: list[dict], json_mode: bool = False) -> Optional[str]:
    if not config.USE_LLM:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=0.1,
                                              **kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("RFP LLM call failed, using rule-based output: %s", exc)
        return None


def extract_tender_fields(rag, path: Path, found: dict) -> dict:
    """Fill the fields the label rules missed, from passages of this document retrieved by RAG."""
    missing = [f for f in FIELD_QUERIES if not found.get(f)]
    if not missing:
        return {}
    hits, seen = [], set()
    for f in missing:
        for h in rag.search(FIELD_QUERIES[f], k=2, where={"source_path": str(path)}):
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                hits.append(h)
    if not hits:
        return {}
    out = _chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"CONTEXT:\n{rag.build_context(hits)}\n\nReturn a JSON object with "
                  f"these keys (null when not stated): {', '.join(missing)}. Dates as YYYY-MM-DD, or exactly as "
                  "written if Hijri. country = the country where the training is delivered."}], json_mode=True)
    try:
        data = json.loads(out) if out else {}
    except json.JSONDecodeError:
        return {}
    return {k: str(v) for k, v in data.items() if k in missing and v}


def summarize(record: dict, context: str) -> Optional[str]:
    return _chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"RECORD:\n{json.dumps(record, ensure_ascii=False, default=str)}\n\n"
                   f"CONTEXT:\n{context or '(no tender document indexed)'}\n\nWrite a 2-3 sentence summary for the "
                   "academy's bid team: what is being bought, for whom, key requirement, deadline. English."}])


def answer(question: str, context: str) -> Optional[str]:
    return _chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer briefly, "
                   "in the language of the question, citing [n]."}])
