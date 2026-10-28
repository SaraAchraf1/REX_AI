"""Optional LLM calls for the Marketing Agent. Each function returns None when no key is set or the call fails, and
the agent keeps the rule-based copy - the workflow never depends on the LLM.

Safety: the model gets the approved FACTS, the audience persona and the brand rules / channel playbook / examples
from the RAG (marketing_refs). Whatever it returns goes through checks.check_variant: a variant with a number that
is not in the facts is rejected, claims are rewritten and channel limits enforced.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.marketing import checks, rules
from rex_ai.core.schemas import ContentVariant

log = logging.getLogger("rex.marketing.llm")

SYSTEM = (
    "You are the marketing copywriter of a training academy in Egypt and Saudi Arabia. Use ONLY the facts in FACTS: "
    "never invent fees, discounts, dates, deadlines, hours, seats, instructors, statistics or testimonials, and do not "
    "add numbers that are not in FACTS. Follow the brand rules and channel playbook in CONTEXT. NEVER say or imply "
    "that a course is accredited, certified or endorsed by any body, and never promise a job, salary or pass rate. "
    "Keep each channel's limits. Return JSON only."
)


def enabled() -> bool:
    return config.USE_LLM


def _chat(messages: list[dict], temperature: float = 0.6) -> Optional[dict]:
    if not config.USE_LLM:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=temperature,
                                              response_format={"type": "json_object"})
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        log.warning("Marketing LLM call failed, keeping the rule-based copy: %s", exc)
        return None


def _payload(variants: list[ContentVariant]) -> list[dict]:
    return [{"channel": v.channel.value, "variant_no": v.variant_no, "angle": v.angle, "headline": v.headline,
             "body": v.body, "cta": v.cta} for v in variants]


def _limits(variants: list[ContentVariant]) -> str:
    out = []
    for ch in dict.fromkeys(v.channel.value for v in variants):
        c = rules.channel(ch)
        out.append(f"{ch}: headline <= {c['headline_max']} chars, body <= {c['body_max']} chars (aim for "
                   f"{c['body_target']}), {c['hashtags']} hashtags" + (f", CTA one of {c['ctas']}" if c.get("ctas")
                                                                      else ""))
    return "\n".join(out)


def merge(original: list[ContentVariant], returned, allowed: set[str], lang: str
          ) -> tuple[list[ContentVariant], list[str], list[str]]:
    """Accept a returned variant only if it matches an existing channel + variant_no and passes the fact check.
    Returns (variants, keys changed, rejection notes)."""
    if not isinstance(returned, list):
        return original, [], []
    by_key = {v.key: v for v in original}
    accepted: dict[str, ContentVariant] = {}
    rejected: list[str] = []
    for item in returned:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('channel')}:{item.get('variant_no')}"
        old = by_key.get(key)
        if old is None:
            continue
        try:
            new = ContentVariant(**{**old.model_dump(), **{k: item[k] for k in ("headline", "body", "cta")
                                                           if isinstance(item.get(k), str) and item[k].strip()}})
        except ValidationError as exc:
            log.warning("LLM variant %s rejected: %s", key, exc)
            continue
        new.checks, new.generated_by = [], "llm"
        new, invented = checks.check_variant(new, allowed, lang)
        if invented:
            rejected.append(f"{key}: LLM version rejected, it used numbers not in the approved data "
                            f"({', '.join(invented)})")
            continue
        if (new.headline, new.body, new.cta) != (old.headline, old.body, old.cta):
            new.hashtags = [w for w in new.body.split() if w.startswith("#")] or old.hashtags
            accepted[key] = new
    return [accepted.get(v.key, v) for v in original], list(accepted), rejected


def refine(variants: list[ContentVariant], facts: dict, audience: dict, strategy: str, context: str, lang: str,
           wishes: Optional[str] = None) -> Optional[list]:
    """Rewrite the rule-based variants into persuasive, natural copy; returns the list of variants (unvalidated)."""
    language = "Arabic (Modern Standard, natural for Egypt and Saudi Arabia)" if lang == "ar" else "English"
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"FACTS:\n{json.dumps(facts, ensure_ascii=False, default=str)}\n\n"
            f"AUDIENCE:\n{json.dumps(audience, ensure_ascii=False, default=str)}\n\nSTRATEGY:\n{strategy}\n\n"
            f"CONTEXT (brand rules, channel playbook, approved examples):\n{context or '(none)'}\n\n"
            f"CHANNEL LIMITS:\n{_limits(variants)}\n\nDRAFT VARIANTS:\n"
            f"{json.dumps(_payload(variants), ensure_ascii=False)}\n\n"
            f"Rewrite every draft variant in {language} for this audience and tone. Keep each variant's angle so the "
            "variations stay different (they are A/B tested), keep the CTA meaning, keep hashtags at the end of social "
            "and LinkedIn posts, keep the opt-out line in e-mail and WhatsApp. "
            + (f"The requester also asked: {wishes}. " if wishes else "")
            + "Return JSON {\"variants\": [{channel, variant_no, headline, body, cta}]}."}])
    return data.get("variants") if isinstance(data, dict) else None


def revise(variants: list[ContentVariant], facts: dict, instructions: str, lang: str) -> Optional[list]:
    """Apply a reviewer's plain-language change request; returns only the variants the model changed."""
    language = "Arabic" if lang == "ar" else "English"
    data = _chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"FACTS:\n{json.dumps(facts, ensure_ascii=False, default=str)}\n\nCHANNEL LIMITS:\n{_limits(variants)}\n\n"
            f"VARIANTS:\n{json.dumps(_payload(variants), ensure_ascii=False)}\n\nREVIEWER REQUEST:\n{instructions}\n\n"
            f"Apply the reviewer's request and change nothing else. Write in {language}. Return JSON "
            "{\"variants\": [only the variants you changed, with channel and variant_no]}."}], temperature=0.3)
    return data.get("variants") if isinstance(data, dict) else None
