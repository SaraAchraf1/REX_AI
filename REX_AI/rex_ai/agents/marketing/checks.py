"""Checks on every piece of campaign copy before a reviewer sees it (brief, Agent 5: accurate course information,
§2: outputs traceable to source data, Agent 3 rule: never claim accreditation).

  facts     every number in the copy (fees, discounts, dates, hours, seats, years) must appear in the approved facts
            snapshot or in what the requester typed (offer, CTA); otherwise it was invented
  claims    accreditation claims (the Curriculum Agent's guard), job / pass guarantees and "best academy"
            superlatives are rewritten; negated mentions ("we do not guarantee a job") are kept
  channel   headline and body limits per channel (channels.yaml); e-mail and WhatsApp keep their opt-out line

Rule-based copy passes by construction; LLM copy that fails the fact check is rejected (the rule version is kept);
a reviewer's own edits are accepted but flagged.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from rex_ai.agents.curriculum.generator import _NEGATION, guard_text
from rex_ai.agents.marketing import rules
from rex_ai.agents.marketing.writer import T, fit
from rex_ai.core.schemas import ContentVariant

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_NUMBER = re.compile(r"\d+(?:[,.]\d+)*")
# Numbers the templates themselves use (WhatsApp "Reply 1")
TEMPLATE_NUMBERS = {"1"}

_MARKETING_CLAIMS = [
    (re.compile(r"\bguarantee(?:d|s)?\s+(?:you\s+)?(?:a\s+)?(?:new\s+)?(?:job|employment|placement|pass(?:ing)?)\b",
                re.I), "prepares you for the job market"),
    (re.compile(r"\b(?:job|employment|placement|pass)\s+guarantee(?:d)?\b", re.I), "career preparation"),
    (re.compile(r"\b100\s*%\s*(?:job|employment|placement|pass)(?:\s+rate)?\b", re.I), "strong career preparation"),
    (re.compile(r"\b(?:the\s+)?(?:best|#1|number\s+one|top)\s+(?:academy|training\s+(?:center|centre|provider))"
                r"(?:\s+in\s+(?:egypt|saudi(?:\s+arabia)?|the\s+region|the\s+world))?", re.I), "a trusted academy"),
    (re.compile(r"(?:ضمان|نضمن(?:\s+لك)?|مضمونة?)\s+(?:وظيفة|التوظيف|عمل|النجاح|الوظيفة)"), "إعداد عملي لسوق العمل"),
    (re.compile(r"(?:وظيفة|توظيف)\s+مضمون(?:ة)?"), "إعداد عملي لسوق العمل"),
    (re.compile(r"أفضل\s+(?:أكاديمية|مركز\s+تدريب)(?:\s+في\s+\S+)?"), "أكاديمية موثوقة"),
]


# ---------------------------------------------------------------------------- numbers
def _norm(token: str) -> str:
    token = token.replace(",", "")
    try:
        x = float(token)
    except ValueError:
        return token
    return str(int(x)) if x.is_integer() else f"{x:g}"


def _tokens(text: str) -> dict[str, str]:
    """normalised number ("10500") -> as written ("10,500")."""
    return {_norm(m): m for m in _NUMBER.findall(text.translate(_AR_DIGITS))}


def numbers(text: str) -> set[str]:
    return set(_tokens(text))


def allowed_numbers(snapshot: dict[str, Any], extra: Iterable[str] = ()) -> set[str]:
    text = json.dumps(snapshot, ensure_ascii=False, default=str) + " " + " ".join(x for x in extra if x)
    return numbers(text) | TEMPLATE_NUMBERS


def unknown_numbers(v: ContentVariant, allowed: set[str]) -> list[str]:
    """Numbers in the copy that are not in the approved data, as written in the copy."""
    found = _tokens(" ".join(x for x in (v.headline, v.body, v.cta) if x))
    return sorted((found[n] for n in set(found) - allowed), key=lambda x: (len(x), x))


# ---------------------------------------------------------------------------- claims
def guard_marketing(text: str) -> tuple[str, int]:
    text, changes = guard_text(text)
    for pattern, repl in _MARKETING_CLAIMS:
        def sub(m: re.Match) -> str:
            nonlocal changes
            if _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                return m.group(0)
            changes += 1
            return repl
        text = pattern.sub(sub, text)
    return text, changes


# ---------------------------------------------------------------------------- one variant
def _opt_out(v: ContentVariant, lang: str) -> tuple[str, bool]:
    L = T["ar" if lang == "ar" else "en"]
    if v.channel.value == "whatsapp" and "STOP" not in v.body:
        return f"{v.body.rstrip()}\n{L['wa_stop']}", True
    if v.channel.value == "email" and not re.search(r"unsubscribe|إلغاء", v.body, re.I):
        return f"{v.body.rstrip()}\n\n{L['unsubscribe']}", True
    return v.body, False


def check_variant(v: ContentVariant, allowed: set[str], lang: str) -> tuple[ContentVariant, list[str]]:
    """Guard claims, enforce channel limits and opt-out lines, and report invented numbers.
    Returns (checked variant, invented numbers). The variant's checks list says what was changed."""
    v = v.model_copy(deep=True)
    spec = rules.channel(v.channel.value)
    notes = [c for c in v.checks if not c.startswith("numbers not in the approved data")]   # recomputed below
    removed = 0
    for attr in ("headline", "body", "cta"):
        value = getattr(v, attr)
        if value:
            value, n = guard_marketing(value)
            setattr(v, attr, value)
            removed += n
    if removed:
        notes.append(f"claims rewritten: {removed} (accreditation / guarantee / superlative)")
    v.body, added = _opt_out(v, lang)
    if added:
        notes.append("opt-out line added")
    if v.headline and len(v.headline) > spec["headline_max"]:
        v.headline, _ = fit(v.headline, spec["headline_max"])
        notes.append(f"headline shortened to {spec['headline_max']} characters")
    if len(v.body) > spec["body_max"]:
        v.body, _ = fit(v.body, spec["body_max"])
        notes.append(f"body shortened to {spec['body_max']} characters")
    invented = unknown_numbers(v, allowed)
    if invented:
        notes.append("numbers not in the approved data: " + ", ".join(invented))
    v.char_count = len(v.body)
    v.checks = list(dict.fromkeys(notes))
    return v, invented
