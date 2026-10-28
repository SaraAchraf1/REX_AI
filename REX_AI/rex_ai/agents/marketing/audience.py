"""Understand the campaign's target audience (brief, Agent 5 item 37).

The audience text (the request's target_audience, else the course's) is matched to the personas in
data/marketing/audiences.yaml by shared keywords; what the consented leads told REX about themselves (skill level,
education, goals) adds weight. The result drives the angle of the copy, the tone, the channel priority and the
targeting suggestions in the strategy.
"""
from __future__ import annotations

import re
from typing import Optional

from rex_ai.agents.marketing import rules
from rex_ai.agents.rex.text_utils import contains_any, normalize

# Lead profile values that point to a persona (REX's LeadProfile fields)
LEAD_HINTS = {"students": ["graduate", "student", "Beginner"], "professionals": ["Intermediate", "Advanced"],
              "career_switchers": ["switch", "change", "career"]}


def _has(text: str, kw: str) -> bool:
    """English keywords match whole words (plural allowed: 'engineer' is not found in 'engineering'); Arabic ones
    anywhere, since Arabic attaches prefixes such as ال and لل to the word."""
    k = normalize(kw)
    if k.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(k)}s?(?![a-z0-9])", text) is not None
    return contains_any(text, [kw])


def _score(persona: dict, text: str) -> int:
    """Distinct keywords found; 'manager' and 'managers' in the same text count once."""
    t = normalize(text)
    found = [kw.lower() for kw in persona["keywords"] if _has(t, kw)]
    return sum(1 for kw in found if not any(o != kw and o.startswith(kw) for o in found))


def resolve(audience_text: Optional[str], course_audience: str, lead_insights: dict, language: str) -> dict:
    text = (audience_text or course_audience or "").strip()
    lead_text = " ".join(str(v) for key in ("skill_levels", "education", "goals", "backgrounds")
                         for v in lead_insights.get(key) or [])
    scored = []
    for n, p in enumerate(rules.personas()):
        score = 2 * _score(p, text) + sum(1 for h in LEAD_HINTS.get(p["persona_id"], []) if h in lead_text)
        scored.append((score, -n, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0][2] if scored[0][0] > 0 else rules.default_persona()
    secondary = next((p for s, _, p in scored if s > 0 and p is not best), None)
    lang = "ar" if language == "ar" else "en"
    return {
        "text": text,
        "from": "request" if audience_text else "course catalogue",
        "persona_id": best["persona_id"],
        "persona": best["name"][lang],
        "persona_en": best["name"]["en"],
        "secondary_persona": secondary["name"][lang] if secondary else None,
        "matched": scored[0][0] > 0,
        "pains": best["pains"][lang],
        "motivations": best["motivations"][lang],
        "tone": best["tone"][lang],
        "tone_en": best["tone"]["en"],
        "preferred_channels": best["channels"],
        "targeting": best["targeting"],
        "lead_insights": lead_insights,
    }
