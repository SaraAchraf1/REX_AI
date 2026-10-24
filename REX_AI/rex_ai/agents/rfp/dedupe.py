"""Duplicate RFP detection (owner: Member 2).

Two levels:
  exact  dedupe_hash = sha256(normalized title | organization | deadline) - the UNIQUE key in the rfps table.
  fuzzy  the same tender re-published on another portal with a slightly different title: same country,
         similar title (token-sorted similarity) and either a similar organisation or the same deadline.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Optional

from rex_ai.agents.rex.text_utils import normalize

TITLE_SIMILARITY = 0.80
ORG_SIMILARITY = 0.70
_STOP = {"the", "of", "for", "and", "a", "an", "to", "in", "on", "rfp", "rfq", "tender", "notice",
         "مناقصة", "مناقصه", "منافسة", "منافسه", "عامة", "عامه", "لتقديم", "تقديم", "في", "من", "على", "ل"}


def _key(text: Optional[str]) -> str:
    """Normalized, stop-word free, order-independent form of a title or organisation name."""
    tokens = re.findall(r"\w+", normalize(text or ""))
    return " ".join(sorted(t for t in tokens if t not in _STOP))


def dedupe_hash(title: str, organization: str, deadline: Optional[date]) -> str:
    raw = f"{_key(title)}|{_key(organization)}|{deadline.isoformat() if deadline else ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def similarity(a: Optional[str], b: Optional[str]) -> float:
    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return 0.0
    return SequenceMatcher(None, ka, kb).ratio()


def is_duplicate(a: dict, b: dict) -> bool:
    """a, b: {"title", "issuing_organization", "country", "deadline"} (country as 'EG'/'SA')."""
    if a["country"] != b["country"]:
        return False
    if similarity(a["title"], b["title"]) < TITLE_SIMILARITY:
        return False
    same_deadline = a.get("deadline") and a.get("deadline") == b.get("deadline")
    return bool(same_deadline or similarity(a["issuing_organization"], b["issuing_organization"]) >= ORG_SIMILARITY)
