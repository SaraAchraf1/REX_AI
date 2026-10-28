"""Turn a raw tender notice into validated fields: country, category, relevance, dates.

Owner: Member 2. Pure functions (no I/O) so each rule has its own unit test in tests/test_rfp_agent.py.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.agents.rfp import taxonomy as tx
from rex_ai.core.schemas import Country, RFPCategory


@dataclass
class RawOpportunity:
    """One notice as a source delivered it (feed row, tender document, web listing). Values are unvalidated."""
    title: str
    source_id: str
    organization: Optional[str] = None
    country: Optional[str] = None
    deadline: Any = None
    publication_date: Any = None
    eligibility: Optional[str] = None
    category: Optional[str] = None
    submission_link: Optional[str] = None
    documents: list[str] = field(default_factory=list)
    description: Optional[str] = None
    source_url: Optional[str] = None
    origin: str = "feed"                       # feed | document | web
    raw_ref: Optional[str] = None              # file name / row number, for the audit trail

    @property
    def text(self) -> str:
        """Everything the classifiers read."""
        return "\n".join(x for x in (self.title, self.organization, self.category, self.description,
                                     self.eligibility) if x)

    @property
    def topic(self) -> str:
        """What is being bought (no eligibility text: 'bilingual trainers' is not a training topic)."""
        return "\n".join(x for x in (self.title, self.description) if x)


# ======================================================================= country
def _place_hits(text: str, places: dict) -> dict:
    return {key: n for key, words in places.items() if (n := sum(contains_any(text, [w]) for w in words))}


def detect_country(raw: RawOpportunity) -> tuple[Optional[Country], str]:
    """(EG | SA | None, explanation). An explicit country field always wins; a field naming another country
    is rejected even if the text mentions Egypt. Without a field, the text must point to EG or SA."""
    if raw.country and str(raw.country).strip():
        for country, aliases in tx.COUNTRY_ALIASES.items():
            if contains_any(str(raw.country), aliases):
                return country, f"country field: {raw.country}"
        return None, f"outside Egypt/Saudi Arabia (country field: {raw.country})"

    hits = _place_hits(raw.text, tx.COUNTRY_PLACES)
    if hits:
        country = max(hits, key=hits.get)
        return country, f"inferred from text ({country.value})"
    others = _place_hits(raw.text, tx.OTHER_PLACES)
    if others:
        return None, f"outside Egypt/Saudi Arabia (text mentions {', '.join(others)})"
    return None, "country unknown - excluded (strict Egypt/Saudi Arabia rule)"


# ======================================================================= relevance
@dataclass
class Classification:
    relevant: bool
    category: Optional[RFPCategory]
    score: float
    reason: str
    terms: list[str] = field(default_factory=list)


def matched_terms(text: str, words: list[str]) -> list[str]:
    return [w for w in words if contains_any(text, [w])]


def classify(raw: RawOpportunity, extra_keywords: Optional[list[str]] = None) -> Classification:
    # what is bought, not who buys it: "Gulf Skills Development Fund" also tenders catering
    text = normalize("\n".join(x for x in (raw.title, raw.category, raw.description, raw.eligibility) if x))
    for place in tx.TRAINING_PLACES:              # "catering for the training center" is not a training tender
        text = text.replace(normalize(place), " ")
    core = matched_terms(text, tx.TRAINING_CORE + list(extra_keywords or []))
    negative = matched_terms(text, tx.NOT_TRAINING)
    if not core:
        return Classification(False, None, 0.0, "no training or education terms")
    if negative and len(negative) >= len(core):
        return Classification(False, None, 0.1, f"supply/works tender, training is incidental ({', '.join(negative)})")

    per_cat = {c: matched_terms(text, words) for c, words in tx.CATEGORY_KEYWORDS.items()}
    hint = tx.CATEGORY_ALIASES.get(normalize(raw.category or "")) or tx.CATEGORY_ALIASES.get(raw.category or "")
    if hint:
        category = hint
    else:
        # the generic "Education and Training" bucket only wins when nothing more specific matched
        specific = {c: len(t) for c, t in per_cat.items() if t and c != RFPCategory.education_training}
        category = max(specific, key=specific.get) if specific else RFPCategory.education_training
    terms = list(dict.fromkeys(core + [t for ts in per_cat.values() for t in ts]))
    score = min(1.0, round((len(core) + sum(map(len, per_cat.values()))) / 6, 2))
    return Classification(True, category, score, "training-related", terms)


# ======================================================================= dates
_MONTHS = {
    **{m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                   "dec"], start=1)},
    **{m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"], start=1)},
    "sept": 9,
    **{normalize(m): i for m, i in [("يناير", 1), ("فبراير", 2), ("مارس", 3), ("أبريل", 4), ("إبريل", 4),
                                    ("مايو", 5), ("يونيو", 6), ("يونيه", 6), ("يوليو", 7), ("يوليه", 7),
                                    ("أغسطس", 8), ("سبتمبر", 9), ("أكتوبر", 10), ("نوفمبر", 11), ("ديسمبر", 12)]},
}
_HIJRI_MARK = re.compile(r"هـ|\bAH\b|hijri|هجري", re.I)


def hijri_to_gregorian(y: int, m: int, d: int) -> date:
    """Tabular Islamic calendar -> Gregorian. Saudi portals use Umm al-Qura, which can differ by 1-2 days,
    so callers note that the converted deadline is approximate."""
    days = d + math.ceil(29.5 * (m - 1)) + (y - 1) * 354 + (3 + 11 * y) // 30 - 1
    return date(622, 7, 19) + timedelta(days=days)


def parse_date(value: Any) -> tuple[Optional[date], bool]:
    """(date or None, approximate). Accepts ISO, day-first numeric (15/11/2026), month names in English or
    Arabic, Arabic digits and Hijri dates (14xx years or an 'هـ' marker)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, False
    if isinstance(value, datetime):
        return value.date(), False
    if isinstance(value, date):
        return value, False
    raw = str(value).strip()
    if not raw:
        return None, False
    s = normalize(raw)
    y = mo = d = None
    if m := re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):
        y, mo, d = map(int, m.groups())
    elif m := re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s):
        d, mo, y = map(int, m.groups())
    elif m := re.search(r"(\d{1,2})\s+([^\d\s]+)\s+(\d{4})", s):            # 15 November 2026 / 15 نوفمبر 2026
        d, mo, y = int(m.group(1)), _MONTHS.get(m.group(2)), int(m.group(3))
    elif m := re.search(r"([a-z]+)\s+(\d{1,2})\s+(\d{4})", s):              # November 15 2026
        mo, d, y = _MONTHS.get(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (y and mo and d):
        return None, False
    try:
        if 1300 <= y <= 1500 or _HIJRI_MARK.search(raw):
            return hijri_to_gregorian(y, mo, d), True
        return date(y, mo, d), False
    except ValueError:
        return None, False


# ======================================================================= tender documents
# "Label: value" lines in RFP / terms-of-reference documents (English and Arabic).
FIELD_LABELS = {
    "title": ["tender title", "title", "rfp title", "subject", "project title", "عنوان المناقصة", "اسم المنافسة",
              "عنوان المنافسة", "موضوع المناقصة", "الموضوع"],
    "organization": ["issuing organization", "issuing organisation", "issued by", "client", "contracting authority",
                     "buyer", "entity", "الجهة", "الجهة المالكة", "جهة الطرح", "الجهة الحكومية", "الجهة المعلنة"],
    "country": ["country", "location", "place of delivery", "الدولة", "مكان التنفيذ", "مكان التدريب"],
    "deadline": ["deadline", "submission deadline", "closing date", "bid submission deadline", "due date",
                 "آخر موعد", "آخر موعد لتقديم العروض", "آخر موعد لاستلام العروض", "تاريخ الإغلاق", "موعد التقديم"],
    "publication_date": ["publication date", "published", "issue date", "date of issue", "تاريخ النشر",
                         "تاريخ الطرح", "تاريخ الإعلان"],
    "eligibility": ["eligibility", "eligibility requirements", "qualification requirements", "bidder requirements",
                    "شروط التأهيل", "الشروط", "متطلبات التأهيل", "شروط المشاركة"],
    "category": ["category", "training category", "التصنيف", "نوع النشاط"],
    "submission_link": ["submission link", "submit at", "submission portal", "رابط التقديم", "بوابة التقديم"],
    "documents": ["documents", "available documents", "attachments", "المستندات", "الوثائق", "المرفقات"],
    "description": ["scope", "scope of work", "description", "summary", "نطاق العمل", "الوصف", "وصف المناقصة"],
}
_LABEL_INDEX = {normalize(lbl): key for key, labels in FIELD_LABELS.items() for lbl in labels}
_MULTILINE = {"eligibility", "documents", "description"}
_URL = re.compile(r"https?://[^\s)>\]]+")
_TITLE_PREFIX = re.compile(r"^(request for (proposals?|quotations?)|invitation to tender|rfp|rfq|tender|"
                           r"terms of reference|كراسة الشروط والمواصفات)\s*[:\-–]\s*", re.I)


def extract_fields(text: str) -> dict:
    """Read 'Label: value' lines (and bullet lists under a label) from a tender document."""
    out: dict[str, Any] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip().strip("*").strip()
        if not stripped:
            current = None if current not in _MULTILINE else current
            continue
        label, sep, value = stripped.partition(":")
        key = _LABEL_INDEX.get(normalize(label.strip("#*- ").strip())) if sep else None
        if key:
            value = value.strip().strip("*").strip()
            current = key
            if key == "documents":
                out[key] = [v.strip() for v in re.split(r"[;,،؛]", value) if v.strip()]
            elif value:
                out[key] = value
            continue
        if stripped.startswith("#"):
            current = None
            out.setdefault("_heading", stripped.lstrip("#").strip())
            continue
        if current in _MULTILINE and re.match(r"^([-*•·]|\d+[.)])\s+", stripped):
            item = re.sub(r"^([-*•·]|\d+[.)])\s+", "", stripped)
            if current == "documents":
                out.setdefault("documents", []).append(item)
            else:
                out[current] = (out.get(current, "") + ("\n" if out.get(current) else "") + "- " + item)
        elif current == "description":
            out["description"] = (out.get("description", "") + " " + stripped).strip()
    out.setdefault("title", out.pop("_heading", None))
    out.pop("_heading", None)
    if out.get("title"):
        out["title"] = _TITLE_PREFIX.sub("", out["title"]).strip()
    if "submission_link" in out and (m := _URL.search(out["submission_link"])):
        out["submission_link"] = m.group(0)
    elif "submission_link" not in out and (m := _URL.search(text)):
        out["submission_link"] = m.group(0)
    return {k: v for k, v in out.items() if v}
