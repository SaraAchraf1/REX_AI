"""Loads the academy's approved data and turns it into searchable knowledge-base chunks.

Owner: Member 1. Data files live in data/*.json (see data/data_sources.yaml).
Every chunk carries the metadata fields from the brief (department, level, audience,
duration, country, source, update date) so retrieval can be filtered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rex_ai import config
from rex_ai.core.schemas import FAQ, Course, Instructor, Promotion, Schedule, SourceRef

SOURCE_TITLES = {
    "SRC-ACAD-COURSES": "Academy course catalogue",
    "SRC-ACAD-INSTRUCTORS": "Instructor profiles",
    "SRC-ACAD-PRICING": "Course fees and promotions",
    "SRC-ACAD-SCHEDULES": "Cohort schedules",
    "SRC-ACAD-FAQ": "Academy FAQs",
    "SRC-ACAD-ADMISSION": "Admission requirements",
    "SRC-ACAD-DOCS": "Academy documents (data/catalog, data/documents)",
}


def source(source_id: str, record_id: Optional[str] = None) -> SourceRef:
    return SourceRef(source_id=source_id, title=SOURCE_TITLES.get(source_id, source_id), record_id=record_id)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    courses: dict[str, Course]
    instructors: dict[str, Instructor]
    schedules: list[Schedule]
    promotions: list[Promotion]
    faqs: list[FAQ]
    admission: dict
    meta: dict
    chunks: list[Chunk] = field(default_factory=list)

    # ---- lookups used by the assistant -------------------------------------
    def schedules_for(self, course_id: str, today: Optional[date] = None) -> list[Schedule]:
        today = today or date.today()
        return sorted((s for s in self.schedules if s.course_id == course_id and s.start_date >= today),
                      key=lambda s: s.start_date)

    def instructors_for(self, course_id: str) -> list[Instructor]:
        return [self.instructors[i] for i in self.courses[course_id].instructor_ids if i in self.instructors]

    def promotions_for(self, course_id: str, today: Optional[date] = None) -> list[Promotion]:
        today = today or date.today()
        return [p for p in self.promotions
                if p.valid_until >= today and (p.applies_to == "all" or course_id in p.applies_to)]

    def best_price(self, course_id: str) -> tuple[Optional[float], Optional[Promotion]]:
        course = self.courses[course_id]
        promos = self.promotions_for(course_id)
        if course.fee is None or not promos:
            return course.fee, None
        best = max(promos, key=lambda p: p.discount_pct)  # one promotion per registration
        return round(course.fee * (1 - best.discount_pct / 100)), best


def _read(data_dir: Path, name: str) -> dict:
    return json.loads((data_dir / name).read_text(encoding="utf-8"))


def load_knowledge_base(data_dir: Optional[Path] = None) -> KnowledgeBase:
    data_dir = data_dir or config.DATA_DIR
    raw_courses = _read(data_dir, "courses.json")
    pricing = _read(data_dir, "pricing_promotions.json")
    prices = {p["course_id"]: p for p in pricing["prices"]}
    meta = raw_courses["_meta"]

    courses: dict[str, Course] = {}
    for c in raw_courses["courses"]:
        p = prices.get(c["course_id"], {})
        courses[c["course_id"]] = Course(
            **c, fee=p.get("fee"), installments=p.get("installments"),
            currency=pricing["_meta"].get("currency", "EGP"),
            last_updated=meta.get("last_updated"), is_sample=meta.get("status") == "sample",
        )

    kb = KnowledgeBase(
        courses=courses,
        instructors={i["instructor_id"]: Instructor(**i) for i in _read(data_dir, "instructors.json")["instructors"]},
        schedules=[Schedule(**s) for s in _read(data_dir, "schedules.json")["schedules"]],
        promotions=[Promotion(**p) for p in pricing["promotions"]],
        faqs=[FAQ(**f) for f in _read(data_dir, "faqs.json")["faqs"]],
        admission=_read(data_dir, "admission_requirements.json"),
        meta=meta,
    )
    kb.chunks = build_chunks(kb)
    return kb


def build_chunks(kb: KnowledgeBase) -> list[Chunk]:
    """One overview chunk + one details chunk per course, one chunk per FAQ (bilingual)."""
    chunks: list[Chunk] = []
    for c in kb.courses.values():
        base_meta = {
            "type": "course", "course_id": c.course_id, "department": c.department,
            "skill_level": c.skill_level.value, "target_audience": c.target_audience,
            "industry": c.industry, "duration_hours": c.duration_hours,
            "country": [x.value for x in c.country], "source_id": c.source_id,
            "last_updated": str(c.last_updated) if c.last_updated else None,
            "accreditation_alignment": c.accreditation_alignment,
        }
        chunks.append(Chunk(
            f"{c.course_id}#overview",
            f"{c.title_en} | {c.title_ar}. {c.description_en} {c.description_ar} "
            f"Department: {c.department}. Level: {c.skill_level.value}. Audience: {c.target_audience}. "
            f"Tags: {', '.join(c.tags)}. Careers: {', '.join(c.career_opportunities)}.",
            {**base_meta, "section": "overview"},
        ))
        chunks.append(Chunk(
            f"{c.course_id}#details",
            f"{c.title_en} | {c.title_ar}. Learning outcomes: {'; '.join(c.learning_outcomes)}. "
            f"Modules: {', '.join(c.modules)}. Prerequisites: {', '.join(c.prerequisites)}. "
            f"Duration: {c.duration_hours} hours over {c.duration_weeks} weeks.",
            {**base_meta, "section": "details"},
        ))
    for f in kb.faqs:
        chunks.append(Chunk(
            f"{f.faq_id}",
            f"{f.q_en} {f.q_ar} {' '.join(f.keywords)} {' '.join(f.keywords)} {f.a_en} {f.a_ar}",
            {"type": "faq", "faq_id": f.faq_id, "category": f.category, "source_id": "SRC-ACAD-FAQ"},
        ))
    return chunks


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    """Cached; rex_ai.core.data_sync clears it when new data is imported."""
    return load_knowledge_base()
