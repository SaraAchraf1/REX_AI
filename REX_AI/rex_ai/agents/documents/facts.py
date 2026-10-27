"""Collect the course information a document is built from (brief, Agent 4: "collect or retrieve relevant course
information"). Nothing is asked again that the system already knows (brief §10 implementation note):

  curriculum_id          the curriculum (Agent 3): objectives, outcomes, modules, hours, assessments, capstone,
                         metadata, alignment reports; plus the catalogue course it belongs to, if any
  course_id              the newest APPROVED curriculum of that course, else the catalogue course itself
  catalogue (always)     description, formats, fees and offers, schedules, instructors, careers, certificate
  RAG curriculum_refs    reference-syllabus passages for each module (key points, reference outcomes, citations)

Every source used ends up in DocumentRecord.sources.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from rex_ai.agents.curriculum.agent import NotFound  # one 404 type for every agent (api/main.py)
from rex_ai.agents.curriculum.generator import HOURS_PER_WEEK, split_hours
from rex_ai.agents.curriculum.research import (CATALOGUE_SOURCE_ID, ReferenceSection, _words,
                                               ensure_references_indexed, parse_section, refs_rag)
from rex_ai.agents.documents import templates as tpl
from rex_ai.core import db
from rex_ai.core.schemas import Course, Curriculum, Instructor, Promotion, Schedule, SourceRef

log = logging.getLogger("rex.documents.facts")


@dataclass
class CourseFacts:
    title: str
    language: str
    origin: str                                   # "curriculum" | "catalogue"
    objectives: list[str]
    audience: str
    prerequisites: list[str]
    outcomes: list[str]
    modules: list[dict]                           # {title, topics, hours, exercises}
    hours: float
    exercises: list[str]
    assessments: list[dict]                       # {name, type, weight_pct}
    capstone: str
    department: Optional[str] = None
    level: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    framework: Optional[str] = None
    course_id: Optional[str] = None
    curriculum_id: Optional[str] = None
    curriculum_version: Optional[int] = None
    curriculum_status: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    alignment: list[dict] = field(default_factory=list)      # latest gap report per framework
    course: Optional[Course] = None
    instructors: list[Instructor] = field(default_factory=list)
    schedules: list[Schedule] = field(default_factory=list)
    promotions: list[Promotion] = field(default_factory=list)
    best_price: Optional[float] = None
    sources: list[SourceRef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    module_refs: dict[int, list[ReferenceSection]] = field(default_factory=dict)
    default_assessments: bool = False

    @property
    def weeks(self) -> int:
        return max(1, math.ceil(self.hours / HOURS_PER_WEEK)) if self.hours else 0

    @property
    def teaching_modules(self) -> list[tuple[int, dict]]:
        """(number, module) without the capstone module (always last when the curriculum has one)."""
        mods = list(enumerate(self.modules, start=1))
        if mods and self.capstone and _is_capstone(mods[-1][1]["title"]):
            mods = mods[:-1]
        return mods

    def snapshot(self) -> dict[str, Any]:
        """The facts the document relies on, stored in its metadata (traceability; Marketing reuses them)."""
        c = self.course
        return {
            "course_title": self.title, "course_id": self.course_id, "curriculum_id": self.curriculum_id,
            "curriculum_version": self.curriculum_version, "curriculum_status": self.curriculum_status,
            "department": self.department, "skill_level": self.level, "industry": self.industry,
            "country": self.country, "training_hours": self.hours, "duration_weeks": self.weeks,
            "modules": [m["title"] for m in self.modules], "learning_outcomes": len(self.outcomes),
            "accreditation_framework": self.framework,
            "fee": c.fee if c else None, "currency": c.currency if c else None, "best_price": self.best_price,
            "formats": [f.value for f in c.formats] if c else [],
            "next_start": self.schedules[0].start_date.isoformat() if self.schedules else None,
            "instructors": [i.name for i in self.instructors],
        }


def _is_capstone(title: str) -> bool:
    t = title.lower()
    return "capstone" in t or "ختامي" in title or "final project" in t


def _latest_approved(course_id: str) -> Optional[dict]:
    rows = [r for r in db.list_curricula(status="approved") if (r.get("course_id") or "").upper() == course_id]
    return rows[0] if rows else None


def _alignment(curriculum_id: str) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in db.list_gap_reports(curriculum_id):
        latest.setdefault(r.get("framework_id") or r["framework"], r)
    return list(latest.values())


def collect(kb, curriculum_id: Optional[str], course_id: Optional[str], language: Optional[str]) -> CourseFacts:
    """language None = the curriculum's language (else English)."""
    row = None
    if curriculum_id:
        row = db.get_curriculum(curriculum_id)
        if not row:
            raise NotFound(f"Curriculum {curriculum_id} not found")
    elif course_id:
        if course_id.upper() not in kb.courses:
            raise NotFound(f"Course {course_id} not found in the catalogue")
        row = _latest_approved(course_id.upper())
    if row:
        cur = Curriculum(**row["content"])
        facts = _from_curriculum(cur, kb, language)
        if not curriculum_id:
            facts.notes.append(f"Built from the approved curriculum {cur.curriculum_id} v{cur.version} of "
                               f"{course_id.upper()}")
    else:
        facts = _from_catalogue(kb.courses[course_id.upper()], language or "en")
    if facts.course:
        _add_catalogue_extras(facts, kb)
    return facts


def _from_curriculum(cur: Curriculum, kb, language: Optional[str]) -> CourseFacts:
    m = cur.metadata
    data = cur.model_dump(mode="json")
    lang = language or cur.language.value
    facts = CourseFacts(
        title=cur.course_title, language=lang, origin="curriculum", objectives=list(cur.objectives),
        audience=cur.target_audience, prerequisites=list(cur.prerequisites), outcomes=list(cur.learning_outcomes),
        modules=data["modules"], hours=float(cur.training_hours), exercises=list(cur.practical_exercises),
        assessments=data["assessments"], capstone=cur.capstone_project, department=m.get("department"),
        level=m.get("skill_level"), industry=m.get("industry"), country=m.get("country"),
        framework=m.get("accreditation_framework"), course_id=cur.course_id, curriculum_id=cur.curriculum_id,
        curriculum_version=cur.version, curriculum_status=cur.status.value, approved_by=m.get("approved_by"),
        approved_at=m.get("approved_at"), alignment=_alignment(cur.curriculum_id),
        course=kb.courses.get((cur.course_id or "").upper()), sources=list(cur.sources))
    facts.sources.insert(0, SourceRef(source_id="REX-CURRICULUM", title=f"Curriculum {cur.course_title} "
                                      f"v{cur.version} ({cur.status.value})", record_id=cur.curriculum_id))
    if cur.status.value != "approved":
        facts.notes.append(f"The curriculum is {cur.status.value}, not approved: this document can be reviewed but "
                           "not approved until the curriculum is")
    if lang != cur.language.value:
        facts.notes.append(f"The curriculum is written in {cur.language.value}; its content stays in that language "
                           f"unless the LLM is enabled (it translates while refining)")
    return facts


def _from_catalogue(c: Course, lang: str) -> CourseFacts:
    """A catalogue course has module titles, outcomes and hours but no topics, exercises or assessment plan."""
    hours = split_hours(float(c.duration_hours), len(c.modules))
    capstone = next((m for m in c.modules if "project" in m.lower()), "")
    defaults = tpl.meta().get("default_assessments") or []
    facts = CourseFacts(
        title=c.title_ar if lang == "ar" else c.title_en, language=lang, origin="catalogue",
        objectives=[c.description_ar if lang == "ar" else c.description_en], audience=c.target_audience,
        prerequisites=list(c.prerequisites), outcomes=list(c.learning_outcomes),
        modules=[{"title": t, "topics": [], "hours": h, "exercises": []} for t, h in zip(c.modules, hours)],
        hours=float(c.duration_hours), exercises=[], capstone=capstone, department=c.department,
        level=c.skill_level.value, industry=(c.industry or [None])[0], country=(c.country or [None])[0].value
        if c.country else None, framework=c.accreditation_alignment, course_id=c.course_id, course=c,
        assessments=[{"name": a["name"].get(lang) or a["name"]["en"], "type": a["type"],
                      "weight_pct": a["weight_pct"]} for a in defaults], default_assessments=bool(defaults))
    facts.notes.append(f"No approved curriculum for {c.course_id}: built from the catalogue course (module titles "
                       "only; topics come from reference syllabi where found). The assessment plan is the academy "
                       "default - generate and approve a curriculum for the real one")
    return facts


def _add_catalogue_extras(facts: CourseFacts, kb) -> None:
    c = facts.course
    facts.instructors = kb.instructors_for(c.course_id)
    facts.schedules = kb.schedules_for(c.course_id)
    facts.promotions = kb.promotions_for(c.course_id)
    facts.best_price = kb.best_price(c.course_id)[0]
    facts.sources.append(SourceRef(source_id=c.source_id or CATALOGUE_SOURCE_ID, title=f"Academy course {c.title_en}",
                                   record_id=c.course_id))
    if facts.instructors:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-INSTRUCTORS", title="Instructor profiles",
                                       record_id=", ".join(i.instructor_id for i in facts.instructors)))
    if facts.schedules:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-SCHEDULES", title="Cohort schedules",
                                       record_id=c.course_id))
    if c.fee is not None:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-PRICING", title="Course fees and current promotions",
                                       record_id=c.course_id))


# ---------------------------------------------------------------------------- reference syllabi per module (RAG)
def add_module_references(facts: CourseFacts, per_module: int = 2) -> None:
    """Search curriculum_refs for every teaching module. A passage is kept only if it shares a word with the module
    (nearest neighbours always exist; relevant ones may not), so unrelated syllabi never leak into a document."""
    rag = refs_rag()
    if rag is None:
        facts.notes.append("Reference syllabi unavailable (RAG) - module notes come from the curriculum only")
        return
    try:
        ensure_references_indexed(rag)
    except Exception as exc:
        log.warning("Reference indexing failed: %s", exc)
        return
    found = 0
    for n, mod in facts.teaching_modules:
        words = _words(f"{mod['title']} {' '.join(mod.get('topics') or [])}")
        query = f"{mod['title']}. {'; '.join(mod.get('topics') or [])}"
        try:
            hits = rag.search(query, k=6, where={"department": facts.department} if facts.department else None)
            if not hits and facts.department:
                hits = rag.search(query, k=6)
        except Exception as exc:
            log.warning("Reference search failed for %s: %s", mod["title"], exc)
            continue
        kept = []
        for h in hits:
            sec = parse_section(h)
            if sec and words & _words(f"{sec.title} {' '.join(sec.topics)}"):
                kept.append(sec)
            if len(kept) >= per_module:
                break
        if kept:
            facts.module_refs[n] = kept
            found += 1
            for s in kept:
                facts.sources.append(SourceRef(source_id=s.source_id, title=s.citation, record_id=s.doc or s.title))
    if not found:
        facts.notes.append("No reference syllabus matched these modules - add one to data/curriculum/references/")
