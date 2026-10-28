"""Collect the approved facts a campaign may use (brief, Agent 5 item 36 "read approved course details" and item 41
"connect with the Curriculum Agent to ensure accurate course information"):

  catalogue (always)     title, description, level, audience, hours, formats, careers, certificate, prerequisites
  curriculum (Agent 3)   the request's curriculum_id, else the course's newest APPROVED curriculum: its outcomes,
                         modules and hours replace the catalogue's (the catalogue may be out of date)
  document (Agent 4)     an approved document handed over to Marketing: its curriculum is used
  pricing / schedules    fee, installments, best current promotion and its condition, upcoming cohorts, seats left
  instructors            name, title, years of experience
  RAG academy_kb         the FAQs a lead is most likely to ask about this course (objection handling)
  leads (REX)            how many consented leads asked about the course and their common goals (no personal data)

The copy may only use these facts: they are stored in Campaign.course_facts_used and every number in the copy is
checked against them (checks.py).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from rex_ai.agents.curriculum.agent import NotFound  # one 404 type for every agent (api/main.py)
from rex_ai.core import db
from rex_ai.core.schemas import Course, Curriculum, SourceRef

log = logging.getLogger("rex.marketing.facts")

CATALOGUE_SOURCE_ID = "SRC-ACAD-COURSES"
# Promotions limited to one kind of buyer (read from the promotion's English name and condition) and the personas
# that may be offered them; other promotions are open to everyone
PROMO_TARGETS = {"students": re.compile(r"student|graduat", re.I),
                 "groups": re.compile(r"group|team|\b\d+\s+or\s+more\b|corporate", re.I)}
PROMO_AUDIENCES = {"students": {"students"}, "managers": {"groups"}, "corporate": {"groups"}}
# "Pay in full at least 3 weeks before the cohort starts" -> the offer ends 21 days before the start date
LEAD_TIME = re.compile(r"(\d+)\s*(weeks?|days?)\s+before", re.I)
# What leads usually ask before they register; answered from the academy FAQs (via the RAG index)
OBJECTION_QUERIES = {
    "en": ["Can I pay in installments?", "Do I get a certificate?", "Are sessions recorded if I miss one?",
           "Is there a refund policy?"],
    "ar": ["هل يمكن الدفع بالتقسيط؟", "هل أحصل على شهادة؟", "هل المحاضرات مسجلة؟", "ما سياسة الاسترداد؟"],
}


@dataclass
class MarketingFacts:
    course: Course
    language: str
    origin: str                                   # "curriculum" | "catalogue"
    outcomes: list[str]
    modules: list[str]
    hours: float
    weeks: int
    curriculum_id: Optional[str] = None
    curriculum_version: Optional[int] = None
    curriculum_status: Optional[str] = None
    document_id: Optional[str] = None
    document_status: Optional[str] = None
    capstone: Optional[str] = None
    best_price: Optional[float] = None
    promotion: Optional[dict] = None              # the best current promotion
    promotions: list[dict] = field(default_factory=list)
    schedules: list[dict] = field(default_factory=list)
    instructors: list[dict] = field(default_factory=list)
    objections: list[dict] = field(default_factory=list)
    lead_insights: dict = field(default_factory=dict)
    sources: list[SourceRef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.course.title_ar if self.language == "ar" else self.course.title_en

    @property
    def description(self) -> str:
        return self.course.description_ar if self.language == "ar" else self.course.description_en

    @property
    def next_start(self) -> Optional[dict]:
        return self.schedules[0] if self.schedules else None

    def choose_promotion(self, persona_id: Optional[str], today: Optional[date] = None) -> None:
        """The offer to advertise: the biggest discount this audience is eligible for. A student discount is never
        the headline offer of a campaign for managers, and a group discount only for teams and companies. An offer
        that must be paid "N weeks before the cohort starts" ends then for the next cohort: if that date has passed,
        it is not advertised. promotion["deadline"] is the date the copy quotes."""
        today = today or date.today()
        allowed = PROMO_AUDIENCES.get(persona_id or "", set())
        start = date.fromisoformat(self.next_start["start_date"]) if self.next_start else None
        eligible = []
        for p in self.promotions:
            if p.get("for") is not None and p["for"] not in allowed:
                continue
            deadline = date.fromisoformat(p["valid_until"])
            if p.get("lead_days") and start:
                deadline = min(deadline, start - timedelta(days=p["lead_days"]))
            if deadline < today:
                continue
            eligible.append({**p, "deadline": deadline.isoformat()})
        best = max(eligible, key=lambda p: (p["discount_pct"], p.get("for") is not None), default=None)
        self.promotion = best
        fee = self.course.fee
        self.best_price = round(fee * (1 - best["discount_pct"] / 100)) if best and fee is not None else None

    def snapshot(self) -> dict[str, Any]:
        """Campaign.course_facts_used: everything the copy may rely on, as it was when the campaign was made."""
        c = self.course
        return {
            "course_id": c.course_id, "title": self.title, "title_en": c.title_en, "title_ar": c.title_ar,
            "department": c.department, "skill_level": c.skill_level.value, "audience": c.target_audience,
            "description": self.description, "hours": self.hours, "weeks": self.weeks,
            "formats": [f.value for f in c.formats], "languages": [x.value for x in c.language],
            "outcomes": self.outcomes, "modules": self.modules, "capstone": self.capstone,
            "prerequisites": c.prerequisites, "careers": c.career_opportunities, "certificate": c.certificate,
            "tags": c.tags, "framework": c.accreditation_alignment, "countries": [x.value for x in c.country],
            "fee": c.fee, "currency": c.currency, "installments": c.installments, "best_price": self.best_price,
            "promotion": self.promotion, "promotions": self.promotions, "next_start": self.next_start,
            "schedules": self.schedules, "instructors": self.instructors, "origin": self.origin,
            "curriculum_id": self.curriculum_id, "curriculum_version": self.curriculum_version,
            "curriculum_status": self.curriculum_status, "document_id": self.document_id,
            "document_status": self.document_status, "objections": self.objections,
            "lead_insights": self.lead_insights, "data_status": "sample" if c.is_sample else "approved",
            "data_updated": str(c.last_updated) if c.last_updated else None,
        }


def _latest_approved(course_id: str) -> Optional[dict]:
    rows = [r for r in db.list_curricula(status="approved") if (r.get("course_id") or "").upper() == course_id]
    return rows[0] if rows else None


def collect(kb, course_id: str, curriculum_id: Optional[str] = None, document_id: Optional[str] = None,
            language: str = "en", retriever=None) -> MarketingFacts:
    course = kb.courses.get(course_id.upper())
    if not course:
        raise NotFound(f"Course {course_id} not found in the catalogue")
    notes: list[str] = []
    doc_status = None
    if document_id:
        doc = db.get_document(document_id)
        if not doc or not doc.get("content"):
            raise NotFound(f"Document {document_id} not found")
        if (doc.get("course_id") or "").upper() not in ("", course.course_id):
            raise ValueError(f"Document {document_id} is about {doc['course_id']}, not {course.course_id}")
        doc_status = doc["status"]
        if doc_status != "approved":
            notes.append(f"Document {document_id} is {doc_status}, not approved")
        curriculum_id = curriculum_id or doc.get("curriculum_id")

    row = None
    if curriculum_id:
        row = db.get_curriculum(curriculum_id)
        if not row:
            raise NotFound(f"Curriculum {curriculum_id} not found")
        if (row.get("course_id") or course.course_id).upper() != course.course_id:
            raise ValueError(f"Curriculum {curriculum_id} belongs to {row['course_id']}, not {course.course_id}")
    else:
        row = _latest_approved(course.course_id)

    if row:
        cur = Curriculum(**row["content"])
        facts = MarketingFacts(
            course=course, language=language, origin="curriculum", outcomes=list(cur.learning_outcomes),
            modules=[m.title for m in cur.modules], hours=float(cur.training_hours),
            weeks=course.duration_weeks, curriculum_id=cur.curriculum_id, curriculum_version=cur.version,
            curriculum_status=cur.status.value, capstone=cur.capstone_project or None)
        facts.sources.append(SourceRef(source_id="REX-CURRICULUM", record_id=cur.curriculum_id,
                                       title=f"Curriculum {cur.course_title} v{cur.version} ({cur.status.value})"))
        if cur.status.value != "approved":
            notes.append(f"Curriculum {cur.curriculum_id} is {cur.status.value}: the campaign can be reviewed but not "
                         "approved until the curriculum is approved")
        if cur.language.value != language:
            notes.append(f"The curriculum's outcomes are in {cur.language.value}; enable the LLM to translate them")
    else:
        facts = MarketingFacts(course=course, language=language, origin="catalogue",
                               outcomes=list(course.learning_outcomes), modules=list(course.modules),
                               hours=float(course.duration_hours), weeks=course.duration_weeks,
                               capstone=next((m for m in course.modules if "project" in m.lower()), None))
        notes.append(f"No approved curriculum for {course.course_id} yet: outcomes and modules come from the catalogue")
        if language == "ar":
            notes.append("Catalogue outcomes and modules are in English; enable the LLM to translate them")
    facts.document_id, facts.document_status = document_id, doc_status
    facts.notes = notes + facts.notes
    if document_id:
        facts.sources.append(SourceRef(source_id="REX-DOCUMENT", title=f"Document {document_id} ({doc_status})",
                                       record_id=document_id))
    _add_catalogue(facts, kb)
    facts.choose_promotion(None)                     # open offers only, until the audience is known
    _add_objections(facts, kb, retriever)
    _add_lead_insights(facts)
    if course.is_sample:
        facts.notes.append("Course data is SAMPLE data: replace it with the academy's approved data before publishing")
    return facts


def _add_catalogue(facts: MarketingFacts, kb) -> None:
    c, lang = facts.course, facts.language
    facts.sources.append(SourceRef(source_id=c.source_id or CATALOGUE_SOURCE_ID, title=f"Academy course {c.title_en}",
                                   record_id=c.course_id))
    def promo(p) -> dict:
        target = next((k for k, rx in PROMO_TARGETS.items() if rx.search(f"{p.name_en} {p.condition_en}")), None)
        lead = LEAD_TIME.search(p.condition_en or "")
        lead_days = int(lead.group(1)) * (7 if lead.group(2).lower().startswith("week") else 1) if lead else None
        return {"promo_id": p.promo_id, "name": p.name_ar if lang == "ar" else p.name_en,
                "discount_pct": p.discount_pct, "condition": p.condition_ar if lang == "ar" else p.condition_en,
                "valid_until": p.valid_until.isoformat(), "for": target, "lead_days": lead_days}

    facts.promotions = [promo(p) for p in kb.promotions_for(c.course_id)]
    if c.fee is not None:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-PRICING", title="Course fees and current promotions",
                                       record_id=c.course_id))
    else:
        facts.notes.append("No fee in the price list: the copy invites people to ask for the price")
    facts.schedules = [{"start_date": s.start_date.isoformat(), "format": s.format.value, "days": s.days,
                        "time": s.time, "location": s.location, "seats_left": s.seats_left}
                       for s in kb.schedules_for(c.course_id)[:3]]
    if facts.schedules:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-SCHEDULES", title="Cohort schedules", record_id=c.course_id))
    else:
        facts.notes.append("No upcoming cohort in the schedule: the copy says dates will be announced")
    facts.instructors = [{"instructor_id": i.instructor_id, "name": i.name, "title": i.title,
                          "years_experience": i.years_experience} for i in kb.instructors_for(c.course_id)]
    if facts.instructors:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-INSTRUCTORS", title="Instructor profiles",
                                       record_id=", ".join(i["instructor_id"] for i in facts.instructors)))


def _add_objections(facts: MarketingFacts, kb, retriever) -> None:
    """The FAQs that answer what leads ask before registering, found through REX's RAG index (academy_kb)."""
    faqs = {f.faq_id: f for f in kb.faqs}
    found: list[str] = []
    if retriever is not None:
        min_score = getattr(retriever, "thresholds", {}).get("faq_min", 0.0)
        for q in OBJECTION_QUERIES["ar" if facts.language == "ar" else "en"]:
            try:
                hits = retriever.search(q, k=1, filters={"type": "faq"}, min_score=min_score)
            except Exception as exc:
                log.warning("FAQ search failed: %s", exc)
                break
            for h in hits:
                fid = h.chunk.metadata.get("faq_id")
                if fid in faqs and fid not in found:
                    found.append(fid)
    if not found:                                    # no RAG: the FAQ categories that answer the same questions
        found = [f.faq_id for f in kb.faqs if f.category in ("payment", "certificate", "format", "refund")][:4]
    ar = facts.language == "ar"
    facts.objections = [{"faq_id": fid, "question": faqs[fid].q_ar if ar else faqs[fid].q_en,
                         "answer": faqs[fid].a_ar if ar else faqs[fid].a_en} for fid in found]
    if found:
        facts.sources.append(SourceRef(source_id="SRC-ACAD-FAQ", title="Academy FAQs", record_id=", ".join(found)))


def _add_lead_insights(facts: MarketingFacts) -> None:
    """What the consented leads interested in this course told REX (aggregated, no personal data)."""
    leads = db.lead_interest(facts.course.course_id)
    if not leads:
        facts.lead_insights = {"leads": 0}
        return
    profiles = [x["profile"] for x in leads]

    def top(key: str) -> list[str]:
        values = [str(p[key]) for p in profiles if p.get(key)]
        return [v for v, _ in Counter(values).most_common(3)]

    facts.lead_insights = {"leads": len(leads), "skill_levels": top("skill_level"), "goals": top("career_goal"),
                           "backgrounds": top("background"), "formats": top("preferred_format"),
                           "education": top("education_level")}
    facts.sources.append(SourceRef(source_id="SRC-REX-LEADS", title=f"{len(leads)} consented leads (aggregated)",
                                   record_id=facts.course.course_id))
