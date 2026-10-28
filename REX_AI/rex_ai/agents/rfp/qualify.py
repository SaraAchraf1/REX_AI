"""Can the academy bid? (brief, Agent 2 item 15: "assess whether the academy may qualify").

Owner: Member 2. Two checks, both explained in qualification_notes:
  1. capability  - which catalogue courses could deliver the training. Course tags are matched on the tender
                   text (Arabic topics mapped to English tags), then the shared RAG retriever over academy_kb
                   (the index the REX Assistant uses) adds semantically close courses.
  2. eligibility - requirements in the notice compared with data/rfp/academy_profile.yaml
                   (registration country, years in business, ISO certificates, licences, capacity).

The result is advice for the bid team, never a promise: unknown requirements lower the fit to "possible".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from rex_ai import config
from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.agents.rfp import taxonomy as tx
from rex_ai.core.schemas import QualificationFit, SourceRef

log = logging.getLogger("rex.rfp.qualify")

PROFILE_SOURCE_ID = "SRC-ACAD-PROFILE"


@dataclass
class Assessment:
    fit: QualificationFit
    notes: list[str]
    course_ids: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)


def load_profile() -> dict:
    path = config.RFP_DIR / "academy_profile.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


class Qualifier:
    def __init__(self, profile: Optional[dict] = None, use_rag: bool = True):
        from rex_ai.agents.rex.knowledge_base import get_kb
        self.profile = profile if profile is not None else load_profile()
        self.kb = get_kb()
        self.retriever = None
        if use_rag:
            try:                                    # same RAG index (academy_kb) the REX Assistant searches
                from rex_ai.core import data_sync
                self.retriever = data_sync.get_assistant().retriever
            except Exception as exc:
                log.warning("Course retriever unavailable, using tag matching only: %s", exc)

    # ---------------------------------------------------------------- capability
    def match_courses(self, text: str, country: str, limit: int = 3) -> list[tuple[str, str, bool]]:
        """[(course_id, why, semantic_only)] best first. Course tags matched on the tender topic decide; the RAG
        retriever is asked only when no tag matches (e.g. unusual wording), and such matches stay tentative."""
        en_text = text + " " + " ".join(tag for ar, tag in tx.ARABIC_TOPICS.items() if contains_any(text, [ar]))
        scored: dict[str, tuple[float, str]] = {}
        for c in self.kb.courses.values():
            if country not in [x.value for x in c.country]:
                continue
            tags = [t for t in c.tags if t not in tx.GENERIC_TAGS and contains_any(en_text, [t])]
            if tags:
                scored[c.course_id] = (len(tags), "topics: " + ", ".join(tags))
        if self.retriever is not None and not scored:
            try:
                min_score = self.retriever.thresholds["course_min"]
                for h in self.retriever.search(en_text[:1000], k=3, filters={"type": "course"}, min_score=min_score):
                    cid = h.chunk.metadata.get("course_id")
                    course = self.kb.courses.get(cid)
                    if course and country in [x.value for x in course.country] and cid not in scored:
                        scored[cid] = (h.score - 10, f"semantic match {h.score:.2f}, confirm the topic fit")
            except Exception as exc:
                log.warning("Course search failed: %s", exc)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])[:limit]
        return [(cid, why, score < 0) for cid, (score, why) in ranked]

    # ---------------------------------------------------------------- eligibility
    def check_requirements(self, text: str, country: str) -> tuple[list[str], list[str], list[str]]:
        """(blockers, checks, info) found in the notice text."""
        p = self.profile
        blockers, checks, info = [], [], []
        registered = set(p.get("registered_countries", []))
        if contains_any(text, tx.SAUDI_REGISTRATION) and "SA" not in registered:
            (checks if p.get("saudi_partner") else blockers).append(
                "requires a Saudi-registered bidder (commercial registration / Etimad)"
                + (" - possible through the academy's Saudi partner" if p.get("saudi_partner") else ""))
        if contains_any(text, tx.EGYPT_REGISTRATION) and "EG" not in registered:
            blockers.append("requires an Egyptian-registered company")
        if contains_any(text, tx.TVTC_LICENCE) and not any("tvtc" in normalize(x) for x in p.get("licences", [])):
            blockers.append("requires a TVTC training licence, which the academy profile does not list")

        years = p.get("years_in_business")
        experience = [s for s in re.split(r"[\n.;؛]", normalize(text)) if re.search(r"experience|خبر|سابقه اعمال", s)]
        for m in re.finditer(r"(\d{1,2})\s*\+?\s*(?:years?|سنوات|سنه|اعوام|عاما|عام)", " ".join(experience)):
            needed = int(m.group(1))
            if years is not None and needed > years:
                blockers.append(f"asks for {needed} years of experience; the academy has {years}")
            elif years is not None:
                info.append(f"experience requirement ({needed} years) met")
            break

        held = [normalize(c) for c in p.get("certifications", [])]
        for iso in dict.fromkeys(re.findall(r"iso\s*/?\s*(?:iec\s*)?(\d{4,5})", normalize(text))):
            if any(iso in h for h in held):
                info.append(f"ISO {iso} held")
            else:
                blockers.append(f"requires ISO {iso}, not in the academy profile")

        plain = text.lower().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٬", "0123456789,"))   # normalize() drops commas
        trainees = [int(n.replace(",", "")) for n in
                    re.findall(r"(\d[\d,]{2,})\s*(?:trainees|participants|employees|learners|teachers|متدرب|موظف)",
                               plain)]
        cap = p.get("max_trainees_per_year")
        if trainees and cap and max(trainees) > cap:
            checks.append(f"{max(trainees):,} trainees exceeds the yearly capacity of {cap:,}; may need a partner")
        if contains_any(text, tx.LOCAL_CONTENT):
            checks.append("local-content requirement: confirm the academy's local-content score")
        if contains_any(text, tx.BID_BOND):
            info.append("bid/performance guarantee required - involve finance early")
        if contains_any(text, tx.ACCREDITED_CERTIFICATES):
            checks.append("asks for accredited certificates: confirm evidence; never claim accreditation without it")
        return blockers, checks, info

    # ---------------------------------------------------------------- verdict
    def assess(self, text: str, eligibility: Optional[str], country: str, topic: Optional[str] = None) -> Assessment:
        """text: the whole notice (requirements are read here); topic: title + description (course matching)."""
        courses = self.match_courses(topic or text, country)
        blockers, checks, info = self.check_requirements(text, country)
        notes = ["Matching courses: " + "; ".join(f"{cid} ({why})" for cid, why, _ in courses)] if courses else \
                ["No course in the academy catalogue covers this topic"]
        notes += [f"Blocker: {b}" for b in blockers] + [f"Check: {c}" for c in checks] + info
        if not eligibility:
            notes.append("Eligibility not published in the notice - read the tender documents")

        if blockers or not courses:
            fit = QualificationFit.unlikely
        elif checks or not eligibility or all(semantic for _, _, semantic in courses):
            fit = QualificationFit.possible
        else:
            fit = QualificationFit.likely
        sources = [SourceRef(source_id="SRC-ACAD-COURSES", title="Academy course catalogue", record_id=cid)
                   for cid, _, _ in courses]
        if self.profile:
            sources.append(SourceRef(source_id=PROFILE_SOURCE_ID, title="Academy profile for tender eligibility",
                                     record_id="academy_profile.yaml"))
        return Assessment(fit, notes, [cid for cid, _, _ in courses], sources)
