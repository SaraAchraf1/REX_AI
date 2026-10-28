"""Agent 3 - Curriculum Research & Accreditation Alignment Agent (owner: Member 2).

    request (discovery answers)  ->  research (catalogue, RAG references, industry skills, framework)
        ->  structured curriculum (rules; refined by the LLM when configured)  ->  finalize (hours, weights,
            requested outcomes, no accreditation claims)  ->  alignment reports (framework, country checklist,
            quality baseline)  ->  stored as version 1, status draft
    review: submit (pending_approval) -> approved / changes_requested / rejected
    revise: reviewer instructions or edited content -> version n+1 (draft), re-aligned
    approved -> HandoffEvent curriculum -> document (docs/interfaces.md §4) ; PDF export at any time

    from rex_ai.agents.curriculum.agent import CurriculumAgent
    cur = CurriculumAgent().generate(CurriculumRequest(course_title=..., target_audience=..., ...))
"""
from __future__ import annotations

import copy
import logging
import math
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.curriculum import frameworks as fw
from rex_ai.agents.curriculum import llm
from rex_ai.agents.curriculum.alignment import Aligner
from rex_ai.agents.curriculum.generator import HOURS_PER_WEEK, T, CurriculumBuilder, finalize, guard_claims, \
    split_hours
from rex_ai.agents.curriculum.research import infer_department, research
from rex_ai.agents.curriculum.revise import apply_instructions
from rex_ai.core import db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, Curriculum, CurriculumImport, CurriculumRequest,
                                 CurriculumResearch, CurriculumRevision, FrameworkInfo, GapReport, ReviewAction,
                                 SourceRef)

log = logging.getLogger("rex.curriculum")

AGENT = AgentName.curriculum.value
EDITABLE = {"course_title", "objectives", "target_audience", "prerequisites", "learning_outcomes", "modules",
            "training_hours", "practical_exercises", "assessments", "capstone_project"}


class NotFound(LookupError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _valid_content(content: dict) -> bool:
    """LLM output must still be a valid curriculum; otherwise the rule-based result is kept."""
    try:
        Curriculum(curriculum_id="check", **{k: v for k, v in content.items() if k in EDITABLE})
        return True
    except (ValidationError, TypeError) as exc:
        log.warning("LLM curriculum rejected, using the rule-based one: %s", exc)
        return False


_synced = False


class CurriculumAgent:
    def __init__(self, use_rag: bool = True, aligner: Optional[Aligner] = None):
        global _synced
        from rex_ai.agents.rex.knowledge_base import get_kb
        self.kb = get_kb()
        self.use_rag = use_rag
        self.aligner = aligner or Aligner(use_embeddings=use_rag)
        if not _synced:
            fw.sync_to_db()                       # frameworks -> accreditation_standards table
            _synced = True

    # ================================================================== discovery + research
    def fill_request(self, req: CurriculumRequest) -> dict:
        """Use stored catalogue data instead of asking again (brief §10 implementation note)."""
        data = req.model_dump(mode="json")
        if req.course_id:
            course = self.kb.courses.get(req.course_id.upper())
            if not course:
                raise NotFound(f"Course {req.course_id} not found in the catalogue")
            data["course_id"] = course.course_id
            filled = {"course_title": course.title_ar if req.language.value == "ar" else course.title_en,
                      "target_audience": course.target_audience, "skill_level": course.skill_level.value,
                      "total_hours": course.duration_hours, "department": course.department,
                      "industry": course.industry[0] if course.industry else None}
            for k, v in filled.items():
                data[k] = data.get(k) or v
            if not data["expected_outcomes"]:
                data["expected_outcomes"] = list(course.learning_outcomes)
        return data

    def research(self, req: CurriculumRequest) -> CurriculumResearch:
        return research(self.fill_request(req), self.kb, self.use_rag).summary

    # ================================================================== generate
    def generate(self, req: CurriculumRequest) -> Curriculum:
        job_id = db.start_job(AGENT, req.model_dump(mode="json"))
        try:
            cur = self._generate(req)
        except Exception as exc:
            db.finish_job(job_id, "failed", error=repr(exc))
            db.log_event("ERROR", "curriculum_generate_failed", agent=AGENT, detail=repr(exc))
            raise
        db.finish_job(job_id, "succeeded", output_ref=cur.curriculum_id)
        db.log_event("INFO", "curriculum_generated", agent=AGENT,
                     detail=f"{job_id}: {cur.curriculum_id} '{cur.course_title}' by {cur.metadata.get('generated_by')}")
        return cur

    def _generate(self, req: CurriculumRequest) -> Curriculum:
        filled = self.fill_request(req)
        lang = filled["language"]
        bundle = research(filled, self.kb, self.use_rag)
        if lang == "ar" and filled.get("course_id") and not llm.enabled():
            bundle.summary.notes.append("The catalogue lists this course's modules and outcomes in English only, so "
                                        "they stay in English; set OPENAI_API_KEY or add Arabic catalogue data for a "
                                        "fully Arabic curriculum")
        draft = CurriculumBuilder(filled, bundle, self.kb).build()
        evidence = draft.pop("evidence")
        generated_by = "rules"
        refined = llm.refine(filled, draft, bundle.context, lang) if llm.enabled() else None
        if refined and _valid_content({**draft, **refined}):
            draft, removed = finalize({**copy.deepcopy(draft), **refined}, filled, lang)
            generated_by = "llm"
        if generated_by == "rules":
            draft, removed = finalize(draft, filled, lang)
        now = _now()
        research_summary = bundle.summary.model_dump(mode="json", exclude={"sources"})
        metadata = {
            "department": bundle.department, "target_audience": filled["target_audience"],
            "skill_level": filled["skill_level"], "industry": filled.get("industry") or
            (bundle.industries[0] if bundle.industries else None), "duration_hours": draft["training_hours"],
            "duration_weeks": math.ceil(draft["training_hours"] / HOURS_PER_WEEK),
            "learning_outcomes": len(draft["learning_outcomes"]), "prerequisites": draft["prerequisites"],
            "accreditation_framework": bundle.framework.short_name if bundle.framework else None,
            "framework_id": bundle.framework.framework_id if bundle.framework else None,
            "framework_reason": bundle.summary.framework_reason, "country": filled["country"],
            "data_sources": list(dict.fromkeys(s.source_id for s in bundle.summary.sources)),
            "update_date": date.today().isoformat(), "generated_by": generated_by, "evidence": evidence,
            "claims_removed": removed, "research": research_summary, "origin": "generated",
        }
        cur = Curriculum(curriculum_id="CUR-" + uuid.uuid4().hex[:8].upper(), course_id=filled.get("course_id"),
                         language=lang, metadata=metadata, sources=bundle.summary.sources,
                         status=ApprovalStatus.draft, version=1, created_at=now, updated_at=now, **draft)
        return self._save_aligned(cur, filled, change_note=f"generated ({generated_by})", changed_by=AGENT)

    # ================================================================== import a proposed curriculum
    def import_curriculum(self, imp: CurriculumImport) -> Curriculum:
        """Bring in an existing catalogue course or an instructor's proposal, to compare it with a framework."""
        now = _now()
        lang = imp.language.value
        if imp.course_id:
            c = self.kb.courses.get(imp.course_id.upper())
            if not c:
                raise NotFound(f"Course {imp.course_id} not found in the catalogue")
            hours = split_hours(float(c.duration_hours), len(c.modules))
            content = {
                "course_title": c.title_ar if lang == "ar" else c.title_en,
                "objectives": [c.description_ar if lang == "ar" else c.description_en],
                "target_audience": c.target_audience, "prerequisites": list(c.prerequisites),
                "learning_outcomes": list(c.learning_outcomes),
                "modules": [{"title": m, "topics": [], "hours": h, "exercises": []} for m, h in zip(c.modules, hours)],
                "training_hours": float(c.duration_hours), "practical_exercises": [], "assessments": [],
                "capstone_project": next((m for m in c.modules if "project" in m.lower()), ""),
            }
            department, level, industry = c.department, c.skill_level.value, (c.industry or [None])[0]
            sources = [SourceRef(source_id=c.source_id, title=f"Academy course {c.title_en}", record_id=c.course_id)]
            origin = f"catalogue course {c.course_id}"
        else:
            p = imp.proposal
            content = p.model_dump(mode="json", exclude={"department", "skill_level"})
            content["training_hours"] = p.training_hours or sum(m.hours for m in p.modules)
            department = p.department or infer_department(p.course_title, " ".join(p.learning_outcomes), self.kb)
            level, industry = p.skill_level.value if p.skill_level else None, None
            sources = [SourceRef(source_id="USER-PROPOSAL", title="Curriculum proposed by the course owner",
                                 record_id=p.course_title)]
            origin = "proposal"
        content, removed = guard_claims(content)
        framework = fw.find_framework(imp.framework) if imp.framework else None
        if imp.framework and framework is None:
            raise NotFound(f"Framework '{imp.framework}' not found")
        reason = "selected by the user"
        if framework is None:
            framework, reason = fw.suggest_framework(content["course_title"], department, imp.country.value)
        metadata = {
            "department": department, "target_audience": content.get("target_audience"), "skill_level": level,
            "industry": industry, "duration_hours": content["training_hours"],
            "duration_weeks": math.ceil(content["training_hours"] / HOURS_PER_WEEK) if content["training_hours"] else None,
            "learning_outcomes": len(content.get("learning_outcomes") or []), "prerequisites": content.get("prerequisites"),
            "accreditation_framework": framework.short_name if framework else None,
            "framework_id": framework.framework_id if framework else None, "framework_reason": reason,
            "country": imp.country.value, "data_sources": [s.source_id for s in sources],
            "update_date": date.today().isoformat(), "generated_by": "imported", "origin": origin,
            "claims_removed": removed,
        }
        if framework:
            sources.append(framework.source_ref())
        cur = Curriculum(curriculum_id="CUR-" + uuid.uuid4().hex[:8].upper(), course_id=imp.course_id and
                         imp.course_id.upper(), language=lang, metadata=metadata, sources=sources,
                         status=ApprovalStatus.draft, version=1, created_at=now, updated_at=now, **content)
        cur = self._save_aligned(cur, imp.model_dump(mode="json"), change_note=f"imported from {origin}",
                                 changed_by=AGENT)
        db.log_event("INFO", "curriculum_imported", agent=AGENT, detail=f"{cur.curriculum_id}: {origin}")
        return cur

    # ================================================================== alignment
    def _frameworks_for(self, cur: Curriculum) -> list[fw.Framework]:
        """Selected/suggested framework + the country's design checklist + the academy quality baseline."""
        found = [fw.load_frameworks().get(cur.metadata.get("framework_id") or ""),
                 fw.country_framework(cur.metadata.get("country")), fw.baseline()]
        return list({f.framework_id: f for f in found if f}.values())

    def _save_aligned(self, cur: Curriculum, request: dict, change_note: str, changed_by: str) -> Curriculum:
        data = cur.model_dump(mode="json")
        reports = [self.aligner.compare(data, f, request, cur.metadata.get("department"))
                   for f in self._frameworks_for(cur)]
        cur.metadata["alignment"] = [{"framework_id": r.framework_id, "framework": r.framework,
                                      "coverage_pct": r.coverage_pct, "report_id": r.report_id} for r in reports]
        db.save_curriculum(cur.model_dump(mode="json"), request, change_note=change_note, changed_by=changed_by)
        for r in reports:
            r.curriculum_id = cur.curriculum_id
            db.save_gap_report(r.model_dump(mode="json"))
        return cur

    def gap_report(self, curriculum_id: str, framework: Optional[str] = None) -> GapReport:
        row = self._row(curriculum_id)
        cur = Curriculum(**row["content"])
        if framework:
            f = fw.find_framework(framework)
            if f is None:
                raise NotFound(f"Framework '{framework}' not found. Available: "
                               + ", ".join(x.short_name for x in fw.load_frameworks().values()))
        else:
            f = fw.load_frameworks().get(cur.metadata.get("framework_id") or "") or \
                fw.suggest_framework(cur.course_title, cur.metadata.get("department"), cur.metadata.get("country", "EG"))[0]
        report = self.aligner.compare(cur.model_dump(mode="json"), f, row["request"], cur.metadata.get("department"))
        db.save_gap_report(report.model_dump(mode="json"))
        summary = [a for a in cur.metadata.get("alignment", []) if a["framework_id"] != f.framework_id]
        cur.metadata["alignment"] = summary + [{"framework_id": f.framework_id, "framework": f.name,
                                                "coverage_pct": report.coverage_pct, "report_id": report.report_id}]
        db.save_curriculum(cur.model_dump(mode="json"), row["request"], change_note=f"gap report {f.framework_id}",
                           changed_by=AGENT)
        db.log_event("INFO", "curriculum_gap_report", agent=AGENT,
                     detail=f"{curriculum_id}: {f.framework_id} {report.coverage_pct}%")
        return report

    def gap_reports(self, curriculum_id: str) -> list[GapReport]:
        """Latest report per framework, primary framework first."""
        cur = self.get(curriculum_id)
        latest: dict[str, GapReport] = {}
        for r in db.list_gap_reports(curriculum_id):
            latest.setdefault(r.get("framework_id") or r["framework"], GapReport(**r))
        primary = cur.metadata.get("framework_id")
        return sorted(latest.values(), key=lambda r: (r.framework_id != primary, r.framework_id == fw.BASELINE_ID))

    # ================================================================== review workflow
    def review(self, curriculum_id: str, action: ReviewAction) -> Curriculum:
        row = self._row(curriculum_id)
        cur = Curriculum(**row["content"])
        a = action.action
        if a == ApprovalStatus.draft:
            raise ValueError("A review action is pending_approval, approved, changes_requested or rejected")
        if cur.status == ApprovalStatus.approved and a != ApprovalStatus.rejected:
            raise ValueError("This curriculum is already approved; submit a revision to change it")
        if a != ApprovalStatus.pending_approval:
            db.save_review("curriculum", curriculum_id, a.value, action.reviewer, action.comment)
        if action.comment or a != ApprovalStatus.pending_approval:
            cur.reviewer_comments.append(f"{action.reviewer} ({a.value})" + (f": {action.comment}"
                                                                             if action.comment else ""))
        cur.status = a
        cur.updated_at = _now()
        if a == ApprovalStatus.approved:
            event = db.save_handoff(AGENT, AgentName.document.value, "curriculum", curriculum_id)
            cur.metadata.update(approved_by=action.reviewer, approved_at=cur.updated_at.isoformat(),
                                handoff_event=event["event_id"])
        db.save_curriculum(cur.model_dump(mode="json"), row["request"], change_note=f"review: {a.value}",
                           changed_by=action.reviewer)
        db.log_event("INFO", "curriculum_review", agent=AGENT,
                     detail=f"{curriculum_id} v{cur.version}: {a.value} by {action.reviewer}: {action.comment or ''}")
        return cur

    def revise(self, curriculum_id: str, rev: CurriculumRevision) -> Curriculum:
        """Apply a modification request: new version, back to draft, alignment re-checked."""
        row = self._row(curriculum_id)
        cur = Curriculum(**row["content"])
        lang = cur.language.value
        data = cur.model_dump(mode="json")
        before = {k: data[k] for k in EDITABLE}
        applied, not_applied = [], []
        if rev.content:
            unknown = set(rev.content) - EDITABLE
            if unknown:
                raise ValueError(f"Fields that cannot be edited: {', '.join(sorted(unknown))}")
            data.update(rev.content)
            if "modules" in rev.content and "training_hours" not in rev.content:
                data["training_hours"] = float(sum(float(m.get("hours") or 0) for m in data["modules"]))
            applied.append("edited: " + ", ".join(sorted(rev.content)))
        if rev.instructions:
            done = llm.revise(data, rev.instructions, lang) if llm.enabled() else None
            if done and _valid_content({**data, **done}):
                data.update(done)
                applied.append(f"LLM applied: {rev.instructions[:200]}")
            else:
                data, ok, ko = apply_instructions(data, rev.instructions, capstone_title=T[lang]["capstone_module"])
                applied += ok
                not_applied += ko
        data, removed = guard_claims(data)
        try:
            new = Curriculum(**data)
        except Exception as exc:
            raise ValueError(f"The edited curriculum is not valid: {exc}") from exc
        note = rev.instructions or "edited content"
        db.save_review("curriculum", curriculum_id, ApprovalStatus.changes_requested.value, rev.reviewer, note)
        new.reviewer_comments = cur.reviewer_comments + [f"{rev.reviewer} (changes_requested): {note}"]
        new.metadata["last_revision"] = {"by": rev.reviewer, "applied": applied, "not_applied": not_applied,
                                         "claims_removed": removed, "at": _now().isoformat()}
        changed = {k: new.model_dump(mode="json")[k] for k in EDITABLE} != before
        if not changed:
            new.status = ApprovalStatus.changes_requested
            db.save_curriculum(new.model_dump(mode="json"), row["request"], change_note="revision: nothing applied",
                               changed_by=rev.reviewer)
            return new
        new.version = cur.version + 1
        new.status = ApprovalStatus.draft
        new.updated_at = _now()
        new.metadata.update(duration_hours=new.training_hours, learning_outcomes=len(new.learning_outcomes),
                            prerequisites=new.prerequisites)
        for k in ("approved_by", "approved_at", "handoff_event"):
            new.metadata.pop(k, None)
        new = self._save_aligned(new, row["request"], change_note="; ".join(applied)[:500], changed_by=rev.reviewer)
        db.log_event("INFO", "curriculum_revised", agent=AGENT,
                     detail=f"{curriculum_id} v{new.version} by {rev.reviewer}: {len(applied)} applied, "
                            f"{len(not_applied)} not applied")
        return new

    # ================================================================== read / export
    def _row(self, curriculum_id: str) -> dict:
        row = db.get_curriculum(curriculum_id)
        if not row:
            raise NotFound(f"Curriculum {curriculum_id} not found")
        return row

    def get(self, curriculum_id: str) -> Curriculum:
        return Curriculum(**self._row(curriculum_id)["content"])

    def search(self, **filters) -> list[Curriculum]:
        return [Curriculum(**r["content"]) for r in db.list_curricula(**filters)]

    def history(self, curriculum_id: str) -> dict:
        self._row(curriculum_id)
        return {
            "versions": [{k: v for k, v in r.items() if k != "content"} | {"status": r["content"].get("status")}
                         for r in db.list_curriculum_versions(curriculum_id)],
            "reviews": db.list_reviews("curriculum", curriculum_id),
            "gap_reports": [{"report_id": r["report_id"], "framework": r["framework"], "coverage_pct": r["coverage_pct"],
                             "curriculum_version": r.get("curriculum_version"), "created_at": r.get("created_at")}
                            for r in db.list_gap_reports(curriculum_id)],
            "handoffs": db.list_handoffs(curriculum_id),
        }

    def export_pdf(self, curriculum_id: str, include_alignment: bool = True) -> Path:
        from rex_ai.agents.curriculum.pdf import render_curriculum_pdf
        cur = self.get(curriculum_id)
        reports = [r.model_dump(mode="json") for r in self.gap_reports(curriculum_id)] if include_alignment else []
        suffix = "" if cur.status == ApprovalStatus.approved else f"_{cur.status.value}"
        path = config.CURRICULUM_OUTPUT_DIR / f"{curriculum_id}_v{cur.version}{suffix}.pdf"
        render_curriculum_pdf(cur.model_dump(mode="json"), path, report=reports[0] if reports else None,
                              other_reports=reports[1:], reviews=db.list_reviews("curriculum", curriculum_id))
        db.log_event("INFO", "curriculum_pdf", agent=AGENT, detail=f"{curriculum_id}: {path.name}")
        return path

    @staticmethod
    def frameworks() -> list[FrameworkInfo]:
        return [f.info() for f in fw.load_frameworks().values()]
