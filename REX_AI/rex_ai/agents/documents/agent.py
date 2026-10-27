"""Agent 4 - Educational Content & Document Generation Agent (owner: Member 3).

    request (doc type, curriculum or course, language, audience, template)
        ->  collect facts (curriculum, catalogue, schedules, fees, instructors; RAG reference syllabi per module)
        ->  template sections (data/doc_templates/templates.yaml) filled by rules
        ->  optional LLM polish of unlocked sections, styled on the reference examples (RAG doc_templates)
        ->  accreditation-claim guard  ->  PDF (DRAFT watermark)  ->  stored as version 1, status draft
    edit: reviewer edits sections / removes them / gives instructions -> version n+1 (draft)
    review: submit (pending_approval) -> approved / changes_requested / rejected
    approved -> HandoffEvent document -> marketing (docs/interfaces.md §4); e-mail allowed only when approved

    from rex_ai.agents.documents.agent import DocumentAgent
    doc = DocumentAgent().generate(DocumentRequest(doc_type="participant_guide", curriculum_id="CUR-..."))
"""
from __future__ import annotations

import copy
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rex_ai import config
from rex_ai.agents.curriculum.generator import guard_claims
from rex_ai.agents.documents import llm
from rex_ai.agents.documents import templates as tpl
from rex_ai.agents.documents.builders import P, build_sections
from rex_ai.agents.documents.facts import CourseFacts, NotFound, add_module_references, collect
from rex_ai.agents.documents.mailer import send_document
from rex_ai.agents.documents.pdf import render_document_pdf
from rex_ai.core import db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, DocSection, DocumentEdit, DocumentEmail,
                                 DocumentPackRequest, DocumentRecord, DocumentRequest, DocumentType,
                                 DocumentTypeInfo, ReviewAction, SourceRef)

log = logging.getLogger("rex.documents")

AGENT = AgentName.document.value
# Document types whose modules get reference-syllabus notes from the RAG (key points, reading, extra questions)
NEEDS_REFERENCES = {DocumentType.participant_guide, DocumentType.facilitator_guide, DocumentType.presentation_outline,
                    DocumentType.question_bank, DocumentType.capstone_brief}

__all__ = ["DocumentAgent", "NotFound"]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _dedupe(refs: list[SourceRef]) -> list[SourceRef]:
    seen, out = set(), []
    for r in refs:
        key = (r.source_id, r.record_id or r.title)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _guard(sections: list[DocSection]) -> tuple[list[DocSection], int]:
    data, removed = guard_claims([s.model_dump() for s in sections])
    return [DocSection(**s) for s in data], removed


def _refs_context(facts: CourseFacts) -> str:
    parts = []
    for n, refs in sorted(facts.module_refs.items()):
        for r in refs:
            outcome = f"\nOutcome: {r.outcome}" if r.outcome else ""
            parts.append(f"[R{n}] Source: {r.citation}\nModule: {r.title}\n- " + "\n- ".join(r.topics) + outcome)
    return "\n\n".join(parts)[:5000]


class DocumentAgent:
    def __init__(self, use_rag: bool = True):
        from rex_ai.agents.rex.knowledge_base import get_kb
        self.kb = get_kb()
        self.use_rag = use_rag

    @staticmethod
    def types() -> list[DocumentTypeInfo]:
        return tpl.document_types()

    # ================================================================== generate
    def generate(self, req: DocumentRequest) -> DocumentRecord:
        job_id = db.start_job(AGENT, req.model_dump(mode="json"))
        try:
            doc = self._generate(req)
        except Exception as exc:
            db.finish_job(job_id, "failed", error=repr(exc))
            db.log_event("ERROR", "document_generate_failed", agent=AGENT, detail=repr(exc))
            raise
        db.finish_job(job_id, "succeeded", output_ref=doc.document_id)
        db.log_event("INFO", "document_generated", agent=AGENT,
                     detail=f"{job_id}: {doc.document_id} {doc.doc_type.value} '{doc.title}' by "
                            f"{doc.metadata.get('generated_by')}")
        return doc

    def _generate(self, req: DocumentRequest) -> DocumentRecord:
        try:
            template = tpl.get_template(req.doc_type.value, req.template_id)
        except LookupError as exc:
            raise NotFound(str(exc)) from exc
        explicit_lang = req.language.value if "language" in req.model_fields_set else None
        facts = collect(self.kb, req.curriculum_id, req.course_id, explicit_lang)
        lang = facts.language
        if self.use_rag and req.doc_type in NEEDS_REFERENCES:
            add_module_references(facts)
        sections = build_sections(facts, template, lang, client=req.audience)
        generated_by, example_refs = "rules", []
        if llm.enabled():
            context, example_refs = tpl.reference_examples(req.doc_type.value, f"{template.title['en']} "
                                                           f"{facts.title}") if self.use_rag else ("", [])
            context = "\n\n".join(x for x in (context, _refs_context(facts)) if x)
            returned = llm.refine(req.doc_type.value, facts.title, sections, facts.snapshot(), context, lang,
                                  req.instructions)
            sections, changed = llm.merge_sections(sections, returned)
            if changed:
                generated_by = "llm"
                if req.doc_type == DocumentType.question_bank:
                    sections = self._usage_note(sections, lang)
            else:
                example_refs = []                    # nothing from the examples ended up in the document
        sections, removed = _guard(sections)
        now = _now()
        doc_id = "DOC-" + uuid.uuid4().hex[:8].upper()
        type_title = template.title.get(lang) or template.title["en"]
        metadata = {
            "type_title": type_title, "course_title": facts.title, "generated_by": generated_by,
            "origin": facts.origin, "facts": facts.snapshot(), "notes": facts.notes, "claims_removed": removed,
            "prepared_for": req.audience, "pending_email": [str(e) for e in req.email_to or []],
            "reference_modules": sorted(facts.module_refs), "instructions": req.instructions,
        }
        sources = _dedupe([tpl.template_source(template), *facts.sources, *example_refs])
        rec = DocumentRecord(document_id=doc_id, doc_type=req.doc_type, title=f"{type_title}: {facts.title}",
                             course_id=facts.course_id, curriculum_id=facts.curriculum_id, language=lang,
                             file_path="", download_url=f"/documents/{doc_id}/download",
                             status=ApprovalStatus.draft, sources=sources, created_at=now, updated_at=now,
                             version=1, template_id=template.template_id, sections=sections, metadata=metadata)
        return self._save(rec, req.model_dump(mode="json"), f"generated ({generated_by})", AGENT)

    @staticmethod
    def _usage_note(sections: list[DocSection], lang: str) -> list[DocSection]:
        """The question-bank usage note must say who wrote the items."""
        L = P.get(lang, P["en"])
        for s in sections:
            if s.key == "usage":
                s.paragraphs = [L["usage_llm"] if p == L["usage_rules"] else p for p in s.paragraphs]
        return sections

    def generate_pack(self, req: DocumentPackRequest) -> list[DocumentRecord]:
        """Several documents from one curriculum or course (e.g. right after the curriculum was approved)."""
        out = []
        for t in dict.fromkeys(req.doc_types):
            payload = {"doc_type": t, "curriculum_id": req.curriculum_id, "course_id": req.course_id,
                       "audience": req.audience}
            if req.language:
                payload["language"] = req.language
            out.append(self.generate(DocumentRequest(**payload)))
        return out

    # ================================================================== storage / export
    def _pdf_path(self, rec: DocumentRecord) -> Path:
        suffix = "" if rec.status == ApprovalStatus.approved else f"_{rec.status.value}"
        return config.DOCUMENT_OUTPUT_DIR / f"{rec.document_id}_v{rec.version}_{rec.doc_type.value}{suffix}.pdf"

    def _render(self, rec: DocumentRecord) -> None:
        path = self._pdf_path(rec)
        render_document_pdf(rec.model_dump(mode="json"), path, reviews=db.list_reviews("document", rec.document_id))
        rec.file_path = str(path)

    def _save(self, rec: DocumentRecord, request: dict, change_note: str, changed_by: str) -> DocumentRecord:
        self._render(rec)
        db.save_document(rec.model_dump(mode="json"), request, change_note=change_note, changed_by=changed_by)
        return rec

    def _row(self, document_id: str) -> dict:
        row = db.get_document(document_id)
        if not row or not row.get("content"):
            raise NotFound(f"Document {document_id} not found")
        return row

    def get(self, document_id: str) -> DocumentRecord:
        return DocumentRecord(**self._row(document_id)["content"])

    def search(self, **filters) -> list[DocumentRecord]:
        return [DocumentRecord(**r["content"]) for r in db.list_documents(**filters)]

    def pdf(self, document_id: str) -> Path:
        """The PDF of the current version (rebuilt if the file was deleted)."""
        rec = self.get(document_id)
        path = Path(rec.file_path)
        if not path.exists():
            row = self._row(document_id)
            self._save(rec, row["request"], "pdf rebuilt", AGENT)
            path = Path(rec.file_path)
        db.log_event("INFO", "document_download", agent=AGENT, detail=f"{document_id}: {path.name}")
        return path

    def history(self, document_id: str) -> dict:
        self._row(document_id)
        return {
            "versions": [{k: v for k, v in r.items() if k != "content"} | {"status": r["content"].get("status")}
                         for r in db.list_document_versions(document_id)],
            "reviews": db.list_reviews("document", document_id),
            "handoffs": db.list_handoffs(document_id),
        }

    # ================================================================== review and edit
    def _source_check(self, rec: DocumentRecord) -> Optional[str]:
        """A document built from a curriculum can be approved only from the current, approved version of it."""
        facts = rec.metadata.get("facts") or {}
        cid = facts.get("curriculum_id")
        if not cid:
            return None
        row = db.get_curriculum(cid)
        if not row:
            return f"Its curriculum {cid} no longer exists"
        status, version = row["content"].get("status"), row["content"].get("version")
        if status != ApprovalStatus.approved.value:
            return f"Its curriculum {cid} is {status}: approve the curriculum first"
        if version != facts.get("curriculum_version"):
            return (f"Its curriculum {cid} changed (v{facts.get('curriculum_version')} -> v{version}): "
                    "regenerate the document from the approved version")
        return None

    def review(self, document_id: str, action: ReviewAction) -> DocumentRecord:
        row = self._row(document_id)
        rec = DocumentRecord(**row["content"])
        a = action.action
        if a == ApprovalStatus.draft:
            raise ValueError("A review action is pending_approval, approved, changes_requested or rejected")
        if rec.status == ApprovalStatus.approved and a != ApprovalStatus.rejected:
            raise ValueError("This document is already approved; edit it to make a new version")
        if a == ApprovalStatus.approved and (problem := self._source_check(rec)):
            raise ValueError(f"Cannot approve {document_id}. {problem}.")
        if a != ApprovalStatus.pending_approval:
            db.save_review("document", document_id, a.value, action.reviewer, action.comment)
        if action.comment or a != ApprovalStatus.pending_approval:
            rec.reviewer_comments.append(f"{action.reviewer} ({a.value})" + (f": {action.comment}"
                                                                             if action.comment else ""))
        rec.status = a
        rec.updated_at = _now()
        if a == ApprovalStatus.approved:
            event = db.save_handoff(AGENT, AgentName.marketing.value, "document", document_id)
            rec.metadata.update(approved_by=action.reviewer, approved_at=rec.updated_at.isoformat(),
                                handoff_event=event["event_id"])
        rec = self._save(rec, row["request"], f"review: {a.value}", action.reviewer)
        db.log_event("INFO", "document_review", agent=AGENT,
                     detail=f"{document_id} v{rec.version}: {a.value} by {action.reviewer}: {action.comment or ''}")
        if a == ApprovalStatus.approved and rec.metadata.get("pending_email"):
            rec = self.email(document_id, DocumentEmail(to=rec.metadata["pending_email"], sender=action.reviewer))
        return rec

    def edit(self, document_id: str, edit: DocumentEdit) -> DocumentRecord:
        """Apply a reviewer's changes: new version in draft; approval (if any) is cleared."""
        row = self._row(document_id)
        old = DocumentRecord(**row["content"])
        sections = copy.deepcopy(old.sections)
        applied, not_applied = [], []
        if edit.remove_sections:
            keep = [s for s in sections if not any(s.key == k or s.key.startswith(f"{k}:")
                                                   for k in edit.remove_sections)]
            if len(keep) != len(sections):
                applied.append("removed: " + ", ".join(edit.remove_sections))
            else:
                not_applied.append("no section with key " + ", ".join(edit.remove_sections))
            sections = keep
        for new in edit.sections:
            idx = next((i for i, s in enumerate(sections) if s.key == new.key), None)
            if idx is None:
                sections.append(new)
                applied.append(f"added section {new.key}")
            elif sections[idx] != new:
                new.locked = sections[idx].locked                # a reviewer may edit locked facts; the flag stays
                sections[idx] = new
                applied.append(f"edited section {new.key}")
        title = edit.title.strip() if edit.title and edit.title.strip() else old.title
        if title != old.title:
            applied.append("title")
        if edit.instructions:
            returned = llm.revise(old.doc_type.value, sections, edit.instructions, old.language.value) \
                if llm.enabled() else None
            merged, changed = llm.merge_sections(sections, returned)
            if changed:
                sections = merged
                applied.append(f"LLM applied to {', '.join(changed)}: {edit.instructions[:200]}")
            else:
                not_applied.append(edit.instructions if llm.enabled() else
                                   f"{edit.instructions} (needs the LLM - edit the sections directly instead)")
        sections, removed = _guard(sections)
        note = edit.instructions or "; ".join(applied) or "edit"
        db.save_review("document", document_id, ApprovalStatus.changes_requested.value, edit.reviewer, note[:500])
        rec = old.model_copy(deep=True)
        rec.sections, rec.title = sections, title
        rec.reviewer_comments = old.reviewer_comments + [f"{edit.reviewer} (changes_requested): {note}"]
        rec.metadata["last_edit"] = {"by": edit.reviewer, "applied": applied, "not_applied": not_applied,
                                     "claims_removed": removed, "at": _now().isoformat()}
        rec.updated_at = _now()
        if sections == old.sections and title == old.title:
            rec.status = ApprovalStatus.changes_requested
            return self._save(rec, row["request"], "edit: nothing applied", edit.reviewer)
        rec.version = old.version + 1
        rec.status = ApprovalStatus.draft
        for k in ("approved_by", "approved_at", "handoff_event"):
            rec.metadata.pop(k, None)
        rec = self._save(rec, row["request"], "; ".join(applied)[:500], edit.reviewer)
        db.log_event("INFO", "document_edited", agent=AGENT,
                     detail=f"{document_id} v{rec.version} by {edit.reviewer}: {len(applied)} applied, "
                            f"{len(not_applied)} not applied")
        return rec

    # ================================================================== distribution
    def email(self, document_id: str, req: DocumentEmail) -> DocumentRecord:
        """Only approved documents leave the academy (brief: manual approval before distribution)."""
        row = self._row(document_id)
        rec = DocumentRecord(**row["content"])
        if rec.status != ApprovalStatus.approved:
            raise ValueError(f"Only approved documents can be e-mailed; {document_id} is {rec.status.value}")
        to = [str(x) for x in req.to]
        body = (req.message or f"Please find attached: {rec.title} (version {rec.version}).") + \
            "\n\n-- Sent by REX AI on behalf of the academy"
        result = send_document(to, rec.title, body, self.pdf(document_id))
        result.update(by=req.sender, at=_now().isoformat())
        rec.metadata.setdefault("emails", []).append(result)
        if result["sent"]:
            rec.emailed_to = list(dict.fromkeys(rec.emailed_to + to))
        rec.metadata["pending_email"] = []
        db.save_document(rec.model_dump(mode="json"), row["request"], change_note=f"e-mail ({result['channel']})",
                         changed_by=req.sender)
        db.log_event("INFO", "document_email", agent=AGENT,
                     detail=f"{document_id} -> {', '.join(to)} via {result['channel']} by {req.sender}")
        return rec

    # ================================================================== handoffs (Curriculum -> Document)
    def pending_handoffs(self) -> list[dict]:
        """Approved curricula handed to this agent, with the documents already made from them."""
        out, seen = [], set()
        for ev in db.list_handoffs_to(AGENT):
            cid = ev["payload_id"]
            if ev["payload_type"] != "curriculum" or cid in seen:
                continue
            seen.add(cid)
            row = db.get_curriculum(cid)
            if not row:
                continue
            docs = db.list_documents(curriculum_id=cid)
            out.append({"event_id": ev["event_id"], "curriculum_id": cid, "course_title": row["course_title"],
                        "curriculum_status": row["content"].get("status"), "version": row["content"].get("version"),
                        "handed_over_at": ev["created_at"], "documents": len(docs),
                        "doc_types": sorted({d["doc_type"] for d in docs})})
        return out
