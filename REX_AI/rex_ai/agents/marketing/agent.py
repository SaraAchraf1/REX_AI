"""Agent 5 - AI Marketing Agent (owner: Member 1).

    request (course, curriculum / document, audience, channels, objective, offer, CTA, budget, variations, language)
        ->  approved facts: catalogue, approved curriculum (Agent 3), approved document (Agent 4), fees, offers,
            cohorts, instructors; FAQs via RAG (academy_kb); consented-lead insights
        ->  audience persona (pains, motivations, tone, preferred channels)
        ->  rule-based copy per channel x variation (different angles)  ->  fact / claim / length checks
        ->  strategy: channel mix, budget split, dated timeline, KPIs, A/B tests (channel playbook via RAG)
        ->  optional LLM rewrite in house style (brand guide + examples via RAG marketing_refs), checked again
        ->  stored as version 1, status pending_approval, campaign brief PDF
    edit: reviewer edits / removes variants, changes the strategy or gives instructions -> version n+1
    review: approved (only if the curriculum is approved and unchanged and the course facts are still current)
            / changes_requested / rejected
    publish: approved campaigns only -> distribution package + e-mail variant to consented recipients

Example workflow (brief): New Course -> Curriculum Approved -> Marketing Agent Generates Campaign -> Admin Reviews ->
Content Approved for Distribution.

    from rex_ai.agents.marketing.agent import MarketingAgent
    camp = MarketingAgent().generate(CampaignRequest(course_id="DS-101", channels=["linkedin_post", "email"]))
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rex_ai import config
from rex_ai.agents.marketing import audience as aud
from rex_ai.agents.marketing import checks, llm, rules, strategy
from rex_ai.agents.marketing.facts import NotFound, collect
from rex_ai.agents.marketing.pdf import render_campaign_pdf
from rex_ai.agents.marketing.publish import send_email, write_package
from rex_ai.agents.marketing.writer import CopyWriter
from rex_ai.core import db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, Campaign, CampaignEdit, CampaignPublish, CampaignRequest,
                                 ChannelInfo, ContentVariant, ReviewAction, SourceRef)

log = logging.getLogger("rex.marketing")

AGENT = AgentName.marketing.value


def _quoted(snap: dict) -> dict:
    """The facts the copy quotes, which must still be true when the campaign is approved or published (seats left
    change every day and are not compared)."""
    promo, start = snap.get("promotion") or {}, snap.get("next_start") or {}
    return {"title": snap.get("title"), "hours": snap.get("hours"), "fee": snap.get("fee"),
            "best_price": snap.get("best_price"),
            "promotion": [promo.get(k) for k in ("promo_id", "discount_pct", "deadline")] if promo else None,
            "next_start": [start.get(k) for k in ("start_date", "format", "days", "time")] if start else None}


__all__ = ["MarketingAgent", "NotFound"]


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


class MarketingAgent:
    def __init__(self, use_rag: bool = True):
        from rex_ai.agents.rex.knowledge_base import get_kb
        self.kb = get_kb()
        self.use_rag = use_rag
        self.retriever = None
        if use_rag:
            try:                                    # same RAG index (academy_kb) the REX Assistant searches
                from rex_ai.core import data_sync
                self.retriever = data_sync.get_assistant().retriever
            except Exception as exc:
                log.warning("Course retriever unavailable, FAQs chosen by category: %s", exc)

    @staticmethod
    def channels() -> list[ChannelInfo]:
        return rules.channel_infos()

    # ================================================================== generate
    def generate(self, req: CampaignRequest) -> Campaign:
        job_id = db.start_job(AGENT, req.model_dump(mode="json"))
        try:
            camp = self._generate(req)
        except Exception as exc:
            db.finish_job(job_id, "failed", error=repr(exc))
            db.log_event("ERROR", "campaign_generate_failed", agent=AGENT, detail=repr(exc))
            raise
        db.finish_job(job_id, "succeeded", output_ref=camp.campaign_id)
        db.log_event("INFO", "campaign_generated", agent=AGENT,
                     detail=f"{job_id}: {camp.campaign_id} {camp.course_id} {len(camp.variants)} variants by "
                            f"{camp.metadata.get('generated_by')}")
        return camp

    def _generate(self, req: CampaignRequest) -> Campaign:
        lang = req.language.value
        facts = collect(self.kb, req.course_id, req.curriculum_id, req.document_id, lang, self.retriever)
        audience = aud.resolve(req.target_audience, facts.course.target_audience, facts.lead_insights, lang)
        facts.choose_promotion(audience["persona_id"])
        objective, ocfg = rules.objective(req.objective)
        writer = CopyWriter(facts, audience, objective, ocfg, req.offer, req.cta)
        channels = [c.value for c in req.channels]
        snapshot = {**facts.snapshot(), "requested_offer": req.offer, "requested_cta": req.cta}
        allowed = checks.allowed_numbers(snapshot)
        variants = [checks.check_variant(writer.variant(ch, n), allowed, lang)[0]
                    for ch in channels for n in range(1, req.variations_per_channel + 1)]
        context, guide_refs, tips = rules.guidance(channels, f"{facts.course.title_en} {audience['persona_en']}",
                                                   lang) if self.use_rag else ("", [], {})
        plan, summary = strategy.build(facts, audience, writer, objective, ocfg, channels,
                                       req.variations_per_channel, req.budget, tips)
        generated_by, rejected = "rules", []
        if llm.enabled():
            persona = {k: audience[k] for k in ("persona", "pains", "motivations", "tone", "text")}
            returned = llm.refine(variants, snapshot, persona, summary, context, lang, req.instructions)
            variants, changed, rejected = llm.merge(variants, returned, allowed, lang)
            if changed:
                generated_by = "llm" if len(changed) == len(variants) else "llm+rules"
        now = _now()
        cid = "CMP-" + uuid.uuid4().hex[:8].upper()
        handoff = next((h for h in db.list_handoffs(req.document_id) if h["to_agent"] == AGENT), None) \
            if req.document_id else None
        camp = Campaign(
            campaign_id=cid, course_id=facts.course.course_id, strategy=summary, variants=variants,
            course_facts_used=snapshot, status=ApprovalStatus.pending_approval,
            sources=_dedupe([*facts.sources, *rules.rule_sources(), *guide_refs]), created_at=now, updated_at=now,
            title=f"{facts.course.title_en}: {objective} campaign", curriculum_id=facts.curriculum_id,
            document_id=req.document_id, language=req.language, objective=objective,
            target_audience=audience["text"], channels=req.channels, headlines=writer.headline_options(),
            ctas=writer.cta_options(), strategy_plan=plan, version=1, download_url=f"/campaigns/{cid}/pdf",
            metadata={"generated_by": generated_by, "notes": facts.notes, "llm_rejected": rejected,
                      "instructions": req.instructions, "persona": audience["persona_id"],
                      "rag": {"marketing_refs": len(guide_refs), "faqs": [o["faq_id"] for o in facts.objections]},
                      "handoff_event": handoff and handoff["event_id"]})
        return self._save(camp, req.model_dump(mode="json"), f"generated ({generated_by})", AGENT)

    # ================================================================== storage / export
    def _pdf_path(self, c: Campaign) -> Path:
        suffix = "" if c.status == ApprovalStatus.approved else f"_{c.status.value}"
        return config.MARKETING_OUTPUT_DIR / f"{c.campaign_id}_v{c.version}_brief{suffix}.pdf"

    def _save(self, c: Campaign, request: dict, change_note: str, changed_by: str) -> Campaign:
        path = self._pdf_path(c)
        try:
            render_campaign_pdf(c, path, reviews=db.list_reviews("campaign", c.campaign_id))
            c.metadata["file_path"] = str(path)
        except Exception as exc:                      # the brief is a convenience: never lose the campaign over it
            log.warning("Campaign brief PDF failed for %s: %s", c.campaign_id, exc)
            c.metadata["file_path"] = None
        db.save_campaign(c.model_dump(mode="json"), request, change_note=change_note, changed_by=changed_by)
        return c

    def _row(self, campaign_id: str) -> dict:
        row = db.get_campaign(campaign_id)
        if not row or not row.get("content"):
            raise NotFound(f"Campaign {campaign_id} not found")
        return row

    def get(self, campaign_id: str) -> Campaign:
        return Campaign(**self._row(campaign_id)["content"])

    def search(self, **filters) -> list[Campaign]:
        return [Campaign(**r["content"]) for r in db.list_campaigns(**filters)]

    def pdf(self, campaign_id: str) -> Path:
        """The campaign brief PDF of the current version (rebuilt if the file was deleted)."""
        c = self.get(campaign_id)
        path = Path(c.metadata.get("file_path") or self._pdf_path(c))
        if not path.exists():
            c = self._save(c, self._row(campaign_id)["request"], "pdf rebuilt", AGENT)
            if not c.metadata.get("file_path"):
                raise RuntimeError("The campaign brief PDF could not be created")
            path = Path(c.metadata["file_path"])
        return path

    def history(self, campaign_id: str) -> dict:
        c = self.get(campaign_id)
        return {
            "versions": [{k: v for k, v in r.items() if k != "content"} | {"status": r["content"].get("status")}
                         for r in db.list_campaign_versions(campaign_id)],
            "reviews": db.list_reviews("campaign", campaign_id),
            "publications": c.metadata.get("publications") or [],
            "handoffs": db.list_handoffs(c.document_id) if c.document_id else [],
        }

    # ================================================================== review and edit
    def source_check(self, c: Campaign) -> list[str]:
        """Why the campaign cannot be approved / published now (empty = fine): its curriculum and document must be
        approved and unchanged, and the course facts the copy quotes must still be current."""
        f, problems = c.course_facts_used, []
        if cid := f.get("curriculum_id"):
            row = db.get_curriculum(cid)
            if not row:
                problems.append(f"its curriculum {cid} no longer exists")
            else:
                status, version = row["content"].get("status"), row["content"].get("version")
                if status != ApprovalStatus.approved.value:
                    problems.append(f"its curriculum {cid} is {status}: approve the curriculum first")
                elif version != f.get("curriculum_version"):
                    problems.append(f"its curriculum {cid} changed (v{f.get('curriculum_version')} -> v{version}): "
                                    "regenerate the campaign")
        if did := f.get("document_id"):
            doc = db.get_document(did)
            if not doc or doc["status"] != ApprovalStatus.approved.value:
                problems.append(f"its document {did} is not approved")
        try:
            current = collect(self.kb, c.course_id, f.get("curriculum_id"), None, c.language.value)
        except (NotFound, ValueError) as exc:
            return problems + [str(exc)]
        current.choose_promotion(c.metadata.get("persona"))
        now = current.snapshot()
        before, after = _quoted(f), _quoted(now)
        changed = [k for k in before if before[k] != after[k]]
        if changed:
            problems.append("course facts changed since the copy was written (" + ", ".join(changed) +
                            "): regenerate the campaign")
        return problems

    def review(self, campaign_id: str, action: ReviewAction) -> Campaign:
        row = self._row(campaign_id)
        c = Campaign(**row["content"])
        a = action.action
        if a == ApprovalStatus.draft:
            raise ValueError("A review action is pending_approval, approved, changes_requested or rejected")
        if c.status == ApprovalStatus.approved and a != ApprovalStatus.rejected:
            raise ValueError("This campaign is already approved; edit it to make a new version")
        if a == ApprovalStatus.approved and (problems := self.source_check(c)):
            raise ValueError(f"Cannot approve {campaign_id}: " + "; ".join(problems) + ".")
        if a != ApprovalStatus.pending_approval:
            db.save_review("campaign", campaign_id, a.value, action.reviewer, action.comment)
        if action.comment or a != ApprovalStatus.pending_approval:
            c.reviewer_comments.append(f"{action.reviewer} ({a.value})" + (f": {action.comment}"
                                                                           if action.comment else ""))
        c.status = a
        c.updated_at = _now()
        if a == ApprovalStatus.approved:
            c.metadata.update(approved_by=action.reviewer, approved_at=c.updated_at.isoformat())
        c = self._save(c, row["request"], f"review: {a.value}", action.reviewer)
        db.log_event("INFO", "campaign_review", agent=AGENT,
                     detail=f"{campaign_id} v{c.version}: {a.value} by {action.reviewer}: {action.comment or ''}")
        return c

    def edit(self, campaign_id: str, edit: CampaignEdit) -> Campaign:
        """Apply a reviewer's changes: new version, back to pending_approval; any approval is cleared."""
        row = self._row(campaign_id)
        old = Campaign(**row["content"])
        lang = old.language.value
        allowed = checks.allowed_numbers(old.course_facts_used)
        variants = [v.model_copy(deep=True) for v in old.variants]
        applied, not_applied = [], []
        if edit.remove_variants:
            keep = [v for v in variants if v.key not in edit.remove_variants and v.channel.value not in
                    edit.remove_variants]
            if len(keep) != len(variants):
                applied.append("removed: " + ", ".join(edit.remove_variants))
            else:
                not_applied.append("no variant " + ", ".join(edit.remove_variants))
            variants = keep
        for new in edit.variants:
            new = new.model_copy(update={"generated_by": f"reviewer ({edit.reviewer})"})
            new, invented = checks.check_variant(new, allowed, lang)   # a reviewer's numbers are flagged, not refused
            idx = next((i for i, v in enumerate(variants) if v.key == new.key), None)
            if idx is None:
                variants.append(new)
                applied.append(f"added {new.key}")
            elif (variants[idx].headline, variants[idx].body, variants[idx].cta) != (new.headline, new.body, new.cta):
                new.angle = new.angle or variants[idx].angle
                variants[idx] = new
                applied.append(f"edited {new.key}" + (f" (check numbers {', '.join(invented)})" if invented else ""))
        strategy_text = edit.strategy.strip() if edit.strategy and edit.strategy.strip() else old.strategy
        if strategy_text != old.strategy:
            applied.append("strategy")
        if edit.instructions:
            returned = llm.revise(variants, old.course_facts_used, edit.instructions, lang) if llm.enabled() else None
            merged, changed, rejected = llm.merge(variants, returned, allowed, lang)
            if changed:
                variants = merged
                applied.append(f"LLM applied to {', '.join(changed)}: {edit.instructions[:200]}")
            not_applied += rejected
            if not changed:
                not_applied.append(edit.instructions if llm.enabled() else
                                   f"{edit.instructions} (needs the LLM - edit the variants directly instead)")
        variants.sort(key=lambda v: ([c.value for c in old.channels].index(v.channel.value)
                                     if v.channel in old.channels else 99, v.variant_no))
        note = edit.instructions or "; ".join(applied) or "edit"
        db.save_review("campaign", campaign_id, ApprovalStatus.changes_requested.value, edit.reviewer, note[:500])
        c = old.model_copy(deep=True)
        c.variants, c.strategy = variants, strategy_text
        c.reviewer_comments = old.reviewer_comments + [f"{edit.reviewer} (changes_requested): {note}"]
        c.metadata["last_edit"] = {"by": edit.reviewer, "applied": applied, "not_applied": not_applied,
                                   "at": _now().isoformat()}
        c.updated_at = _now()
        if not applied:
            c.status = ApprovalStatus.changes_requested
            return self._save(c, row["request"], "edit: nothing applied", edit.reviewer)
        c.version = old.version + 1
        c.status = ApprovalStatus.pending_approval
        for k in ("approved_by", "approved_at"):
            c.metadata.pop(k, None)
        c = self._save(c, row["request"], "; ".join(applied)[:500], edit.reviewer)
        db.log_event("INFO", "campaign_edited", agent=AGENT,
                     detail=f"{campaign_id} v{c.version} by {edit.reviewer}: {len(applied)} applied, "
                            f"{len(not_applied)} not applied")
        return c

    # ================================================================== distribution
    def publish(self, campaign_id: str, req: CampaignPublish) -> Campaign:
        """Only approved campaigns leave the academy, and only while their facts are still current."""
        row = self._row(campaign_id)
        c = Campaign(**row["content"])
        if c.status != ApprovalStatus.approved:
            raise ValueError(f"Only approved campaigns can be published; {campaign_id} is {c.status.value}")
        if problems := self.source_check(c):
            raise ValueError(f"Cannot publish {campaign_id}: " + "; ".join(problems) + ".")
        by_key = {v.key: v for v in c.variants}
        unknown = [k for k in req.variant_keys if k not in by_key]
        if unknown:
            raise LookupError(f"Unknown variants {', '.join(unknown)}. Available: {', '.join(by_key)}")
        chosen = [by_key[k] for k in req.variant_keys] if req.variant_keys else list(c.variants)
        email_result = None
        if req.email_to:
            email_variant = next((v for v in chosen if v.channel.value == "email"), None)
            if email_variant is None:
                raise LookupError("No e-mail variant chosen to send")
            email_result = send_email([str(x) for x in req.email_to], email_variant)
        package = write_package(c, chosen, req.publisher)
        record = {"by": req.publisher, "at": _now().isoformat(), "version": c.version,
                  "variants": [v.key for v in chosen], "folder": package["folder"], "files": package["files"],
                  "email": email_result}
        c.metadata.setdefault("publications", []).append(record)
        c.updated_at = _now()
        db.save_campaign(c.model_dump(mode="json"), row["request"], change_note=f"published by {req.publisher}",
                         changed_by=req.publisher)
        db.log_event("INFO", "campaign_published", agent=AGENT,
                     detail=f"{campaign_id} v{c.version}: {len(chosen)} variants by {req.publisher}"
                            + (f", e-mail via {email_result['channel']}" if email_result else ""))
        return c

    # ================================================================== handoffs (Curriculum / Document -> Marketing)
    def pending_handoffs(self) -> list[dict]:
        """What other agents handed over: approved documents (HandoffEvent document -> marketing) and approved
        curricula of catalogue courses (brief workflow: Curriculum Approved -> Marketing Agent Generates Campaign),
        each with the campaigns already made for it."""
        out, seen = [], set()
        for ev in db.list_handoffs_to(AGENT):
            if ev["payload_type"] != "document" or ev["payload_id"] in seen:
                continue
            seen.add(ev["payload_id"])
            doc = db.get_document(ev["payload_id"])
            if not doc or not doc.get("course_id"):
                continue
            out.append({"kind": "document", "id": doc["document_id"], "course_id": doc["course_id"],
                        "curriculum_id": doc.get("curriculum_id"), "title": doc["title"], "status": doc["status"],
                        "handed_over_at": ev["created_at"],
                        "campaigns": len([r for r in db.list_campaigns(course_id=doc["course_id"])
                                          if r.get("document_id") == doc["document_id"]])})
        for row in db.list_curricula(status="approved"):
            course_id = (row.get("course_id") or "").upper()
            if not course_id or course_id not in self.kb.courses:
                continue
            out.append({"kind": "curriculum", "id": row["curriculum_id"], "course_id": course_id,
                        "curriculum_id": row["curriculum_id"], "title": row["course_title"],
                        "status": row["status"], "version": row["version"], "handed_over_at": row["updated_at"],
                        "campaigns": len(db.list_campaigns(curriculum_id=row["curriculum_id"]))})
        return out
