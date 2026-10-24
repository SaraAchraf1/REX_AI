"""Agent 2 - Educational RFP & Tender Research Agent (owner: Member 2).

    collect  ->  country filter  ->  relevance  ->  expiry / requested filters  ->  duplicates
             ->  qualification  ->  summary  ->  store (rfps table)  ->  RAG index  ->  alerts

Strict rule: only Egypt (EG) and Saudi Arabia (SA). A notice whose country is missing, unclear or anything
else is rejected and listed in RFPRunResult.rejections with the reason. Egypt matches are flagged for review
and alerted immediately; Saudi matches go to the monitored pipeline.

    from rex_ai.agents.rfp.agent import RFPResearchAgent
    result = RFPResearchAgent().run(RFPRunRequest())
"""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date, datetime, timezone
from typing import Optional

from rex_ai import config
from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.agents.rfp import llm
from rex_ai.agents.rfp.dedupe import dedupe_hash, is_duplicate
from rex_ai.agents.rfp.extract import RawOpportunity, classify, detect_country, matched_terms, parse_date
from rex_ai.agents.rfp.notify import COUNTRY_NAMES, notify
from rex_ai.agents.rfp.qualify import Qualifier
from rex_ai.agents.rfp.sources import Source, build_sources, load_registry, rag_or_none
from rex_ai.core import db
from rex_ai.core.schemas import (AgentName, Country, RFPAskRequest, RFPAskResponse, RFPRecord, RFPRunRequest,
                                 RFPRunResult, RFPStatus, SourceRef)

log = logging.getLogger("rex.rfp")

AGENT = AgentName.rfp_research.value
REVIEW_STATUS = {"EG": RFPStatus.flagged_review, "SA": RFPStatus.monitoring}
EXPORT_COLUMNS = ["rfp_id", "title", "issuing_organization", "country", "training_category", "deadline",
                  "publication_date", "status", "qualification", "priority", "relevance_score", "matched_course_ids",
                  "eligibility_requirements", "summary", "qualification_notes", "submission_link",
                  "available_documents", "source_id", "source_url", "first_seen", "last_checked"]


def _today() -> date:
    return date.fromisoformat(config.RFP_TODAY) if config.RFP_TODAY else datetime.now(timezone.utc).date()


def row_to_record(row: dict) -> RFPRecord:
    return RFPRecord(
        **{k: row.get(k) for k in ("rfp_id", "title", "issuing_organization", "country", "deadline",
                                   "eligibility_requirements", "training_category", "submission_link",
                                   "publication_date", "available_documents", "summary", "qualification",
                                   "qualification_notes", "status", "dedupe_hash", "last_checked",
                                   "matched_course_ids", "priority", "first_seen")},
        relevance_score=row.get("relevance_score") or 0.0,
        source=SourceRef(source_id=row["source_id"], title=row.get("source_title") or row["source_id"],
                         url=row.get("source_url"), record_id=row["rfp_id"]))


class RFPResearchAgent:
    def __init__(self, sources: Optional[list[Source]] = None, qualifier: Optional[Qualifier] = None,
                 today: Optional[date] = None, use_rag: bool = True):
        self.sources = sources
        self._qualifier = qualifier
        self.today = today or _today()
        self.use_rag = use_rag
        self.registry = load_registry()

    @property
    def qualifier(self) -> Qualifier:
        if self._qualifier is None:
            self._qualifier = Qualifier(use_rag=self.use_rag)
        return self._qualifier

    @property
    def rag(self):
        return rag_or_none() if self.use_rag else None

    # ================================================================== run
    def run(self, req: Optional[RFPRunRequest] = None) -> RFPRunResult:
        req = req or RFPRunRequest()
        job_id = db.start_job(AGENT, req.model_dump(mode="json"))
        try:
            result = self._run(req, job_id)
        except Exception as exc:
            db.finish_job(job_id, "failed", error=repr(exc))
            db.log_event("ERROR", "rfp_run_failed", agent=AGENT, detail=repr(exc))
            raise
        db.finish_job(job_id, "succeeded", output_ref=",".join(r.rfp_id for r in result.records)[:2000])
        db.log_event("INFO", "rfp_run", agent=AGENT,
                     detail=f"{job_id}: found={result.found} kept={result.kept} new_eg={len(result.new_flagged_egypt)}")
        return result

    def _collect(self, req: RFPRunRequest) -> tuple[list[RawOpportunity], list[dict]]:
        sources = self.sources if self.sources is not None else build_sources(
            req.source_ids, req.include_live_sources, self.registry)
        raws, errors = [], []
        for s in sources:
            try:
                items = s.fetch()
            except Exception as exc:                       # one broken portal never stops the run
                log.warning("Source %s failed: %s", s.source_id, exc)
                errors.append({"source_id": s.source_id, "error": str(exc)})
                continue
            if req.source_ids:
                items = [r for r in items if r.source_id in req.source_ids]
            raws.extend(items)
        return raws, errors

    def _run(self, req: RFPRunRequest, job_id: str) -> RFPRunResult:
        raws, source_errors = self._collect(req)
        db.expire_rfps(self.today.isoformat())
        stats = {"rejected_country": 0, "rejected_irrelevant": 0, "expired": 0, "rejected_filters": 0,
                 "duplicates": 0, "already_known": 0}
        rejections: list[dict] = []

        def reject(kind: str, raw: RawOpportunity, reason: str):
            stats[kind] += 1
            rejections.append({"title": raw.title, "source_id": raw.source_id, "ref": raw.raw_ref,
                               "rejected": kind, "reason": reason})

        # ---- validate each notice
        candidates: list[dict] = []
        for raw in raws:
            country, why = detect_country(raw)
            if country is None:
                reject("rejected_country", raw, why)
                continue
            if country not in req.countries:
                reject("rejected_country", raw, f"{country.value} not requested in this run")
                continue
            cls = classify(raw, req.keywords)
            if not cls.relevant:
                reject("rejected_irrelevant", raw, cls.reason)
                continue
            deadline, approx = parse_date(raw.deadline)
            if deadline and deadline < self.today:
                reject("expired", raw, f"deadline {deadline} has passed")
                continue
            if cls.category not in req.categories:
                reject("rejected_filters", raw, f"category {cls.category.value} not requested")
                continue
            if req.deadline_from or req.deadline_to:
                if not deadline:
                    reject("rejected_filters", raw, "deadline not stated (a deadline range was requested)")
                    continue
                if (req.deadline_from and deadline < req.deadline_from) or \
                        (req.deadline_to and deadline > req.deadline_to):
                    reject("rejected_filters", raw, f"deadline {deadline} outside the requested range")
                    continue
            if req.keywords and not matched_terms(raw.text, req.keywords):
                reject("rejected_filters", raw, f"does not mention: {', '.join(req.keywords)}")
                continue
            cand = self._candidate(raw, country, cls, deadline, approx)
            dup = next((c for c in candidates if c["dedupe_hash"] == cand["dedupe_hash"] or is_duplicate(c, cand)),
                       None)
            if dup:
                self._merge(dup, cand)
                reject("duplicates", raw, f"duplicate of '{dup['title']}' ({dup['source_id']})")
                continue
            candidates.append(cand)

        # ---- compare with what is already stored, then assess and save
        stored = db.list_rfps(limit=10000)
        records: list[RFPRecord] = []
        new: list[RFPRecord] = []
        for cand in candidates:
            known = db.find_rfp_by_hash(cand["dedupe_hash"]) or next(
                (r for r in stored if is_duplicate(r, cand)), None)
            if known:
                stats["already_known" if known["dedupe_hash"] == cand["dedupe_hash"] else "duplicates"] += 1
            rec = self._build(cand, known)
            db.save_rfp(rec.model_dump(mode="json"))
            records.append(row_to_record(db.get_rfp(rec.rfp_id)))
            if not known:
                new.append(rec)

        self._index(records)
        alerts = notify([r.model_dump(mode="json") for r in new]) if req.notify and new else []
        records.sort(key=lambda r: self.rank(r, req.priority_organizations))
        return RFPRunResult(
            job_id=job_id, found=len(raws), kept=len(records), records=records, rejections=rejections,
            new_flagged_egypt=[r.rfp_id for r in new if r.country == Country.EG],
            new_monitoring_saudi=[r.rfp_id for r in new if r.country == Country.SA],
            alerts_sent=len(alerts), source_errors=source_errors, **stats)

    # ================================================================== steps
    def _candidate(self, raw: RawOpportunity, country: Country, cls, deadline, approx) -> dict:
        organization = (raw.organization or "").strip() or "Not stated"
        source = self.registry.get(raw.source_id, {})
        return {
            "title": " ".join(raw.title.split()), "issuing_organization": organization, "country": country.value,
            "deadline": deadline.isoformat() if deadline else None, "deadline_approx": approx,
            "publication_date": (d.isoformat() if (d := parse_date(raw.publication_date)[0]) else None),
            "eligibility_requirements": raw.eligibility, "training_category": cls.category.value,
            "submission_link": raw.submission_link, "available_documents": list(raw.documents),
            "relevance_score": cls.score, "source_id": raw.source_id, "source_title": source.get("name"),
            "source_url": raw.source_url or source.get("url"), "text": raw.text, "topic": raw.topic, "also_seen": [],
            "dedupe_hash": dedupe_hash(raw.title, organization, deadline),
        }

    @staticmethod
    def _merge(keep: dict, other: dict) -> None:
        """Same tender from two sources: keep the first, fill its gaps from the second."""
        for k in ("deadline", "publication_date", "eligibility_requirements", "submission_link"):
            if not keep.get(k) and other.get(k):
                keep[k] = other[k]
        if keep["issuing_organization"] == "Not stated":
            keep["issuing_organization"] = other["issuing_organization"]
        keep["available_documents"] = list(dict.fromkeys(keep["available_documents"] + other["available_documents"]))
        keep["text"] += "\n" + other["text"]
        keep["topic"] += "\n" + other["topic"]
        keep["relevance_score"] = max(keep["relevance_score"], other["relevance_score"])
        keep["also_seen"].append(other["source_id"] + (f" ({other['source_url']})" if other.get("source_url") else ""))

    def _build(self, cand: dict, known: Optional[dict]) -> RFPRecord:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        a = self.qualifier.assess(cand["text"], cand["eligibility_requirements"], cand["country"], cand["topic"])
        notes = list(a.notes)
        if cand["deadline_approx"]:
            notes.append("Deadline converted from the Hijri calendar - confirm the exact date on the portal")
        if cand["also_seen"]:
            notes.append("Also published in: " + "; ".join(dict.fromkeys(cand["also_seen"])))
        country = cand["country"]
        fields = {k: cand[k] for k in ("title", "issuing_organization", "deadline", "publication_date",
                                       "eligibility_requirements", "training_category", "submission_link",
                                       "available_documents", "relevance_score", "dedupe_hash")}
        if known:                                       # keep the stored identity and the reviewer's decisions
            fields["available_documents"] = list(dict.fromkeys(known["available_documents"]
                                                               + fields["available_documents"]))
            for k in ("deadline", "eligibility_requirements", "submission_link"):
                fields[k] = fields[k] or known.get(k)
            fields.update(title=known["title"], issuing_organization=known["issuing_organization"],
                          dedupe_hash=known["dedupe_hash"])
        rec = RFPRecord(
            **fields, country=country,
            rfp_id=known["rfp_id"] if known else f"RFP-{country}-{cand['dedupe_hash'][:8].upper()}",
            qualification=a.fit, qualification_notes="\n".join(notes), matched_course_ids=a.course_ids,
            status=known["status"] if known else REVIEW_STATUS[country],
            priority="high" if country == "EG" else "normal",
            source=SourceRef(source_id=known["source_id"] if known else cand["source_id"],
                             title=(known or {}).get("source_title") or cand["source_title"] or cand["source_id"],
                             url=(known or {}).get("source_url") or cand["source_url"]),
            last_checked=now, first_seen=(known or {}).get("first_seen") or now,
            summary=(known or {}).get("summary"))
        if not rec.summary:
            rec.summary = self.summarize(rec, a.course_ids)
        return rec

    def summarize(self, rec: RFPRecord, course_ids: list[str]) -> str:
        """LLM summary grounded in the tender's own documents (RAG) when available, else a fixed template."""
        rag = self.rag
        if llm.enabled() and rag is not None:
            where = {"origin": "rfp_document"}
            if rec.available_documents:
                where["source"] = rec.available_documents
            hits = rag.search(rec.title, k=4, where=where)
            text = llm.summarize(rec.model_dump(mode="json", exclude={"summary", "source"}), rag.build_context(hits))
            if text:
                return text
        parts = [f"{rec.issuing_organization} ({COUNTRY_NAMES[rec.country.value]}) is tendering "
                 f"{rec.training_category.value.lower()}: {rec.title}."]
        if rec.deadline:
            days = (rec.deadline - self.today).days
            parts.append(f"Bids close on {rec.deadline} ({days} day{'s' if days != 1 else ''} left).")
        else:
            parts.append("The deadline is not stated in the notice.")
        if rec.eligibility_requirements:
            first = re.split(r"\n|; ", rec.eligibility_requirements.lstrip("- "))[0].strip()
            parts.append(f"Key requirement: {first.rstrip('.')}.")
        parts.append(f"Academy fit: {rec.qualification.value}"
                     + (f" (courses {', '.join(course_ids)})." if course_ids else "."))
        return " ".join(parts)

    def _index(self, records: list[RFPRecord]) -> None:
        """Index every kept record in the RAG collection so POST /rfp/ask can search them semantically."""
        rag = self.rag
        if rag is None or not records:
            return
        try:
            rag.add_texts(
                [f"{r.title}\n{r.issuing_organization} - {COUNTRY_NAMES[r.country.value]}\n"
                 f"Category: {r.training_category.value}\nDeadline: {r.deadline or 'not stated'}\n"
                 f"Eligibility: {r.eligibility_requirements or 'not stated'}\nSummary: {r.summary}" for r in records],
                [{"origin": "rfp_record", "rfp_id": r.rfp_id, "country": r.country.value, "status": r.status.value,
                  "category": r.training_category.value, "source_id": r.source.source_id, "title": r.title,
                  "source": r.rfp_id, "url": r.submission_link} for r in records],
                ids=[f"rec:{r.rfp_id}" for r in records])
        except Exception as exc:
            log.warning("Could not index RFP records in RAG: %s", exc)

    @staticmethod
    def is_priority(r: RFPRecord, priority_orgs: list[str]) -> bool:
        return bool(priority_orgs) and contains_any(f"{r.issuing_organization} {r.title}", priority_orgs)

    def rank(self, r: RFPRecord, priority_orgs: list[str]):
        """Priority organisations first, then Egypt (high priority), then the nearest deadline."""
        return (not self.is_priority(r, priority_orgs), r.priority != "high", r.deadline or date.max)

    # ================================================================== dashboard helpers
    def filter_stored(self, records: list[RFPRecord], req: RFPRunRequest) -> list[RFPRecord]:
        """Apply a run's filters (countries, categories, deadline range, keywords) to stored records,
        ranked the same way as a run's result."""
        def keep(r: RFPRecord) -> bool:
            if r.country not in req.countries or r.training_category not in req.categories:
                return False
            if req.deadline_from or req.deadline_to:
                if not r.deadline or (req.deadline_from and r.deadline < req.deadline_from) or \
                        (req.deadline_to and r.deadline > req.deadline_to):
                    return False
            text = f"{r.title} {r.issuing_organization} {r.eligibility_requirements or ''} {r.summary or ''}"
            return not req.keywords or bool(matched_terms(text, req.keywords))
        return sorted((r for r in records if keep(r)), key=lambda r: self.rank(r, req.priority_organizations))

    def search(self, country: Optional[str] = None, status: Optional[str] = None, category: Optional[str] = None,
               qualification: Optional[str] = None, q: Optional[str] = None,
               open_only: bool = False) -> list[RFPRecord]:
        rows = db.list_rfps(country=country, status=status, category=category, qualification=qualification, q=q,
                            open_only=open_only, today=self.today.isoformat())
        return [row_to_record(r) for r in rows]

    @staticmethod
    def export_csv(records: list[RFPRecord]) -> str:
        """CSV with a UTF-8 BOM so Excel and Google Sheets show Arabic correctly."""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = r.model_dump(mode="json")
            row.update(source_id=r.source.source_id, source_url=r.source.url,
                       matched_course_ids=", ".join(r.matched_course_ids),
                       available_documents="; ".join(r.available_documents))
            w.writerow(row)
        return "\ufeff" + buf.getvalue()

    # ================================================================== Q&A over stored tenders (RAG)
    def ask(self, req: RFPAskRequest) -> RFPAskResponse:
        rag = self.rag
        if rag is None:
            return self._ask_without_rag(req)
        # tender records first; passages from the tender documents as supporting evidence
        where = {"origin": "rfp_record", **({"country": req.country.value} if req.country else {})}
        hits = rag.search(req.question, k=req.k, where=where)
        hits += rag.search(req.question, k=2, where={"origin": "rfp_document"}, mmr=True)
        if not hits:
            return RFPAskResponse(answer="No stored tender matches that question yet. Run the RFP research first.")
        rfp_ids = list(dict.fromkeys(h.metadata["rfp_id"] for h in hits if h.metadata.get("rfp_id")))
        sources = rag.to_source_refs(hits)
        text = llm.answer(req.question, rag.build_context(hits))
        if text:
            return RFPAskResponse(answer=text, rfp_ids=rfp_ids, sources=sources, llm_used=True)
        tenders, passages = [], []
        for n, h in enumerate(hits, start=1):
            if h.metadata.get("origin") == "rfp_record":
                if row := db.get_rfp(h.metadata["rfp_id"]):
                    tenders.append(f"{len(tenders) + 1}. {row['title']} - {row['issuing_organization']} "
                                   f"({COUNTRY_NAMES[row['country']]}), deadline {row['deadline'] or 'not stated'}, "
                                   f"status {row['status']}, fit {row['qualification']} [{n}]")
            else:
                passages.append(f"- {h.citation}: {' '.join(h.text.split()[-40:])} [{n}]")
        answer = "Most relevant tenders:\n" + "\n".join(tenders or ["(none)"])
        if passages:
            answer += "\n\nFrom the tender documents:\n" + "\n".join(passages)
        return RFPAskResponse(answer=answer, rfp_ids=rfp_ids, sources=sources)

    def _ask_without_rag(self, req: RFPAskRequest) -> RFPAskResponse:
        words = {w for w in normalize(req.question).split() if len(w) > 3}
        rows = db.list_rfps(country=req.country.value if req.country else None)
        scored = sorted(((len(words & set(normalize(f"{r['title']} {r.get('summary') or ''}").split())), r)
                         for r in rows), key=lambda x: -x[0])
        top = [r for s, r in scored if s][: req.k]
        if not top:
            return RFPAskResponse(answer="No stored tender matches that question yet.")
        return RFPAskResponse(
            answer="Most relevant tenders:\n" + "\n".join(f"- {r['title']} ({r['country']}, {r['deadline']})"
                                                          for r in top),
            rfp_ids=[r["rfp_id"] for r in top],
            sources=[SourceRef(source_id=r["source_id"], title=r["title"], record_id=r["rfp_id"]) for r in top])
