"""REX AI backend API (FastAPI). Owner: Member 3; agent endpoints co-owned by agent owners.

Run:  uvicorn rex_ai.api.main:app --reload
Docs: http://localhost:8000/docs  (OpenAPI generated from rex_ai/core/schemas.py)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent, NotFound
from rex_ai.agents.documents.agent import DocumentAgent
from rex_ai.agents.marketing.agent import MarketingAgent
from rex_ai.agents.rex.assistant import RexAssistant
from rex_ai.agents.rex.knowledge_base import get_kb
from rex_ai.agents.rex.text_utils import contains_any
from rex_ai.agents.rfp import sources as rfp_sources
from rex_ai.agents.rfp.agent import RFPResearchAgent, row_to_record
from rex_ai.agents.rfp.taxonomy import COUNTRY_PLACES
from rex_ai.core import data_sync, db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, Campaign, CampaignEdit, CampaignPublish, CampaignRequest,
                                 Channel, ChannelInfo, ChatRequest, ChatResponse, Country, Course, Curriculum,
                                 CurriculumImport, CurriculumRequest, CurriculumResearch, CurriculumRevision,
                                 DocumentEdit, DocumentEmail, DocumentPackRequest, DocumentRecord, DocumentRequest,
                                 DocumentType, DocumentTypeInfo, ErrorResponse, FrameworkInfo, GapReport, GapReportRequest, Lead, LeadCreate, RAGHit,
                                 RAGSearchRequest, RAGSearchResponse, ReviewAction, RFPAlert, RFPAskRequest,
                                 RFPAskResponse, RFPCategory, RFPRecord, RFPRunRequest, RFPRunResult, RFPStatus,
                                 RFPStatusUpdate, RouteRequest, RouteResponse, SkillLevel)

log = logging.getLogger("rex.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rex()                                # merges/indexes new files in data/catalog/ and data/documents/
    yield


app = FastAPI(title="REX AI API", version="0.1.0 (Sprint 1)", lifespan=lifespan,
              description="Multi-agent academy assistant. Sprint 1 ships the REX Course Assistant; "
                          "other agents' contracts are defined in rex_ai/core/schemas.py.")


def rex() -> RexAssistant:
    """The assistant on the latest data: rebuilt automatically when a data file is added or changed."""
    return data_sync.get_assistant()


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content=ErrorResponse(
        code="validation_error", message="Invalid input", details={"errors": errors}).model_dump())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("Unhandled error")
    db.log_event("ERROR", "unhandled_exception", detail=f"{request.url.path}: {exc!r}")
    return JSONResponse(status_code=500, content=ErrorResponse(
        code="internal_error", message="Something went wrong. Please try again.").model_dump())


@app.get("/health")
def health() -> dict:
    retriever = rex().retriever
    kb = get_kb()
    return {"status": "ok", "courses": len(kb.courses), "data_status": kb.meta.get("status"),
            "llm_enabled": config.USE_LLM, "retriever": type(retriever).__name__,
            "embedder": getattr(retriever, "backend", "tfidf")}


# ---------------------------------------------------------------- REX (Agent 1)
@app.post("/chat", response_model=ChatResponse, tags=["REX Assistant"])
def chat(req: ChatRequest) -> ChatResponse:
    resp = rex().handle(req)
    db.log_event("INFO", "chat", agent=AgentName.rex_assistant.value, detail=f"{req.session_id}:{resp.intent}")
    return resp


@app.get("/courses", response_model=list[Course], tags=["REX Assistant"])
def list_courses(department: str | None = None, skill_level: str | None = None) -> list[Course]:
    courses = list(rex().kb.courses.values())
    if department:
        courses = [c for c in courses if c.department.lower() == department.lower()]
    if skill_level:
        courses = [c for c in courses if c.skill_level.value.lower() == skill_level.lower()]
    return courses


@app.get("/courses/{course_id}", response_model=Course, tags=["REX Assistant"],
         responses={404: {"model": ErrorResponse}})
def get_course(course_id: str) -> Course:
    course = rex().kb.courses.get(course_id.upper())
    if not course:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    return course


@app.post("/leads", response_model=Lead, status_code=201, tags=["REX Assistant"])
def create_lead(lead: LeadCreate) -> Lead:
    rex()  # make sure the DB is initialised
    saved = db.save_lead(lead.model_dump(mode="json"))
    db.log_event("INFO", "lead_saved", agent=AgentName.rex_assistant.value, detail=saved["lead_id"])
    return Lead(**saved)


# ------------------------------------------------------------ Academy data (add, never overwrite)
@app.post("/data/upload", tags=["Academy data"], responses={400: {"model": ErrorResponse}})
def upload_data(file: UploadFile = File(...)) -> dict:
    """Upload a PDF / Word / Excel / CSV / TXT file. Course catalogues are merged into the course data;
    any other document is indexed so REX can answer from it. Existing files and courses are never removed;
    a file with the same name is saved next to the old one (name-2.pdf)."""
    try:
        result = data_sync.save_upload(file.filename, file.file.read())
    except ValueError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(code="bad_file", message=str(exc)).model_dump())
    db.log_event("INFO", "data_upload", detail=f"{result['file']}:{result['kind']}:{result['status']}")
    return result


@app.post("/data/courses", tags=["Academy data"])
def add_courses(courses: list[dict] = Body(..., embed=True, examples=[[{
        "course_id": "AI-401", "title_en": "MLOps in Production", "department": "AI", "skill_level": "Advanced",
        "duration_hours": 40, "description_en": "Ship and monitor ML models.", "tags": ["mlops", "machine learning"],
        "formats": ["Online live"], "fee": 15000, "schedule": ["2027-01-10 | Online live | Sat, Mon | 19:00-22:00"]}]]),
        status: str | None = Body(None, embed=True)) -> dict:
    """Add or update courses from another system. Keys are field names or catalogue labels
    (see rex_ai/core/catalog_import.py). A new course_id is added, an existing one is updated, nothing is deleted."""
    try:
        result = data_sync.add_courses(courses, status)
    except RuntimeError as exc:                  # merged data failed validation; previous files restored
        return JSONResponse(status_code=400, content=ErrorResponse(code="invalid_courses", message=str(exc)).model_dump())
    db.log_event("INFO", "data_courses", detail=f"added={result['added']} updated={result['updated']}")
    return result


@app.post("/data/reload", tags=["Academy data"])
def reload_data() -> dict:
    """Import everything new in data/catalog/ and data/documents/ now (also happens automatically)."""
    report = data_sync.reload()
    cat, docs = report.get("catalog") or {}, report.get("documents") or {}
    merged = cat.get("merge") or {}
    return {"added": merged.get("added", []), "updated": merged.get("updated", []),
            "catalog_errors": {r.file: r.errors for r in cat.get("files", []) if r.errors},
            "documents_indexed": docs.get("files", 0), "document_errors": docs.get("errors", []),
            **data_sync.status()}


@app.get("/data/status", tags=["Academy data"])
def data_status() -> dict:
    """Courses REX knows and the files it learned from."""
    rex()
    return data_sync.status()


# ------------------------------------------------------------ Shared RAG (all agents)
@app.post("/rag/search", response_model=RAGSearchResponse, tags=["RAG"])
def rag_search(req: RAGSearchRequest) -> RAGSearchResponse:
    """Semantic search over any RAG collection (academy_kb, curriculum_refs, rfp_docs, templates...)."""
    from rex_ai.core.rag import get_rag
    rag = get_rag(req.collection)
    hits = rag.search(req.query, k=req.k, where=req.where, mmr=req.mmr, min_score=req.min_score)
    db.log_event("INFO", "rag_search", detail=f"{req.collection}:{len(hits)} hits")
    return RAGSearchResponse(
        query=req.query, collection=req.collection,
        hits=[RAGHit(chunk_id=h.chunk_id, text=h.text, score=round(h.score, 4), citation=h.citation,
                     metadata=h.metadata) for h in hits],
        context=rag.build_context(hits), sources=rag.to_source_refs(hits))


@app.get("/rag/collections/{collection}", tags=["RAG"])
def rag_stats(collection: str) -> dict:
    """How many documents and chunks a collection holds, and which embedder built it."""
    from rex_ai.core.rag import get_rag
    stats = get_rag(collection).stats()
    return {k: v for k, v in stats.items() if k != "by_document"} | {"documents_list": sorted(stats["by_document"])}


# ------------------------------------------------------------ RFP & Tender Research (Agent 2)
def rfp_agent() -> RFPResearchAgent:
    rex()                                # DB initialised and academy_kb indexed (used for qualification)
    return RFPResearchAgent()


@app.post("/rfp/run", response_model=RFPRunResult, tags=["RFP Research"])
def rfp_run(req: RFPRunRequest = Body(default_factory=RFPRunRequest)) -> RFPRunResult:
    """Collect tenders (feeds, tender documents, optionally live portals), keep only training-related, open,
    non-duplicate notices in Egypt and Saudi Arabia, assess the academy's fit, store them and send alerts."""
    return rfp_agent().run(req)


@app.get("/rfp", response_model=list[RFPRecord], tags=["RFP Research"])
def rfp_list(country: Country | None = None, status: RFPStatus | None = None, category: RFPCategory | None = None,
             qualification: str | None = None, q: str | None = None, open_only: bool = False) -> list[RFPRecord]:
    """Searchable pipeline: Egypt (high priority) first, then by deadline. q searches title, organisation,
    eligibility and summary."""
    return rfp_agent().search(country and country.value, status and status.value, category and category.value,
                              qualification, q, open_only)


@app.get("/rfp/export.csv", tags=["RFP Research"], response_class=Response)
def rfp_export(country: Country | None = None, status: RFPStatus | None = None) -> Response:
    """The pipeline as CSV (UTF-8 with BOM) for Excel or Google Sheets."""
    agent = rfp_agent()
    csv_text = agent.export_csv(agent.search(country and country.value, status and status.value))
    return Response(csv_text, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=rex_rfp_pipeline.csv"})


@app.get("/rfp/alerts", response_model=list[RFPAlert], tags=["RFP Research"])
def rfp_alerts(limit: int = 50) -> list[RFPAlert]:
    rex()
    return [RFPAlert(**a) for a in db.list_rfp_alerts(limit)]


@app.post("/rfp/ask", response_model=RFPAskResponse, tags=["RFP Research"])
def rfp_ask(req: RFPAskRequest) -> RFPAskResponse:
    """Question over the stored tenders and their documents (RAG collection rfp_docs), with citations."""
    return rfp_agent().ask(req)


@app.post("/rfp/upload", tags=["RFP Research"], responses={400: {"model": ErrorResponse}})
def rfp_upload(file: UploadFile = File(...)) -> dict:
    """Add a tender document (PDF / Word / TXT / MD) or a listing export (JSON / CSV / Excel).
    It is processed on the next POST /rfp/run. Existing files are never overwritten."""
    name = data_sync._safe_name(file.filename)
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in rfp_sources.DOC_EXTENSIONS:
        folder, kind = rfp_sources.documents_dir(), "document"
    elif ext in rfp_sources.FEED_EXTENSIONS:
        folder, kind = rfp_sources.feeds_dir(), "feed"
    else:
        allowed = sorted(rfp_sources.DOC_EXTENSIONS | rfp_sources.FEED_EXTENSIONS)
        return JSONResponse(status_code=400, content=ErrorResponse(
            code="bad_file", message=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}").model_dump())
    folder.mkdir(parents=True, exist_ok=True)
    path = data_sync._unique_path(folder, name)
    path.write_bytes(file.file.read())
    db.log_event("INFO", "rfp_upload", agent=AgentName.rfp_research.value, detail=f"{path.name}:{kind}")
    return {"file": path.name, "kind": kind, "status": "saved - run POST /rfp/run to process it"}


@app.get("/rfp/{rfp_id}", response_model=RFPRecord, tags=["RFP Research"], responses={404: {"model": ErrorResponse}})
def rfp_get(rfp_id: str) -> RFPRecord:
    rex()
    row = db.get_rfp(rfp_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    return row_to_record(row)


@app.patch("/rfp/{rfp_id}/status", response_model=RFPRecord, tags=["RFP Research"],
           responses={404: {"model": ErrorResponse}})
def rfp_set_status(rfp_id: str, update: RFPStatusUpdate) -> RFPRecord:
    """Reviewer decision, e.g. flagged_review -> qualified (bid) or rejected (no bid)."""
    rex()
    if not db.update_rfp_status(rfp_id, update.status.value):
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    db.log_event("INFO", "rfp_review", agent=AgentName.rfp_research.value,
                 detail=f"{rfp_id}:{update.status.value} by {update.reviewer}: {update.comment or ''}")
    return rfp_get(rfp_id)


# ------------------------------------------------------------ Curriculum & Accreditation Alignment (Agent 3)
def curriculum_agent() -> CurriculumAgent:
    rex()                                # DB initialised, catalogue loaded
    return CurriculumAgent()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(code=code, message=message).model_dump())


@app.get("/curriculum/frameworks", response_model=list[FrameworkInfo], tags=["Curriculum"])
def curriculum_frameworks() -> list[FrameworkInfo]:
    """Accreditation / competency frameworks available for alignment (team summaries, status to_verify)."""
    return CurriculumAgent.frameworks()


@app.post("/curriculum/research", response_model=CurriculumResearch, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}})
def curriculum_research(req: CurriculumRequest):
    """Preview what the agent found for a request: department, framework, related courses, industry skills and
    reference syllabus modules (RAG over curriculum_refs), with sources. Nothing is stored."""
    try:
        return curriculum_agent().research(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/curriculum", response_model=Curriculum, status_code=201, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}})
def curriculum_generate(req: CurriculumRequest):
    """Research and generate a structured curriculum (objectives, audience, prerequisites, outcomes, modules,
    hours, exercises, assessments, capstone), check it against the framework, the country checklist and the
    academy baseline, and store it as a draft for review."""
    try:
        return curriculum_agent().generate(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/curriculum/import", response_model=Curriculum, status_code=201, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}})
def curriculum_import(req: CurriculumImport):
    """Bring in an existing catalogue course or an instructor's proposed curriculum to compare it with a framework."""
    try:
        return curriculum_agent().import_curriculum(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/curriculum", response_model=list[Curriculum], tags=["Curriculum"])
def curriculum_list(department: str | None = None, skill_level: SkillLevel | None = None,
                    country: Country | None = None, status: ApprovalStatus | None = None,
                    framework: str | None = None, industry: str | None = None, audience: str | None = None,
                    min_hours: float | None = None, max_hours: float | None = None,
                    q: str | None = None) -> list[Curriculum]:
    """Metadata filtering: department, skill level, country, status, framework, industry, audience, duration range,
    plus free text over the title and content."""
    return curriculum_agent().search(department=department, skill_level=skill_level and skill_level.value,
                                     country=country and country.value, status=status and status.value,
                                     framework=framework, industry=industry, audience=audience,
                                     min_hours=min_hours, max_hours=max_hours, q=q)


@app.get("/curriculum/{curriculum_id}", response_model=Curriculum, tags=["Curriculum"],
         responses={404: {"model": ErrorResponse}})
def curriculum_get(curriculum_id: str):
    try:
        return curriculum_agent().get(curriculum_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/curriculum/{curriculum_id}/gap-report", response_model=GapReport, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}})
def curriculum_gap_report(curriculum_id: str, req: GapReportRequest = Body(default_factory=GapReportRequest)):
    """Compare the curriculum with a framework: covered / partial / missing competencies with evidence, missing
    topics and outcomes, coverage % and recommendations. Alignment only - never evidence of accreditation."""
    try:
        return curriculum_agent().gap_report(curriculum_id, req.framework)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/curriculum/{curriculum_id}/gap-reports", response_model=list[GapReport], tags=["Curriculum"],
         responses={404: {"model": ErrorResponse}})
def curriculum_gap_reports(curriculum_id: str):
    """Latest report per framework (the curriculum's framework first)."""
    try:
        return curriculum_agent().gap_reports(curriculum_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/curriculum/{curriculum_id}/review", response_model=Curriculum, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def curriculum_review(curriculum_id: str, action: ReviewAction):
    """pending_approval = submit for review; approved (hands off to the Document Agent), changes_requested or
    rejected = the reviewer's decision."""
    try:
        return curriculum_agent().review(curriculum_id, action)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(409, "invalid_transition", str(exc))


@app.post("/curriculum/{curriculum_id}/revise", response_model=Curriculum, tags=["Curriculum"],
          responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
def curriculum_revise(curriculum_id: str, rev: CurriculumRevision):
    """Apply a reviewer's modification request (instructions such as "add module Power BI", "remove topic X",
    "add outcome: ...", "set hours to 48", or edited fields). Creates a new version in draft and re-checks alignment.
    metadata.last_revision lists what was applied and what was not."""
    try:
        return curriculum_agent().revise(curriculum_id, rev)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_revision", str(exc))


@app.get("/curriculum/{curriculum_id}/history", tags=["Curriculum"], responses={404: {"model": ErrorResponse}})
def curriculum_history(curriculum_id: str):
    """Versions, review decisions, alignment reports and agent handoffs."""
    try:
        return curriculum_agent().history(curriculum_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/curriculum/{curriculum_id}/pdf", tags=["Curriculum"], response_class=FileResponse,
         responses={404: {"model": ErrorResponse}})
def curriculum_pdf(curriculum_id: str, include_alignment: bool = True):
    """Professional PDF (English or Arabic) with the alignment report; marked DRAFT until approved."""
    try:
        path = curriculum_agent().export_pdf(curriculum_id, include_alignment)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# ------------------------------------------------------------ Educational Content & Document Generation (Agent 4)
def document_agent() -> DocumentAgent:
    rex()                                # DB initialised, catalogue loaded
    return DocumentAgent()


@app.get("/documents/types", response_model=list[DocumentTypeInfo], tags=["Documents"])
def document_types() -> list[DocumentTypeInfo]:
    """The 10 document types, their default template and its sections (data/doc_templates/templates.yaml)."""
    return DocumentAgent.types()


@app.get("/documents/handoffs", tags=["Documents"])
def document_handoffs() -> list[dict]:
    """Approved curricula handed over by the Curriculum Agent, with the documents already made from them."""
    return document_agent().pending_handoffs()


@app.post("/documents", response_model=DocumentRecord, status_code=201, tags=["Documents"],
          responses={404: {"model": ErrorResponse}})
def document_generate(req: DocumentRequest):
    """Generate one document from a curriculum (curriculum_id) or a catalogue course (course_id): template sections
    filled from the course data, reference syllabi (RAG) and, when configured, polished by the LLM. Stored as a
    draft with its PDF. audience = who it is prepared for (e.g. the client of a proposal); email_to is sent only
    after approval."""
    try:
        return document_agent().generate(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/documents/pack", response_model=list[DocumentRecord], status_code=201, tags=["Documents"],
          responses={404: {"model": ErrorResponse}})
def document_pack(req: DocumentPackRequest):
    """Several documents from one source in one call (default: course description, participant guide, question bank)."""
    try:
        return document_agent().generate_pack(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/documents", response_model=list[DocumentRecord], tags=["Documents"])
def document_list(doc_type: DocumentType | None = None, status: ApprovalStatus | None = None,
                  course_id: str | None = None, curriculum_id: str | None = None, language: str | None = None,
                  q: str | None = None) -> list[DocumentRecord]:
    return document_agent().search(doc_type=doc_type and doc_type.value, status=status and status.value,
                                   course_id=course_id and course_id.upper(), curriculum_id=curriculum_id,
                                   language=language, q=q)


@app.get("/documents/{document_id}", response_model=DocumentRecord, tags=["Documents"],
         responses={404: {"model": ErrorResponse}})
def document_get(document_id: str):
    try:
        return document_agent().get(document_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.patch("/documents/{document_id}", response_model=DocumentRecord, tags=["Documents"],
           responses={404: {"model": ErrorResponse}})
def document_edit(document_id: str, edit: DocumentEdit):
    """Reviewer edits: replace sections (by key) or add new ones, remove sections, change the title, or give
    plain-language instructions (applied by the LLM). Creates a new version in draft; metadata.last_edit lists what
    was applied and what was not."""
    try:
        return document_agent().edit(document_id, edit)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/documents/{document_id}/review", response_model=DocumentRecord, tags=["Documents"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def document_review(document_id: str, action: ReviewAction):
    """pending_approval = submit; approved (only if its curriculum is approved and unchanged; hands off to the
    Marketing Agent and sends any pending e-mail), changes_requested or rejected."""
    try:
        return document_agent().review(document_id, action)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(409, "invalid_transition", str(exc))


@app.post("/documents/{document_id}/email", response_model=DocumentRecord, tags=["Documents"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def document_email(document_id: str, req: DocumentEmail):
    """E-mail the PDF of an APPROVED document (SMTP when configured, else saved to outputs/documents/outbox/)."""
    try:
        return document_agent().email(document_id, req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(409, "not_approved", str(exc))
    except RuntimeError as exc:
        return _error(502, "email_failed", str(exc))


@app.get("/documents/{document_id}/history", tags=["Documents"], responses={404: {"model": ErrorResponse}})
def document_history(document_id: str):
    """Versions, review decisions and agent handoffs."""
    try:
        return document_agent().history(document_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/documents/{document_id}/download", tags=["Documents"], response_class=FileResponse,
         responses={404: {"model": ErrorResponse}})
def document_download(document_id: str):
    """The PDF of the current version (DRAFT watermark until approved)."""
    try:
        path = document_agent().pdf(document_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# ------------------------------------------------------------ AI Marketing (Agent 5)
def marketing_agent() -> MarketingAgent:
    rex()                                # DB initialised, catalogue loaded, academy_kb indexed (FAQs)
    return MarketingAgent()


@app.get("/campaigns/channels", response_model=list[ChannelInfo], tags=["Marketing"])
def campaign_channels() -> list[ChannelInfo]:
    """The 5 channels with their length limits and role (data/marketing/channels.yaml)."""
    return MarketingAgent.channels()


@app.get("/campaigns/handoffs", tags=["Marketing"])
def campaign_handoffs() -> list[dict]:
    """Approved documents (Document Agent) and approved curricula (Curriculum Agent) ready for a campaign, with the
    campaigns already made from them."""
    return marketing_agent().pending_handoffs()


@app.post("/campaigns", response_model=Campaign, status_code=201, tags=["Marketing"],
          responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
def campaign_generate(req: CampaignRequest):
    """Generate a multi-channel campaign from approved course data: strategy (audience, channel mix, budget split,
    timeline, KPIs, A/B tests) and N variations per channel, every number checked against the data. Stored as
    pending_approval - nothing is published before an administrator approves it."""
    try:
        return marketing_agent().generate(req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))


@app.get("/campaigns", response_model=list[Campaign], tags=["Marketing"])
def campaign_list(course_id: str | None = None, status: ApprovalStatus | None = None,
                  curriculum_id: str | None = None, language: str | None = None,
                  q: str | None = None) -> list[Campaign]:
    return marketing_agent().search(course_id=course_id and course_id.upper(), status=status and status.value,
                                    curriculum_id=curriculum_id, language=language, q=q)


@app.get("/campaigns/{campaign_id}", response_model=Campaign, tags=["Marketing"],
         responses={404: {"model": ErrorResponse}})
def campaign_get(campaign_id: str):
    try:
        return marketing_agent().get(campaign_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.patch("/campaigns/{campaign_id}", response_model=Campaign, tags=["Marketing"],
           responses={404: {"model": ErrorResponse}})
def campaign_edit(campaign_id: str, edit: CampaignEdit):
    """Reviewer edits: replace variants (by channel + variant_no) or add new ones, remove variants ("email:2" or a
    whole channel), change the strategy, or give plain-language instructions (LLM). New version, pending_approval;
    metadata.last_edit lists what was applied and what was not."""
    try:
        return marketing_agent().edit(campaign_id, edit)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.post("/campaigns/{campaign_id}/review", response_model=Campaign, tags=["Marketing"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def campaign_review(campaign_id: str, action: ReviewAction):
    """approved (only if its curriculum / document are approved and unchanged and the fees, offers and dates it quotes
    are still current), changes_requested, rejected, or pending_approval to resubmit."""
    try:
        return marketing_agent().review(campaign_id, action)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except ValueError as exc:
        return _error(409, "invalid_transition", str(exc))


@app.post("/campaigns/{campaign_id}/publish", response_model=Campaign, tags=["Marketing"],
          responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
                     502: {"model": ErrorResponse}})
def campaign_publish(campaign_id: str, req: CampaignPublish):
    """Release an APPROVED campaign: writes the distribution package (JSON, CSV, one text file per variant) and
    e-mails the e-mail variant to consented recipients (SMTP, else outputs/campaigns/outbox/)."""
    try:
        return marketing_agent().publish(campaign_id, req)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    except LookupError as exc:
        return _error(400, "bad_variants", str(exc))
    except ValueError as exc:
        return _error(409, "not_approved", str(exc))
    except RuntimeError as exc:
        return _error(502, "email_failed", str(exc))


@app.get("/campaigns/{campaign_id}/history", tags=["Marketing"], responses={404: {"model": ErrorResponse}})
def campaign_history(campaign_id: str):
    """Versions, review decisions, publications and the document handoff it came from."""
    try:
        return marketing_agent().history(campaign_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


@app.get("/campaigns/{campaign_id}/pdf", tags=["Marketing"], response_class=FileResponse,
         responses={404: {"model": ErrorResponse}})
def campaign_pdf(campaign_id: str):
    """Campaign brief PDF for the review (DRAFT watermark until approved)."""
    try:
        path = marketing_agent().pdf(campaign_id)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# ---------------------------------------------------------- Orchestrator
DOC_TYPE_HINTS = {
    "course_description": ["course description", "course brochure", "وصف الدورة", "وصف دورة"],
    "course_proposal": ["proposal", "عرض تدريبي", "عرض فني"],
    "curriculum_pdf": ["curriculum pdf", "curriculum document"],
    "participant_guide": ["participant guide", "learner guide", "student guide", "trainee guide", "workbook",
                          "دليل المتدرب", "دليل المشارك"],
    "facilitator_guide": ["facilitator guide", "instructor guide", "trainer guide", "دليل المدرب", "دليل الميسر"],
    "presentation_outline": ["presentation", "slides", "slide deck", "عرض تقديمي", "شرائح"],
    "question_bank": ["question bank", "quiz questions", "exam questions", "بنك الأسئلة", "بنك الاسئلة", "بنك أسئلة"],
    "assessment_rubric": ["rubric", "معايير التقييم", "روبريك"],
    "capstone_brief": ["capstone brief", "project brief", "موجز المشروع"],
    "outcomes_assessment_plan": ["assessment plan", "outcomes plan", "خطة التقييم", "خطة نواتج التعلم"],
}
# Order matters: a document or campaign request often names its curriculum ("participant guide for curriculum
# CUR-...", "campaign for curriculum CUR-..."), so both are checked before the curriculum keywords.
ROUTING_HINTS = {
    AgentName.rfp_research: ["tender", "rfp", "rfq", "مناقصه", "مناقصة", "مناقصات", "منافسة", "منافسات"],
    AgentName.document: ["pdf"] + [h for hints in DOC_TYPE_HINTS.values() for h in hints],
    # Requests only: not bare "marketing", "offer" or "ad" - a lead asking about the Digital Marketing course, current
    # offers or "your Facebook ad" stays with REX
    AgentName.marketing: ["campaign", "ad copy", "linkedin post", "caption", "whatsapp message", "حمله", "حملة",
                          "promote", "marketing content", "marketing copy", "marketing email", "ترويج", "روّج"],
    AgentName.curriculum: ["curriculum", "syllabus design", "accreditation", "منهج", "اعتماد"],
}


@app.post("/route", response_model=RouteResponse, tags=["Orchestrator"])
def route(req: RouteRequest) -> RouteResponse:
    """Keyword router. Agents not built yet return a 'not_implemented' result."""
    text = req.message.lower()
    for agent, hints in ROUTING_HINTS.items():
        if any(h in text for h in hints):
            if agent == AgentName.rfp_research:
                return RouteResponse(agent=agent, intent="rfp_search", confidence=0.7, result=route_rfp(req.message))
            if agent == AgentName.curriculum:
                return RouteResponse(agent=agent, intent="curriculum_design", confidence=0.7,
                                     result=route_curriculum(req.message))
            if agent == AgentName.document:
                return RouteResponse(agent=agent, intent="document_generation", confidence=0.7,
                                     result=route_document(req.message))
            return RouteResponse(agent=agent, intent="campaign_generation", confidence=0.7,
                                 result=route_marketing(req.message))
    resp = rex().handle(ChatRequest(session_id=req.session_id, message=req.message))
    return RouteResponse(agent=AgentName.rex_assistant, intent=resp.intent, confidence=0.8,
                         result=resp.model_dump(mode="json"))

CURRICULUM_QUESTIONS = {
    "course_title": "What course would you like to create?",
    "target_audience": "Who is the target audience?",
    "skill_level": "What is the intended skill level (beginner, intermediate, advanced)?",
    "total_hours": "How many training hours are available?",
}
CURRICULUM_QUESTIONS_AR = {
    "course_title": "ما الدورة التي تريد تصميم منهجها؟",
    "target_audience": "من هي الفئة المستهدفة؟",
    "skill_level": "ما المستوى المطلوب (مبتدئ، متوسط، متقدم)؟",
    "total_hours": "كم عدد ساعات التدريب المتاحة؟",
}


def parse_curriculum_message(message: str) -> dict:
    """'Design a curriculum for Power BI for finance staff, beginner, 30 hours, ISTQB' -> request fields."""
    import re
    from rex_ai.agents.curriculum.frameworks import load_frameworks
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": detect_language(message)}
    m = re.search(r"(?:curriculum|syllabus|منهج(?:ا)?)\s+(?:for|on|about|of|لـ?|عن|في)\s+(?:an?\s+|the\s+)?"
                  r"(.+?)(?=\s+for\s+|\s+لـ?\s*|,|\.|;|\s+with\s+|\s+at\s+|\s+\d|$)", message, re.I)
    if m and m.group(1).strip():
        out["course_title"] = m.group(1).strip(" '\"")
    rest = message[m.end():] if m else message
    m = re.search(r"\bfor\s+(?!a\b|an\b|the\b)(.+?)(?=,|\.|;|،|\s+\d|\s+at\s+|\s+with\s+|$)", rest, re.I)
    if m:
        out["target_audience"] = m.group(1).strip()
    elif m := re.search(r"(?:^|\s)لل(\w+)", rest):          # "للمحاسبين" = "for the accountants"
        out["target_audience"] = "ال" + m.group(1)
    if h := re.search(r"(\d{1,4})\s*(?:hours?|hrs?|h\b|ساعة|ساعات)", message, re.I):
        out["total_hours"] = int(h.group(1))
    for level, words in (("Beginner", ["beginner", "مبتدئ"]), ("Intermediate", ["intermediate", "متوسط"]),
                         ("Advanced", ["advanced", "متقدم"])):
        if contains_any(message, words):
            out["skill_level"] = level
    for f in load_frameworks().values():
        if contains_any(message, [f.short_name, *f.aliases]) and f.framework_id != "REX-QUALITY":
            out["accreditation_framework"] = f.framework_id
            break
    for code, places in COUNTRY_PLACES.items():
        if contains_any(message, places):
            out["country"] = code
    return out


def route_curriculum(message: str) -> dict:
    """Design a curriculum from chat; asks the brief's discovery questions for anything missing."""
    fields = parse_curriculum_message(message)
    missing = [k for k in CURRICULUM_QUESTIONS if not fields.get(k)]
    if missing:
        questions = CURRICULUM_QUESTIONS_AR if fields["language"] == "ar" else CURRICULUM_QUESTIONS
        return {"status": "need_input", "understood": fields, "questions": [questions[k] for k in missing]}
    cur = curriculum_agent().generate(CurriculumRequest(**fields))
    return {"status": "draft_created", "curriculum_id": cur.curriculum_id, "course_title": cur.course_title,
            "modules": [m.title for m in cur.modules], "alignment": cur.metadata.get("alignment"),
            "next": f"Review it: POST /curriculum/{cur.curriculum_id}/review, PDF: GET /curriculum/{cur.curriculum_id}/pdf"}


def route_rfp(message: str) -> dict:
    """Answer a tender question from chat; runs the research first if nothing is stored yet."""
    agent = rfp_agent()
    if not db.list_rfps(limit=1):
        agent.run(RFPRunRequest())
    countries = [c for c, places in COUNTRY_PLACES.items() if contains_any(message, places)]
    resp = agent.ask(RFPAskRequest(question=message, country=countries[0] if len(countries) == 1 else None))
    return resp.model_dump(mode="json")


DOCUMENT_QUESTIONS = {
    "doc_type": "Which document type do you need? (course description, proposal, curriculum PDF, participant guide, "
                "facilitator guide, presentation outline, question bank, rubrics, capstone brief, outcomes and "
                "assessment plan)",
    "source": "Which course or curriculum should be used? (a course id such as DS-101, or a curriculum id CUR-...)",
}
DOCUMENT_QUESTIONS_AR = {
    "doc_type": "ما نوع المستند المطلوب؟ (وصف الدورة، عرض تدريبي، المنهج PDF، دليل المتدرب، دليل المدرب، مخطط العرض "
                "التقديمي، بنك الأسئلة، معايير التقييم، موجز المشروع الختامي، خطة النواتج والتقييم)",
    "source": "ما الدورة أو المنهج المطلوب؟ (رمز دورة مثل DS-101 أو رقم منهج CUR-...)",
}


def parse_document_message(message: str) -> dict:
    """'Participant guide for DS-101 in Arabic' -> {doc_type, course_id, language}. Uses stored ids and catalogue
    titles, so nothing already known is asked again."""
    import re
    from rex_ai.agents.curriculum.research import _words
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": "ar" if re.search(r"\b(arabic)\b|بالعربي|باللغة العربية", message, re.I)
                 else "en" if re.search(r"\benglish\b|بالإنجليزي|بالانجليزي", message, re.I)
                 else detect_language(message)}
    for doc_type, hints in DOC_TYPE_HINTS.items():
        if contains_any(message, hints):
            out["doc_type"] = doc_type
            break
    if m := re.search(r"\bCUR-[0-9A-F]{8}\b", message, re.I):
        out["curriculum_id"] = m.group(0).upper()
    courses = rex().kb.courses
    if m := re.search(r"\b([A-Z]{2,4}-\d{3})\b", message, re.I):
        if m.group(1).upper() in courses:
            out["course_id"] = m.group(1).upper()
    if not (out.get("course_id") or out.get("curriculum_id")):
        words = _words(message)
        best = max(((len(words & _words(f"{c.title_en} {c.title_ar}")), c.course_id) for c in courses.values()),
                   default=(0, None))
        if best[0] >= 2:
            out["course_id"] = best[1]
    if m := re.search(r"\bfor\s+(?:the\s+)?(.+?)(?:\s+(?:team|staff|department))\b", message, re.I):
        out["audience"] = m.group(0)[4:].strip()
    return out


def route_document(message: str) -> dict:
    """Generate a document from chat; asks the brief's discovery questions (§10) for what is missing."""
    fields = parse_document_message(message)
    missing = [k for k in ("doc_type", "source") if not (fields.get(k) if k == "doc_type" else
                                                         fields.get("course_id") or fields.get("curriculum_id"))]
    if missing:
        questions = DOCUMENT_QUESTIONS_AR if fields["language"] == "ar" else DOCUMENT_QUESTIONS
        return {"status": "need_input", "understood": fields, "questions": [questions[k] for k in missing]}
    try:
        doc = document_agent().generate(DocumentRequest(**fields))
    except NotFound as exc:
        return {"status": "not_found", "message": str(exc)}
    return {"status": "draft_created", "document_id": doc.document_id, "title": doc.title,
            "doc_type": doc.doc_type.value, "sections": [s.heading for s in doc.sections if s.level == 1][:15],
            "notes": doc.metadata.get("notes"), "download_url": doc.download_url,
            "next": f"Review it: POST /documents/{doc.document_id}/review"}


MARKETING_QUESTIONS = {
    "course_id": "Which course are we promoting? (a course id such as DS-101, or its title)",
}
MARKETING_QUESTIONS_AR = {
    "course_id": "ما الدورة التي نروّج لها؟ (رمز دورة مثل DS-101 أو اسمها)",
}
# Asked as follow-ups when not given (brief §10); sensible defaults are used meanwhile and reported as "assumed"
MARKETING_DEFAULTS = {
    "target_audience": "the course's target audience from the catalogue",
    "channels": "all five channels",
    "objective": "registrations",
    "budget": "no paid budget given",
    "offer / cta": "the current promotion and the objective's call to action",
}
CHANNEL_HINTS = {
    Channel.social_caption: ["caption", "social media post", "instagram post", "facebook post", "منشور", "بوست"],
    Channel.facebook_instagram_ad: ["facebook ad", "instagram ad", "meta ad", "ad copy", "ads", "إعلان", "اعلان"],
    Channel.linkedin_post: ["linkedin", "لينكد"],
    Channel.email: ["email", "e-mail", "newsletter", "بريد", "ايميل", "إيميل"],
    Channel.whatsapp: ["whatsapp", "واتساب", "واتس"],
}
OBJECTIVE_HINTS = {"awareness": ["awareness", "reach", "brand", "الوعي", "انتشار"],
                   "webinar sign-ups": ["webinar", "info session", "ندوة", "جلسة تعريفية"],
                   "leads": ["leads", "lead generation", "inquiries", "عملاء محتملين"],
                   "registrations": ["registration", "enrol", "enroll", "sign up", "تسجيل"]}


def parse_marketing_message(message: str) -> dict:
    """'LinkedIn post and WhatsApp message to promote DS-101 for fresh graduates, budget 5000 EGP' -> request fields.
    Uses stored course ids and catalogue titles, so nothing already known is asked again."""
    import re
    from rex_ai.agents.curriculum.research import _words
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": "ar" if re.search(r"\barabic\b|بالعربي|باللغة العربية", message, re.I)
                 else "en" if re.search(r"\benglish\b|بالإنجليزي|بالانجليزي", message, re.I)
                 else detect_language(message)}
    courses = rex().kb.courses
    if m := re.search(r"\b([A-Z]{2,4}-\d{3})\b", message, re.I):
        if m.group(1).upper() in courses:
            out["course_id"] = m.group(1).upper()
    if m := re.search(r"\bCUR-[0-9A-F]{8}\b", message, re.I):
        out["curriculum_id"] = m.group(0).upper()
        row = db.get_curriculum(out["curriculum_id"])
        if row and row.get("course_id") and "course_id" not in out:
            out["course_id"] = row["course_id"].upper()
    if "course_id" not in out:
        words = _words(message)
        best = max(((len(words & _words(f"{c.title_en} {c.title_ar}")), c.course_id) for c in courses.values()),
                   default=(0, None))
        if best[0] >= 2:
            out["course_id"] = best[1]
    channels = [ch.value for ch, hints in CHANNEL_HINTS.items() if contains_any(message, hints)]
    if channels:
        out["channels"] = channels
    for objective, hints in OBJECTIVE_HINTS.items():
        if contains_any(message, hints):
            out["objective"] = objective
            break
    if m := re.search(r"budget\s*(?:of|:)?\s*([\d,]+)|([\d,]+)\s*(?:egp|sar|جنيه|ريال)|ميزانية\s*([\d,]+)", message, re.I):
        out["budget"] = float(next(g for g in m.groups() if g).replace(",", ""))
    if m := re.search(r"\b(?:for|targeting|aimed at)\s+(?!the\s+course\b|[A-Z]{2,4}-\d{3}\b|CUR-)(.+?)"
                      r"(?=,|\.|;|\s+(?:with|budget|in\s+arabic|in\s+english|on\s+(?:linkedin|facebook|instagram|"
                      r"whatsapp))\b|$)", message, re.I):
        out["target_audience"] = m.group(1).strip()
    if m := re.search(r"(\d)\s*(?:variations?|variants?|versions?)", message, re.I):
        out["variations_per_channel"] = max(1, min(5, int(m.group(1))))
    return out


def route_marketing(message: str) -> dict:
    """Generate a campaign from chat. Only the course is required; the brief's other questions (§10) get defaults that
    are reported back, so the admin can refine them in the Marketing page or with POST /campaigns."""
    fields = parse_marketing_message(message)
    if not fields.get("course_id"):
        questions = MARKETING_QUESTIONS_AR if fields["language"] == "ar" else MARKETING_QUESTIONS
        return {"status": "need_input", "understood": fields, "questions": list(questions.values())}
    assumed = {k: v for k, v in MARKETING_DEFAULTS.items()
               if not any(f in fields for f in k.replace(" / ", " ").split())}
    try:
        camp = marketing_agent().generate(CampaignRequest(**fields))
    except NotFound as exc:
        return {"status": "not_found", "message": str(exc)}
    except ValueError as exc:
        return {"status": "invalid_request", "message": str(exc)}
    return {"status": "pending_approval", "campaign_id": camp.campaign_id, "title": camp.title,
            "audience": camp.strategy_plan["audience"]["persona_en"], "channels": [c.value for c in camp.channels],
            "variants": len(camp.variants), "headlines": camp.headlines, "strategy": camp.strategy,
            "assumed": assumed, "notes": camp.metadata.get("notes"), "download_url": camp.download_url,
            "next": f"An administrator reviews it: POST /campaigns/{camp.campaign_id}/review"}
