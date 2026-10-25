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
from rex_ai.agents.rex.assistant import RexAssistant
from rex_ai.agents.rex.knowledge_base import get_kb
from rex_ai.agents.rex.text_utils import contains_any
from rex_ai.agents.rfp import sources as rfp_sources
from rex_ai.agents.rfp.agent import RFPResearchAgent, row_to_record
from rex_ai.agents.rfp.taxonomy import COUNTRY_PLACES
from rex_ai.core import data_sync, db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, ChatRequest, ChatResponse, Country, Course, Curriculum,
                                 CurriculumImport, CurriculumRequest, CurriculumResearch, CurriculumRevision,
                                 ErrorResponse, FrameworkInfo, GapReport, GapReportRequest, Lead, LeadCreate, RAGHit,
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


# ---------------------------------------------------------- Orchestrator
ROUTING_HINTS = {
    AgentName.rfp_research: ["tender", "rfp", "rfq", "مناقصه", "مناقصة", "مناقصات", "منافسة", "منافسات"],
    AgentName.curriculum: ["curriculum", "syllabus design", "accreditation", "منهج", "اعتماد"],
    AgentName.document: ["pdf", "participant guide", "facilitator guide", "question bank", "rubric", "proposal"],
    AgentName.marketing: ["campaign", "ad copy", "linkedin post", "caption", "whatsapp message", "حمله"],
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
            return RouteResponse(agent=agent, intent=agent.value, confidence=0.6,
                                 result={"status": "not_implemented", "available_in": "Sprint 2"})
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
