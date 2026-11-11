"""REX AI backend API (FastAPI). Owner: Member 3; agent endpoints co-owned by agent owners.

Run:  uvicorn rex_ai.api.main:app --reload
Docs: http://localhost:8000/docs  (OpenAPI generated from rex_ai/core/schemas.py)
"""
from __future__ import annotations

import asyncio
import logging
import threading
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
from rex_ai.agents.orchestrator.agent import FlowWatcher, Orchestrator, last_report, summary_for
from rex_ai.agents.orchestrator.tasks import TaskConflict, TaskOrchestrator
from rex_ai.api import read_cache
from rex_ai.agents.rfp.agent import RFPResearchAgent, row_to_record
from rex_ai.agents.rfp.taxonomy import COUNTRY_PLACES
from rex_ai.agents.orchestrator.parsing import (CURRICULUM_QUESTIONS, CURRICULUM_QUESTIONS_AR, DOCUMENT_QUESTIONS,
                                                DOCUMENT_QUESTIONS_AR, MARKETING_QUESTIONS, MARKETING_QUESTIONS_AR,
                                                ROUTING_HINTS, parse_curriculum_message, parse_document_message,
                                                parse_marketing_message)
from rex_ai.core import auth, data_sync, db, filestore
from rex_ai.core.llm_status import LLMUnavailable
from rex_ai.core.mailer import MailError
from rex_ai.mcp import client as mcp_tools
from rex_ai.core.schemas import (AgentName, ApprovalStatus, Campaign, CampaignEdit, CampaignPublish, CampaignRequest,
                                 Channel, ChannelInfo, ChatRequest, ChatResponse, Country, Course, Curriculum,
                                 CurriculumImport, CurriculumRequest, CurriculumResearch, CurriculumRevision,
                                 DocumentEdit, DocumentEmail, DocumentPackRequest, DocumentRecord, DocumentRequest,
                                 DocumentType, DocumentTypeInfo, ErrorResponse, FrameworkInfo, AuthResponse, LoginRequest,
                                 RoleUpdate, SignupRequest, UserPublic, GapReport, GapReportRequest, Lead, LeadCreate, RAGHit,
                                 RAGSearchRequest, RAGSearchResponse, ReviewAction, RFPAlert, RFPAskRequest,
                                 RFPAskResponse, RFPCategory, RFPRecord, RFPRunRequest, RFPRunResult, RFPStatus,
                                 RFPStatusUpdate, RouteRequest, RouteResponse, SkillLevel, TaskAnswers, TaskDecision,
                                 TaskRequest)

log = logging.getLogger("rex.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rex()                                # merges/indexes new files in data/catalog/ and data/documents/
    watcher = FlowWatcher()              # orchestrator: checks and repairs the agent flow every REX_ORCH_INTERVAL s
    watcher.start()
    warm = asyncio.create_task(read_cache.keep_warm(app))   # remembered answers stay fresh while people use them
    yield
    warm.cancel()
    watcher.stop()
    mcp_tools.reset()                    # close the MCP tool-server connection
    db.close_pools()                     # shared database: give the connections back on shutdown


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


# ---------------------------------------------------------------- Accounts and access control
PUBLIC = {"/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json", "/auth/signup", "/auth/login"}
PUBLIC_PREFIXES = ("/courses",)                       # the course catalogue is public information
USER_PATHS = {"/chat", "/leads", "/auth/me", "/auth/logout"}
USER_PREFIXES = ("/chat/",)                           # own conversations
# everything else (RFP, curriculum, documents, campaigns, data, RAG, orchestrator, users) = administrators only


def _bearer(request: Request) -> str | None:
    value = request.headers.get("authorization", "")
    return value[7:].strip() if value.lower().startswith("bearer ") else None


@app.middleware("http")
async def access_control(request: Request, call_next):
    """Who may call what (rex_ai/core/auth.py). With REX_API_AUTH=0 nothing is refused, but a token still
    identifies the account (conversations stay per account)."""
    from starlette.concurrency import run_in_threadpool
    path = request.url.path.rstrip("/") or "/"
    if read_cache.is_refresh(request):           # the API refreshing its own cache (in-process, secret header)
        return await call_next(request)
    token = _bearer(request)
    public = path in PUBLIC or path.startswith(PUBLIC_PREFIXES)    # no account needed: skip the database lookup
    user = await run_in_threadpool(auth.user_for_token, token) if token and not public else None
    request.state.user = user
    if not config.API_AUTH or path in PUBLIC or path.startswith(PUBLIC_PREFIXES):
        return await read_cache.handle(request, call_next, path)
    if user is None:
        return JSONResponse(status_code=401, content=ErrorResponse(
            code="not_authenticated", message="Log in first (Authorization: Bearer <token> from POST /auth/login)"
        ).model_dump())
    if user["role"] != "admin" and not (path in USER_PATHS or path.startswith(USER_PREFIXES)):
        return JSONResponse(status_code=403, content=ErrorResponse(
            code="forbidden", message="Only administrators can use this agent").model_dump())
    return await read_cache.handle(request, call_next, path)      # allowed: answer from memory if nothing changed


def _user(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


@app.post("/auth/signup", response_model=AuthResponse, status_code=201, tags=["Accounts"],
          responses={400: {"model": ErrorResponse}})
def auth_signup(req: SignupRequest):
    """Create an account and log in. role=admin needs admin_code (REX_ADMIN_SIGNUP_CODE), except for the first
    administrator."""
    rex()
    try:
        auth.signup(req.name, req.email, req.password, req.role.value, req.admin_code)
        user, token = auth.login(req.email, req.password)
    except auth.AuthError as exc:
        return _error(400, "signup_failed", str(exc))
    return AuthResponse(token=token, user=UserPublic(**user))


@app.post("/auth/login", response_model=AuthResponse, tags=["Accounts"], responses={401: {"model": ErrorResponse}})
def auth_login(req: LoginRequest):
    rex()
    try:
        user, token = auth.login(req.email, req.password)
    except auth.AuthError as exc:
        return _error(401, "login_failed", str(exc))
    return AuthResponse(token=token, user=UserPublic(**user))


@app.get("/auth/me", response_model=UserPublic, tags=["Accounts"], responses={401: {"model": ErrorResponse}})
def auth_me(request: Request):
    user = _user(request)
    if not user:
        return _error(401, "not_authenticated", "Log in first")
    return UserPublic(**user)


@app.post("/auth/logout", tags=["Accounts"])
def auth_logout(request: Request) -> dict:
    auth.logout(_bearer(request))
    return {"status": "logged_out"}


@app.get("/auth/users", response_model=list[UserPublic], tags=["Accounts"])
def auth_users() -> list[UserPublic]:
    """All accounts (administrators only)."""
    return [UserPublic(**u) for u in db.list_users()]


@app.patch("/auth/users/{user_id}", response_model=UserPublic, tags=["Accounts"],
           responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
def auth_set_role(user_id: str, update: RoleUpdate):
    """Make an account a user or an administrator (administrators only; the last administrator is kept)."""
    try:
        return UserPublic(**auth.set_role(user_id, update.role.value))
    except LookupError as exc:
        return _error(404, "not_found", str(exc))
    except auth.AuthError as exc:
        return _error(400, "invalid_role", str(exc))


@app.get("/health")
def health() -> dict:
    retriever = rex().retriever
    kb = get_kb()
    return {"status": "ok", "courses": len(kb.courses), "data_status": kb.meta.get("status"),
            "llm_enabled": config.USE_LLM, "gemini_edits_enabled": config.USE_GEMINI, "mcp_tools": mcp_tools.target(),
            "flow": summary_for(last_report()) if last_report() else None, "retriever": type(retriever).__name__,
            "embedder": getattr(retriever, "backend", "tfidf"), "database": db.describe()}


# ---------------------------------------------------------------- REX (Agent 1)
@app.post("/chat", response_model=ChatResponse, tags=["REX Assistant"], responses={403: {"model": ErrorResponse}})
def chat(req: ChatRequest, request: Request):
    """Send a message. With a login, the conversation is stored under that account; another account's
    conversation id is refused."""
    assistant = rex()
    user = _user(request)
    if user and not db.claim_session(req.session_id, user["user_id"]):
        return _error(403, "forbidden", "This conversation belongs to another account")
    resp = assistant.handle(req)
    db.log_event("INFO", "chat", agent=AgentName.rex_assistant.value, detail=f"{req.session_id}:{resp.intent}")
    return resp


@app.get("/chat/sessions", tags=["REX Assistant"])
def chat_sessions(request: Request) -> list[dict]:
    """The logged-in account's conversations, newest first (title = first message)."""
    rex()
    user = _user(request)
    return db.list_sessions(user["user_id"]) if user else []


@app.get("/chat/{session_id}/history", tags=["REX Assistant"])
def chat_history(session_id: str, request: Request) -> dict:
    """A stored conversation (messages oldest first), so the chat can be reopened on any server. Only the
    account that owns it gets the messages."""
    rex()
    user = _user(request)
    session = db.load_session(session_id)
    if session and (user or config.API_AUTH) and session.get("user_id") != (user or {}).get("user_id"):
        session = None                                  # someone else's: answer as if it did not exist
    return {"session_id": session_id, "found": session is not None, "language": (session or {}).get("language"),
            "stage": (session or {}).get("stage"), "messages": db.list_messages(session_id) if session else []}


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
def create_lead(lead: LeadCreate, request: Request) -> Lead:
    rex()  # make sure the DB is initialised
    user = _user(request)
    saved = db.save_lead({**lead.model_dump(mode="json"), "user_id": user["user_id"] if user else None})
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
    filestore.push()                     # shared database: other servers get the file too
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
           responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def rfp_set_status(rfp_id: str, update: RFPStatusUpdate):
    """Reviewer decision, e.g. flagged_review -> qualified (bid) or rejected (no bid). Marked qualified with an
    `email`, the tender is e-mailed there at once (502 email_failed if not sent - the status is still saved)."""
    rex()
    try:
        rec = RFPResearchAgent.set_status(rfp_id, update)
    except MailError as exc:
        return _error(502, "email_failed", str(exc))
    if rec is None:
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    return rec


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
    except LLMUnavailable as exc:                 # the LLM could not apply it: nothing changed
        return _error(503, "llm_unavailable", str(exc))
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
    except LLMUnavailable as exc:                 # the LLM could not apply it: nothing changed
        return _error(503, "llm_unavailable", str(exc))


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
    """E-mail the PDF of an APPROVED document over SMTP (502 email_failed with the reason if it is not sent)."""
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
    except LLMUnavailable as exc:                 # the LLM could not apply it: nothing changed
        return _error(503, "llm_unavailable", str(exc))


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
_first_check = threading.Lock()


@app.get("/orchestrator/status", tags=["Orchestrator"])
def orchestrator_status() -> dict:
    """The last flow report (checked at start-up and every REX_ORCH_INTERVAL s). If there is none yet, the first check
    starts in the background and this answers at once with pending=true (a full check takes 10-30 s)."""
    rex()
    if report := last_report():
        return report
    if _first_check.acquire(blocking=False):
        def run():
            try:
                Orchestrator().check(repair=False)
            finally:
                _first_check.release()
        threading.Thread(target=run, name="rex-first-check", daemon=True).start()
    from datetime import datetime, timezone
    return {"status": "warning", "pending": True, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repair": False, "deep": False, "summary": {"warning": 1}, "repairs": [], "pipeline": Orchestrator().pipeline(),
            "checks": [{"area": "infrastructure", "name": "Flow check", "status": "warning", "items": [],
                        "detail": "The first check of every agent and handoff is running - it appears here in a few "
                                  "seconds."}]}


@app.post("/orchestrator/check", tags=["Orchestrator"])
def orchestrator_check(repair: bool = True, deep: bool = False) -> dict:
    """Check every agent, the MCP tools and each handoff now. repair=true fixes missing handoffs, expired tenders,
    stuck jobs and pending document e-mails; deep=true also logs in to the mail server (sends nothing)."""
    rex()
    return Orchestrator().check(repair=repair, deep=deep)


@app.get("/orchestrator/tools", tags=["Orchestrator"])
def orchestrator_tools():
    """The MCP tool server the agents use for every external service, its tools and which services are set up."""
    try:
        return {"server": mcp_tools.target(), "tools": mcp_tools.list_tools(),
                "services": mcp_tools.call("service_status")}
    except mcp_tools.ToolCallError as exc:
        return _error(502, "mcp_unavailable", str(exc))


@app.get("/orchestrator/jobs", tags=["Orchestrator"])
def orchestrator_jobs(status: str | None = None, agent: str | None = None, limit: int = 50) -> list[dict]:
    """Agent runs (generation, routed requests): queued -> running -> succeeded / failed, newest first."""
    rex()
    return db.list_jobs(status=status, agent=agent, limit=min(limit, 500))


def _admin_name(request: Request) -> str:
    user = _user(request)
    return user["name"] if user else "administrator"


def _resume(task_id: str, change) -> dict | JSONResponse:
    """Apply an answer or a confirmation, then carry on in the background unless that ended the task."""
    orch = TaskOrchestrator()
    try:
        task = change(orch)
    except LookupError as exc:
        return _error(404, "not_found", str(exc))
    except TaskConflict as exc:
        return _error(409, "conflict", str(exc))
    if task["status"] == "running":
        orch.start(task_id)
    return task


@app.post("/orchestrator/tasks", status_code=202, tags=["Orchestrator"])
def task_create(req: TaskRequest, request: Request) -> dict:
    """Give the orchestrator a task in plain language. It runs in the background: follow it with GET
    /orchestrator/tasks/{task_id}. status needs_input -> POST .../answers; awaiting_approval -> POST .../decision."""
    rex()
    orch = TaskOrchestrator()
    task = orch.submit(req.message, _admin_name(request))
    orch.start(task["task_id"])
    return task


@app.get("/orchestrator/tasks", tags=["Orchestrator"])
def task_list(status: str | None = None, limit: int = 30) -> list[dict]:
    """Tasks, newest first."""
    rex()
    return db.list_tasks(status=status, limit=min(limit, 200))


@app.get("/orchestrator/tasks/{task_id}", tags=["Orchestrator"], responses={404: {"model": ErrorResponse}})
def task_get(task_id: str):
    """A task: status, each step with its result, what it created, and any question or action waiting for a person."""
    task = db.get_task(task_id)
    return task or _error(404, "not_found", f"Task {task_id} not found")


@app.post("/orchestrator/tasks/{task_id}/answers", tags=["Orchestrator"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def task_answers(task_id: str, req: TaskAnswers):
    """Answer the questions of a task in status needs_input, in the same order."""
    return _resume(task_id, lambda orch: orch.answer(task_id, req.answers))


@app.post("/orchestrator/tasks/{task_id}/decision", tags=["Orchestrator"],
          responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def task_decision(task_id: str, req: TaskDecision, request: Request):
    """Confirm or decline the action a task paused on. Recorded under the logged-in administrator's name."""
    return _resume(task_id, lambda orch: orch.decide(task_id, req.approve, _admin_name(request)))


@app.post("/route", response_model=RouteResponse, tags=["Orchestrator"])
def route(req: RouteRequest) -> RouteResponse:
    """Orchestrator: keyword routing to the right agent. Every routed run is a tracked job; an agent that fails
    returns {"status": "error", "message": ...} instead of a 500."""
    text = req.message.lower()
    handlers = {AgentName.rfp_research: ("rfp_search", route_rfp),
                AgentName.curriculum: ("curriculum_design", route_curriculum),
                AgentName.document: ("document_generation", route_document),
                AgentName.marketing: ("campaign_generation", route_marketing)}
    for agent, hints in ROUTING_HINTS.items():
        if any(h in text for h in hints):
            intent, handler = handlers[agent]
            result = Orchestrator().dispatch(agent, intent, req.message, lambda: handler(req.message))
            return RouteResponse(agent=agent, intent=intent, confidence=0.7, result=result)
    resp = rex().handle(ChatRequest(session_id=req.session_id, message=req.message))
    return RouteResponse(agent=AgentName.rex_assistant, intent=resp.intent, confidence=0.8,
                         result=resp.model_dump(mode="json"))

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


MARKETING_DEFAULTS = {
    "target_audience": "the course's target audience from the catalogue",
    "channels": "all five channels",
    "objective": "registrations",
    "budget": "no paid budget given",
    "offer / cta": "the current promotion and the objective's call to action",
}
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
