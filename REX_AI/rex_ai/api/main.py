"""REX AI backend API (FastAPI). Owner: Member 3; agent endpoints co-owned by agent owners.

Run:  uvicorn rex_ai.api.main:app --reload
Docs: http://localhost:8000/docs  (OpenAPI generated from rex_ai/core/schemas.py)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rex_ai import config
from rex_ai.agents.rex.assistant import RexAssistant
from rex_ai.agents.rex.knowledge_base import get_kb
from rex_ai.core import data_sync, db
from rex_ai.core.schemas import (AgentName, ChatRequest, ChatResponse, Course, ErrorResponse, Lead, LeadCreate,
                                 RAGHit, RAGSearchRequest, RAGSearchResponse, RouteRequest, RouteResponse)

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


# ---------------------------------------------------------- Orchestrator (stub)
ROUTING_HINTS = {
    AgentName.rfp_research: ["tender", "rfp", "rfq", "مناقصه", "مناقصة"],
    AgentName.curriculum: ["curriculum", "syllabus design", "accreditation", "منهج", "اعتماد"],
    AgentName.document: ["pdf", "participant guide", "facilitator guide", "question bank", "rubric", "proposal"],
    AgentName.marketing: ["campaign", "ad copy", "linkedin post", "caption", "whatsapp message", "حمله"],
}


@app.post("/route", response_model=RouteResponse, tags=["Orchestrator"])
def route(req: RouteRequest) -> RouteResponse:
    """Sprint 1 keyword router. Agents not built yet return a 'not_implemented' result."""
    text = req.message.lower()
    for agent, hints in ROUTING_HINTS.items():
        if any(h in text for h in hints):
            return RouteResponse(agent=agent, intent=agent.value, confidence=0.6,
                                 result={"status": "not_implemented", "available_in": "Sprint 2"})
    resp = rex().handle(ChatRequest(session_id=req.session_id, message=req.message))
    return RouteResponse(agent=AgentName.rex_assistant, intent=resp.intent, confidence=0.8,
                         result=resp.model_dump(mode="json"))