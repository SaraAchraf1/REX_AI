"""What the task orchestrator can ask the other agents to do (owner: Member 3).

Three kinds of tool:
    read     looks something up
    draft    creates or changes a draft, which still waits for a person's review
    gated    leaves the academy or records a human decision (approve, e-mail, publish, qualify a tender): the task
             pauses until an administrator confirms this exact action, and that administrator is recorded as the actor

The gate lives in the engine (tasks.py), not in the planner's instructions, so no plan and no text read from a tender
or a document can skip it. Add a capability here and both the planner and the Task page know about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

from rex_ai.agents.curriculum.agent import CurriculumAgent, NotFound
from rex_ai.agents.curriculum.research import _words
from rex_ai.agents.documents.agent import DocumentAgent
from rex_ai.agents.marketing.agent import MarketingAgent
from rex_ai.agents.rfp.agent import RFPResearchAgent
from rex_ai.core import data_sync, db
from rex_ai.core.schemas import (AgentName, ApprovalStatus, CampaignEdit, CampaignPublish, CampaignRequest, ChatRequest,
                                 Country, CurriculumRequest, CurriculumRevision, DocumentEdit, DocumentEmail,
                                 DocumentPackRequest, DocumentRequest, ReviewAction, RFPAskRequest, RFPRunRequest,
                                 RFPStatus, RFPStatusUpdate)

READ, DRAFT, GATED = "read", "draft", "gated"
Decision = Literal["approved", "changes_requested", "rejected"]


@dataclass(frozen=True)
class Context:
    task_id: str
    admin: str                                  # who asked for the task, or who confirmed this action


@dataclass(frozen=True)
class Tool:
    name: str
    agent: AgentName
    kind: str
    description: str
    args: type[BaseModel]
    run: Callable[[Any, Context], dict]
    describe: Optional[Callable[[Any], str]] = None     # gated: the sentence the administrator confirms
    fixed: dict = field(default_factory=dict)           # arguments the planner can neither see nor set


# ---------------------------------------------------------------------- arguments
class NoArgs(BaseModel):
    pass


class CourseSearch(BaseModel):
    query: str = ""
    department: Optional[str] = None
    limit: int = Field(15, ge=1, le=50)


class CourseQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Listing(BaseModel):
    status: Optional[ApprovalStatus] = None
    q: Optional[str] = None
    course_id: Optional[str] = None
    limit: int = Field(20, ge=1, le=50)


class TenderListing(BaseModel):
    country: Optional[Country] = None
    status: Optional[RFPStatus] = None
    q: Optional[str] = None
    limit: int = Field(20, ge=1, le=50)


class CurriculumId(BaseModel):
    curriculum_id: str


class DocumentId(BaseModel):
    document_id: str


class CampaignId(BaseModel):
    campaign_id: str


class TenderId(BaseModel):
    rfp_id: str


class GapArgs(CurriculumId):
    framework: Optional[str] = None


class CurriculumChange(CurriculumId):
    instructions: str = Field(..., min_length=2, max_length=4000)


class DocumentChange(DocumentId):
    instructions: str = Field(..., min_length=2, max_length=4000)


class CampaignChange(CampaignId):
    instructions: str = Field(..., min_length=2, max_length=4000)


class CurriculumDecision(CurriculumId):
    decision: Decision
    comment: Optional[str] = None


class DocumentDecision(DocumentId):
    decision: Decision
    comment: Optional[str] = None


class CampaignDecision(CampaignId):
    decision: Decision
    comment: Optional[str] = None


class SendDocument(DocumentId):
    to: list[EmailStr] = Field(..., min_length=1)
    message: Optional[str] = Field(None, max_length=4000)


class Publish(CampaignId):
    variant_keys: list[str] = Field(default_factory=list)
    email_to: list[EmailStr] = Field(default_factory=list)


class TenderDecision(TenderId):
    status: RFPStatus
    comment: Optional[str] = None
    email: Optional[EmailStr] = None


# ---------------------------------------------------------------------- short results (what the planner and page see)
def _curriculum(c) -> dict:
    return {"curriculum_id": c.curriculum_id, "title": c.course_title, "course_id": c.course_id,
            "status": c.status.value, "version": c.version, "hours": c.training_hours,
            "modules": [m.title for m in c.modules], "outcomes": c.learning_outcomes[:8],
            "alignment": c.metadata.get("alignment"), "generated_by": c.metadata.get("generated_by"),
            "not_applied": c.metadata.get("not_applied")}


def _document(d) -> dict:
    return {"document_id": d.document_id, "title": d.title, "doc_type": d.doc_type.value, "course_id": d.course_id,
            "curriculum_id": d.curriculum_id, "status": d.status.value, "version": d.version,
            "sections": [s.heading for s in d.sections if s.level == 1][:15], "notes": d.metadata.get("notes"),
            "emailed_to": d.emailed_to, "emails": d.metadata.get("emails", [])[-1:], "download_url": d.download_url}


def _campaign(c) -> dict:
    return {"campaign_id": c.campaign_id, "title": c.title, "course_id": c.course_id, "status": c.status.value,
            "version": c.version, "channels": [x.value for x in c.channels], "variants": [v.key for v in c.variants],
            "headlines": c.headlines[:3], "strategy": c.strategy, "notes": c.metadata.get("notes"),
            "publications": len(c.metadata.get("publications", [])), "download_url": c.download_url}


def _tender(r: dict) -> dict:
    return {k: r.get(k) for k in ("rfp_id", "title", "issuing_organization", "country", "deadline", "status",
                                  "training_category", "qualification", "summary")}


# ---------------------------------------------------------------------- read
def _search_courses(a: CourseSearch, ctx: Context) -> dict:
    courses = data_sync.get_assistant().kb.courses.values()
    if a.department:
        courses = [c for c in courses if a.department.lower() in c.department.lower()]
    words = _words(a.query)
    scored = sorted(((len(words & _words(f"{c.course_id} {c.title_en} {c.title_ar} {' '.join(c.tags)}")), c.course_id, c)
                     for c in courses), key=lambda x: (-x[0], x[1]))
    found = [c for score, _, c in scored if score or not words][:a.limit]
    return {"courses": [{"course_id": c.course_id, "title": c.title_en, "department": c.department,
                         "level": c.skill_level.value, "hours": c.duration_hours, "audience": c.target_audience}
                        for c in found]}


def _ask_assistant(a: CourseQuestion, ctx: Context) -> dict:
    reply = data_sync.get_assistant().handle(ChatRequest(session_id=ctx.task_id, message=a.question))
    return {"answer": reply.reply, "recommendations": [r.course_id for r in reply.recommendations]}


def _list_curricula(a: Listing, ctx: Context) -> dict:
    rows = db.list_curricula(status=a.status and a.status.value, q=a.q, limit=200)
    if a.course_id:
        rows = [r for r in rows if (r.get("course_id") or "").upper() == a.course_id.upper()]
    return {"curricula": [{k: r.get(k) for k in ("curriculum_id", "course_title", "course_id", "status", "version")}
                          for r in rows[:a.limit]]}


def _list_documents(a: Listing, ctx: Context) -> dict:
    rows = db.list_documents(status=a.status and a.status.value, q=a.q, course_id=a.course_id, limit=a.limit)
    return {"documents": [{k: r.get(k) for k in ("document_id", "title", "doc_type", "course_id", "status")}
                          for r in rows]}


def _list_campaigns(a: Listing, ctx: Context) -> dict:
    rows = db.list_campaigns(status=a.status and a.status.value, q=a.q, course_id=a.course_id, limit=a.limit)
    return {"campaigns": [{"campaign_id": r["campaign_id"], "title": r["content"].get("title"),
                           "course_id": r.get("course_id"), "status": r["status"]} for r in rows]}


def _list_tenders(a: TenderListing, ctx: Context) -> dict:
    rows = db.list_rfps(country=a.country and a.country.value, status=a.status and a.status.value, q=a.q, limit=a.limit)
    return {"tenders": [_tender(r) for r in rows]}


def _get_tender(a: TenderId, ctx: Context) -> dict:
    row = db.get_rfp(a.rfp_id)
    if not row:
        raise NotFound(f"Tender {a.rfp_id} not found")
    return {**_tender(row), "eligibility": row.get("eligibility_requirements"),
            "matched_course_ids": row.get("matched_course_ids")}


def _ask_tenders(a: RFPAskRequest, ctx: Context) -> dict:
    return RFPResearchAgent().ask(a).model_dump(mode="json", include={"answer", "rfp_ids"})


def _flow_status(a: NoArgs, ctx: Context) -> dict:
    from rex_ai.agents.orchestrator.agent import Orchestrator
    report = Orchestrator().check(repair=False)
    return {"status": report["status"], "summary": report["summary"], "pipeline": report["pipeline"],
            "problems": [f"{c['name']}: {c['detail']}" for c in report["checks"] if c["status"] != "ok"]}


# ---------------------------------------------------------------------- drafts
def _create_curriculum(a: CurriculumRequest, ctx: Context) -> dict:
    return _curriculum(CurriculumAgent().generate(a))


def _revise_curriculum(a: CurriculumChange, ctx: Context) -> dict:
    return _curriculum(CurriculumAgent().revise(a.curriculum_id, CurriculumRevision(reviewer=ctx.admin,
                                                                                     instructions=a.instructions)))


def _gap_report(a: GapArgs, ctx: Context) -> dict:
    r = CurriculumAgent().gap_report(a.curriculum_id, a.framework)
    return {"curriculum_id": r.curriculum_id, "framework": r.framework, "coverage_pct": r.coverage_pct,
            "missing_topics": r.missing_topics[:15], "missing_outcomes": r.missing_outcomes[:15],
            "recommendations": r.recommendations[:10], "disclaimer": r.disclaimer}


def _create_document(a: DocumentRequest, ctx: Context) -> dict:
    return _document(DocumentAgent().generate(a))


def _create_pack(a: DocumentPackRequest, ctx: Context) -> dict:
    return {"documents": [_document(d) for d in DocumentAgent().generate_pack(a)]}


def _edit_document(a: DocumentChange, ctx: Context) -> dict:
    return _document(DocumentAgent().edit(a.document_id, DocumentEdit(reviewer=ctx.admin, instructions=a.instructions)))


def _create_campaign(a: CampaignRequest, ctx: Context) -> dict:
    return _campaign(MarketingAgent().generate(a))


def _edit_campaign(a: CampaignChange, ctx: Context) -> dict:
    return _campaign(MarketingAgent().edit(a.campaign_id, CampaignEdit(reviewer=ctx.admin, instructions=a.instructions)))


def _run_tender_research(a: RFPRunRequest, ctx: Context) -> dict:
    r = RFPResearchAgent().run(a)
    return {"found": r.found, "kept": r.kept, "duplicates": r.duplicates, "expired": r.expired,
            "new_egypt": r.new_flagged_egypt, "new_saudi": r.new_monitoring_saudi,
            "tenders": [_tender(x.model_dump(mode="json")) for x in r.records[:15]],
            "source_errors": r.source_errors[:5]}


# ---------------------------------------------------------------------- gated
def _review_curriculum(a: CurriculumDecision, ctx: Context) -> dict:
    return _curriculum(CurriculumAgent().review(a.curriculum_id, ReviewAction(
        action=a.decision, reviewer=ctx.admin, comment=a.comment)))


def _review_document(a: DocumentDecision, ctx: Context) -> dict:
    return _document(DocumentAgent().review(a.document_id, ReviewAction(
        action=a.decision, reviewer=ctx.admin, comment=a.comment)))


def _review_campaign(a: CampaignDecision, ctx: Context) -> dict:
    return _campaign(MarketingAgent().review(a.campaign_id, ReviewAction(
        action=a.decision, reviewer=ctx.admin, comment=a.comment)))


def _email_document(a: SendDocument, ctx: Context) -> dict:
    return _document(DocumentAgent().email(a.document_id, DocumentEmail(to=a.to, sender=ctx.admin, message=a.message)))


def _publish_campaign(a: Publish, ctx: Context) -> dict:
    return _campaign(MarketingAgent().publish(a.campaign_id, CampaignPublish(
        publisher=ctx.admin, variant_keys=a.variant_keys, email_to=a.email_to)))


def _set_tender_status(a: TenderDecision, ctx: Context) -> dict:
    rec = RFPResearchAgent.set_status(a.rfp_id, RFPStatusUpdate(status=a.status, reviewer=ctx.admin, comment=a.comment,
                                                                email=a.email))
    if rec is None:
        raise NotFound(f"Tender {a.rfp_id} not found")
    return _tender(rec.model_dump(mode="json"))


# Each sentence reads the record first, so a wrong id fails here and the administrator is never asked about it.
_DECISION = {"approved": "Approve", "changes_requested": "Request changes on", "rejected": "Reject"}


def _with_comment(text: str, a) -> str:
    return f"{text} - comment: {a.comment}" if a.comment else text


def _describe_curriculum_review(a: CurriculumDecision) -> str:
    c = CurriculumAgent().get(a.curriculum_id)
    return _with_comment(f"{_DECISION[a.decision]} curriculum {c.curriculum_id} \"{c.course_title}\" (v{c.version})", a)


def _describe_document_review(a: DocumentDecision) -> str:
    d = DocumentAgent().get(a.document_id)
    text = f"{_DECISION[a.decision]} document {d.document_id} \"{d.title}\" (v{d.version})"
    if a.decision == "approved" and d.metadata.get("pending_email"):
        text += f" - this also e-mails it to {', '.join(d.metadata['pending_email'])}"
    return _with_comment(text, a)


def _describe_campaign_review(a: CampaignDecision) -> str:
    c = MarketingAgent().get(a.campaign_id)
    return _with_comment(f"{_DECISION[a.decision]} campaign {c.campaign_id} \"{c.title}\" (v{c.version})", a)


def _describe_email(a: SendDocument) -> str:
    d = DocumentAgent().get(a.document_id)
    return f"E-mail document {d.document_id} \"{d.title}\" to {', '.join(a.to)}"


def _describe_publish(a: Publish) -> str:
    c = MarketingAgent().get(a.campaign_id)
    chosen = ", ".join(a.variant_keys) or f"all {len(c.variants)} variants"
    mail = f", and e-mail it to {', '.join(a.email_to)}" if a.email_to else ""
    return f"Publish campaign {c.campaign_id} \"{c.title}\" ({chosen}){mail}"


def _describe_tender(a: TenderDecision) -> str:
    row = db.get_rfp(a.rfp_id)
    if not row:
        raise NotFound(f"Tender {a.rfp_id} not found")
    mail = f", and e-mail it to {a.email}" if a.email and a.status == RFPStatus.qualified else ""
    return _with_comment(f"Mark tender {a.rfp_id} \"{row['title']}\" as {a.status.value}{mail}", a)


_ALL = [
    # ------------------------------------------------------------------ read
    Tool("search_courses", AgentName.rex_assistant, READ, "Find catalogue courses and their course_id.",
         CourseSearch, _search_courses),
    Tool("ask_course_assistant", AgentName.rex_assistant, READ,
         "Ask the REX Course Assistant about courses, fees, schedules, admission or recommendations.",
         CourseQuestion, _ask_assistant),
    Tool("list_curricula", AgentName.curriculum, READ, "List stored curricula.", Listing, _list_curricula),
    Tool("get_curriculum", AgentName.curriculum, READ, "Read one curriculum.", CurriculumId,
         lambda a, ctx: _curriculum(CurriculumAgent().get(a.curriculum_id))),
    Tool("list_frameworks", AgentName.curriculum, READ, "Accreditation frameworks a curriculum can be aligned with.",
         NoArgs, lambda a, ctx: {"frameworks": [f.model_dump(mode="json", include={"framework_id", "name"})
                                                for f in CurriculumAgent.frameworks()]}),
    Tool("list_documents", AgentName.document, READ, "List stored documents.", Listing, _list_documents),
    Tool("get_document", AgentName.document, READ, "Read one document.", DocumentId,
         lambda a, ctx: _document(DocumentAgent().get(a.document_id))),
    Tool("list_document_types", AgentName.document, READ, "Document types that can be generated.", NoArgs,
         lambda a, ctx: {"types": [t.model_dump(mode="json") for t in DocumentAgent.types()]}),
    Tool("list_campaigns", AgentName.marketing, READ, "List stored campaigns.", Listing, _list_campaigns),
    Tool("get_campaign", AgentName.marketing, READ, "Read one campaign, with its variant keys.", CampaignId,
         lambda a, ctx: _campaign(MarketingAgent().get(a.campaign_id))),
    Tool("list_channels", AgentName.marketing, READ, "Marketing channels and their limits.", NoArgs,
         lambda a, ctx: {"channels": [c.model_dump(mode="json") for c in MarketingAgent.channels()]}),
    Tool("list_tenders", AgentName.rfp_research, READ, "List stored tenders (Egypt and Saudi Arabia only).",
         TenderListing, _list_tenders),
    Tool("get_tender", AgentName.rfp_research, READ, "Read one tender.", TenderId, _get_tender),
    Tool("ask_tenders", AgentName.rfp_research, READ, "Answer a question from the stored tenders.",
         RFPAskRequest, _ask_tenders),
    Tool("flow_status", AgentName.orchestrator, READ,
         "Health of the agents, tools and handoffs, and how much work is at each stage. Repairs nothing.",
         NoArgs, _flow_status),
    # ------------------------------------------------------------------ drafts
    Tool("create_curriculum", AgentName.curriculum, DRAFT,
         "Design a new curriculum as a draft. Needs course_id, OR course_title + target_audience + skill_level + "
         "total_hours.", CurriculumRequest, _create_curriculum),
    Tool("revise_curriculum", AgentName.curriculum, DRAFT,
         "Change a curriculum as described in plain language. Makes a new version and sends it back to draft.",
         CurriculumChange, _revise_curriculum),
    Tool("curriculum_gap_report", AgentName.curriculum, DRAFT, "Compare a curriculum with an accreditation framework.",
         GapArgs, _gap_report),
    Tool("create_document", AgentName.document, DRAFT,
         "Generate one document as a draft, from a catalogue course_id or a curriculum_id. A draft curriculum is "
         "accepted; the document cannot be approved until its curriculum is.", DocumentRequest, _create_document,
         fixed={"email_to": None}),
    Tool("create_document_pack", AgentName.document, DRAFT, "Generate several documents from one source.",
         DocumentPackRequest, _create_pack),
    Tool("edit_document", AgentName.document, DRAFT, "Change a document as described in plain language.",
         DocumentChange, _edit_document),
    Tool("create_campaign", AgentName.marketing, DRAFT,
         "Generate a marketing campaign for review. course_id must be a catalogue course: use search_courses first. "
         "A curriculum created without a course_id cannot be promoted.", CampaignRequest, _create_campaign),
    Tool("edit_campaign", AgentName.marketing, DRAFT, "Change a campaign as described in plain language.",
         CampaignChange, _edit_campaign),
    Tool("run_tender_research", AgentName.rfp_research, DRAFT,
         "Collect and qualify new tenders from the configured sources.", RFPRunRequest, _run_tender_research,
         fixed={"notify": False}),
    # ------------------------------------------------------------------ gated
    Tool("review_curriculum", AgentName.curriculum, GATED, "Record a decision on a curriculum.",
         CurriculumDecision, _review_curriculum, _describe_curriculum_review),
    Tool("review_document", AgentName.document, GATED, "Record a decision on a document.",
         DocumentDecision, _review_document, _describe_document_review),
    Tool("review_campaign", AgentName.marketing, GATED, "Record a decision on a campaign.",
         CampaignDecision, _review_campaign, _describe_campaign_review),
    Tool("email_document", AgentName.document, GATED, "E-mail an approved document.", SendDocument, _email_document,
         _describe_email),
    Tool("publish_campaign", AgentName.marketing, GATED,
         "Publish an approved campaign; email_to also sends its e-mail variant.", Publish, _publish_campaign,
         _describe_publish),
    Tool("set_tender_status", AgentName.rfp_research, GATED,
         "Record a decision on a tender; status qualified with email also e-mails it.",
         TenderDecision, _set_tender_status, _describe_tender),
]
TOOLS: dict[str, Tool] = {t.name: t for t in _ALL}


def signature(tool: Tool) -> str:
    """'create_document(doc_type: a|b, course_id?: string, ...)', from the same model that validates the call."""
    schema = tool.args.model_json_schema()
    defs = schema.get("$defs", {})

    def kind(prop: dict) -> str:
        prop = defs[prop["$ref"].rsplit("/", 1)[-1]] if "$ref" in prop else prop
        if "enum" in prop:
            return "|".join(str(v) for v in prop["enum"])
        if "const" in prop:
            return str(prop["const"])
        if "anyOf" in prop:
            return " or ".join(kind(p) for p in prop["anyOf"] if p.get("type") != "null")
        if prop.get("type") == "array":
            return f"list[{kind(prop.get('items', {}))}]"
        return prop.get("type", "any")

    required = set(schema.get("required", []))
    params = ", ".join(f"{name}{'' if name in required else '?'}: {kind(prop)}"
                       for name, prop in schema.get("properties", {}).items() if name not in tool.fixed)
    return f"{tool.name}({params})"
