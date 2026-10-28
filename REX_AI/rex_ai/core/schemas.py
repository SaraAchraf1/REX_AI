"""Shared data contracts for every REX AI agent.

These Pydantic models ARE the integration interface: the FastAPI endpoints,
the orchestrator and the agent-to-agent handoffs all exchange these objects.
Change a model only through a Pull Request reviewed by the agent owners
that use it (see docs/interfaces.md).

Owner: Member 3 (shared schemas). Agent sections are co-owned:
  REX + Marketing -> Member 1, RFP + Curriculum -> Member 2, Document -> Member 3.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Common enums and types
# --------------------------------------------------------------------------- #
class Language(str, Enum):
    en = "en"
    ar = "ar"


class SkillLevel(str, Enum):
    beginner = "Beginner"
    intermediate = "Intermediate"
    advanced = "Advanced"


class TrainingFormat(str, Enum):
    online_live = "online_live"
    onsite = "onsite"
    hybrid = "hybrid"


class Country(str, Enum):
    EG = "EG"
    SA = "SA"


class ApprovalStatus(str, Enum):
    draft = "draft"
    pending_approval = "pending_approval"
    changes_requested = "changes_requested"
    approved = "approved"
    rejected = "rejected"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class AgentName(str, Enum):
    rex_assistant = "rex_assistant"
    rfp_research = "rfp_research"
    curriculum = "curriculum"
    document = "document"
    marketing = "marketing"


class SourceRef(BaseModel):
    """Every generated answer or record points back to where its facts came from."""
    source_id: str = Field(..., examples=["SRC-ACAD-COURSES"])
    title: str
    url: Optional[str] = None
    record_id: Optional[str] = Field(None, description="e.g. a course_id or FAQ id inside the source")
    retrieved_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    code: str = Field(..., examples=["validation_error", "not_found", "agent_failed"])
    message: str
    details: Optional[dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Course knowledge base (owner: Member 1)
# --------------------------------------------------------------------------- #
class Instructor(BaseModel):
    instructor_id: str
    name: str
    title: str
    years_experience: int
    expertise: list[str]
    bio_en: str
    bio_ar: str


class Schedule(BaseModel):
    schedule_id: str
    course_id: str
    start_date: date
    format: TrainingFormat
    days: str
    time: str
    location: Optional[str] = None
    seats_left: int


class Promotion(BaseModel):
    promo_id: str
    name_en: str
    name_ar: str
    discount_pct: float
    applies_to: list[str] | str  # "all" or list of course_ids
    condition_en: str
    condition_ar: str
    valid_until: date


class Course(BaseModel):
    """Metadata fields follow the brief's knowledge-base table (department, audience, level...)."""
    course_id: str
    title_en: str
    title_ar: str
    department: str
    industry: list[str]
    skill_level: SkillLevel
    target_audience: str
    duration_hours: int
    duration_weeks: int
    formats: list[TrainingFormat]
    language: list[Language]
    description_en: str
    description_ar: str
    learning_outcomes: list[str]
    modules: list[str]
    prerequisites: list[str]
    career_opportunities: list[str]
    tags: list[str]
    certificate: str
    accreditation_alignment: Optional[str] = None
    instructor_ids: list[str]
    country: list[Country]
    # Joined at load time
    fee: Optional[float] = None
    currency: str = "EGP"
    installments: Optional[bool] = None
    source_id: str = "SRC-ACAD-COURSES"
    last_updated: Optional[date] = None
    is_sample: bool = True


class FAQ(BaseModel):
    faq_id: str
    category: str
    q_en: str
    a_en: str
    q_ar: str
    a_ar: str
    keywords: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent 1 - REX Course Assistant (owner: Member 1)
# --------------------------------------------------------------------------- #
class LeadProfile(BaseModel):
    """What REX learns during discovery. All fields optional; REX asks only for what is missing."""
    background: Optional[str] = None
    education_level: Optional[str] = None
    experience_years: Optional[float] = None
    interests: list[str] = Field(default_factory=list, description="Departments or topics, e.g. ['AI','Data']")
    career_goal: Optional[str] = None
    skill_level: Optional[SkillLevel] = None
    preferred_format: Optional[TrainingFormat] = None
    budget: Optional[float] = Field(None, description="Maximum budget in EGP")
    weekly_hours: Optional[float] = None
    language: Language = Language.en


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    language: Optional[Language] = Field(None, description="Force a reply language; auto-detected if omitted")


class CourseRecommendation(BaseModel):
    course_id: str
    title: str
    score: float = Field(..., ge=0, le=1)
    reasons: list[str]
    fee: Optional[float] = None
    currency: str = "EGP"
    next_start: Optional[date] = None
    format_options: list[TrainingFormat] = Field(default_factory=list)


class ChatStage(str, Enum):
    discovery = "discovery"
    recommendation = "recommendation"
    qa = "qa"
    lead_capture = "lead_capture"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    language: Language
    stage: ChatStage
    intent: str
    profile: LeadProfile
    recommendations: list[CourseRecommendation] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    offer_lead_capture: bool = False
    llm_used: bool = False


class LeadCreate(BaseModel):
    session_id: str
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9 \-]{7,20}$")
    interested_course_ids: list[str] = Field(default_factory=list)
    profile: Optional[LeadProfile] = None
    consent: bool = Field(..., description="Must be true; leads are never stored without explicit consent")

    @model_validator(mode="after")
    def _check(self) -> "LeadCreate":
        if not self.consent:
            raise ValueError("Lead cannot be saved without explicit consent")
        if not (self.email or self.phone):
            raise ValueError("Provide an email or a phone number")
        return self


class Lead(LeadCreate):
    lead_id: str
    created_at: datetime


# --------------------------------------------------------------------------- #
# Agent 2 - Tender / RFP Research (owner: Member 2)
# --------------------------------------------------------------------------- #
class RFPCategory(str, Enum):
    education_training = "Education and Training"
    technical_training = "Technical Training"
    professional_development = "Professional Development"
    digital_transformation = "Digital Transformation"
    software_it_training = "Software and IT Training"
    corporate_learning = "Corporate Learning"


class RFPStatus(str, Enum):
    new = "new"
    flagged_review = "flagged_review"   # Egypt: flag immediately for review
    monitoring = "monitoring"           # Saudi Arabia: monitored pipeline
    qualified = "qualified"
    rejected = "rejected"
    expired = "expired"
    duplicate = "duplicate"


class QualificationFit(str, Enum):
    likely = "likely"
    possible = "possible"
    unlikely = "unlikely"
    unknown = "unknown"


class RFPRecord(BaseModel):
    rfp_id: str
    title: str
    issuing_organization: str
    country: Country  # anything outside EG/SA is rejected before it becomes a record
    deadline: Optional[date] = None
    eligibility_requirements: Optional[str] = None
    training_category: RFPCategory
    submission_link: Optional[str] = None
    publication_date: Optional[date] = None
    available_documents: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    qualification: QualificationFit = QualificationFit.unknown
    qualification_notes: Optional[str] = None
    status: RFPStatus = RFPStatus.new
    dedupe_hash: str = Field(..., description="sha256 of normalized title + organization + deadline")
    source: SourceRef
    last_checked: datetime
    # Sprint 2 additions (optional, backwards compatible)
    relevance_score: float = Field(0.0, ge=0, le=1, description="How strongly the text matches training categories")
    matched_course_ids: list[str] = Field(default_factory=list, description="Academy courses that could deliver it")
    priority: str = Field("normal", examples=["high", "normal"], description="high = Egypt, alert immediately")
    first_seen: Optional[datetime] = None


class RFPRunRequest(BaseModel):
    """Discovery questions from the brief (§10): countries, categories, priority organisations/sectors,
    deadline range. Anything outside Egypt and Saudi Arabia is rejected even if requested."""
    countries: list[Country] = Field(default_factory=lambda: [Country.EG, Country.SA])
    categories: list[RFPCategory] = Field(default_factory=lambda: list(RFPCategory))
    deadline_from: Optional[date] = None
    deadline_to: Optional[date] = None
    source_ids: Optional[list[str]] = None
    keywords: list[str] = Field(default_factory=list,
                                description="Topic filter: when set, only tenders mentioning at least one are kept")
    priority_organizations: list[str] = Field(default_factory=list,
                                              description="Organisations/sectors to rank first, e.g. ['HRDF', 'bank']")
    include_live_sources: bool = Field(False, description="Also fetch the public tender portals in data_sources.yaml")
    notify: bool = Field(True, description="Send alerts for newly discovered relevant RFPs")


class RFPRunResult(BaseModel):
    job_id: str
    found: int
    kept: int
    rejected_country: int
    rejected_irrelevant: int
    expired: int
    duplicates: int
    new_flagged_egypt: list[str] = Field(default_factory=list, description="rfp_ids to alert on now")
    records: list[RFPRecord] = Field(default_factory=list)
    already_known: int = Field(0, description="Seen in an earlier run; refreshed, not alerted again")
    rejected_filters: int = Field(0, description="Outside the requested categories or deadline range")
    new_monitoring_saudi: list[str] = Field(default_factory=list)
    alerts_sent: int = 0
    rejections: list[dict[str, Any]] = Field(default_factory=list, description="title, reason - for auditing")
    source_errors: list[dict[str, Any]] = Field(default_factory=list)


class RFPStatusUpdate(BaseModel):
    status: RFPStatus
    reviewer: str = Field(..., min_length=1)
    comment: Optional[str] = None


class RFPAlert(BaseModel):
    alert_id: str
    rfp_id: str
    country: Country
    priority: str
    message: str
    channels: list[str]
    created_at: datetime


class RFPAskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["Which tenders need cybersecurity training?"])
    country: Optional[Country] = None
    k: int = Field(5, ge=1, le=10)


class RFPAskResponse(BaseModel):
    answer: str
    rfp_ids: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    llm_used: bool = False


# --------------------------------------------------------------------------- #
# Agent 3 - Curriculum & Accreditation Alignment (owner: Member 2)
# --------------------------------------------------------------------------- #
class CurriculumRequest(BaseModel):
    """The brief's discovery questions (§10): course, audience, level, outcomes, hours, industry/framework,
    projects. With course_id, anything not given is taken from the catalogue course (no repeated questions)."""
    course_title: Optional[str] = Field(None, examples=["Data Analysis for Healthcare Quality"])
    target_audience: Optional[str] = None
    skill_level: Optional[SkillLevel] = None
    expected_outcomes: list[str] = Field(default_factory=list)
    total_hours: Optional[int] = Field(None, gt=0, le=1000)
    industry: Optional[str] = None
    department: Optional[str] = None
    accreditation_framework: Optional[str] = Field(None, examples=["SFIA 9", "ISTQB Foundation"])
    country: Country = Country.EG
    required_projects: list[str] = Field(default_factory=list)
    language: Language = Language.en
    course_id: Optional[str] = Field(None, description="Existing catalogue course to (re)design; fills missing fields")

    @model_validator(mode="after")
    def _discovery_answers(self) -> "CurriculumRequest":
        if self.course_id:
            return self
        missing = [name for name, value in (("course_title", self.course_title),
                                            ("target_audience", self.target_audience),
                                            ("skill_level", self.skill_level), ("total_hours", self.total_hours))
                   if not value]
        if missing:
            raise ValueError(f"Missing answers to the discovery questions: {', '.join(missing)} "
                             "(or give course_id of an existing course)")
        return self


class CurriculumModule(BaseModel):
    title: str
    topics: list[str]
    hours: float
    exercises: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    name: str
    type: str = Field(..., examples=["quiz", "assignment", "project", "exam"])
    weight_pct: float


class Curriculum(BaseModel):
    curriculum_id: str
    course_title: str
    objectives: list[str]
    target_audience: str
    prerequisites: list[str]
    learning_outcomes: list[str]
    modules: list[CurriculumModule]
    training_hours: float
    practical_exercises: list[str]
    assessments: list[Assessment]
    capstone_project: str
    metadata: dict[str, Any] = Field(default_factory=dict, description="department, industry, level, country...")
    sources: list[SourceRef] = Field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.draft
    reviewer_comments: list[str] = Field(default_factory=list)
    version: int = 1
    # Sprint 2 additions (optional, backwards compatible)
    course_id: Optional[str] = None
    language: Language = Language.en
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GapItem(BaseModel):
    """One framework competency compared with the curriculum."""
    competency_id: str
    competency: str
    kind: str = Field(..., examples=["topic", "outcome", "structure"])
    status: str = Field(..., examples=["covered", "partial", "missing"])
    evidence: Optional[str] = Field(None, description="Where the curriculum covers it, or why it is missing")
    score: float = Field(0.0, ge=0, le=1)


class GapReport(BaseModel):
    curriculum_id: str
    framework: str
    covered: list[str]
    missing_topics: list[str]
    missing_outcomes: list[str]
    coverage_pct: float = Field(..., ge=0, le=100)
    disclaimer: str = ("This report shows alignment with the selected framework only. "
                       "It is not evidence of official accreditation.")
    sources: list[SourceRef] = Field(default_factory=list)
    # Sprint 2 additions (optional, backwards compatible)
    report_id: Optional[str] = None
    framework_id: Optional[str] = None
    country: Optional[str] = None
    partial: list[str] = Field(default_factory=list, description="Only a semantic match - a reviewer should confirm")
    items: list[GapItem] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    curriculum_version: Optional[int] = None
    created_at: Optional[datetime] = None


class ReviewAction(BaseModel):
    action: ApprovalStatus  # approved | changes_requested | rejected (pending_approval = submit for review)
    reviewer: str
    comment: Optional[str] = None


class GapReportRequest(BaseModel):
    framework: Optional[str] = Field(None, description="Framework id or name; default = the curriculum's framework, "
                                     "else the best match, else the academy quality baseline",
                                     examples=["ISTQB Foundation", "SFIA-9", "EG-NAQAAE-PS"])


class CurriculumRevision(BaseModel):
    """A reviewer's modification request. instructions = plain language ("add a module on Power BI",
    "remove topic X", "add outcome: ...", "set hours to 48"); content = edited curriculum fields."""
    reviewer: str = Field(..., min_length=1)
    instructions: Optional[str] = Field(None, max_length=4000)
    content: Optional[dict[str, Any]] = Field(None, description="Fields of Curriculum to replace, e.g. modules")

    @model_validator(mode="after")
    def _need_change(self) -> "CurriculumRevision":
        if not (self.instructions or self.content):
            raise ValueError("Give instructions or edited content")
        return self


class ProposedCurriculum(BaseModel):
    """A curriculum designed outside REX (e.g. by an instructor), imported to be checked against a framework."""
    course_title: str
    target_audience: str = ""
    objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    modules: list[CurriculumModule] = Field(default_factory=list)
    training_hours: Optional[float] = None
    practical_exercises: list[str] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    capstone_project: str = ""
    department: Optional[str] = None
    skill_level: Optional[SkillLevel] = None


class CurriculumImport(BaseModel):
    course_id: Optional[str] = Field(None, description="Import an existing catalogue course as a curriculum")
    proposal: Optional[ProposedCurriculum] = None
    framework: Optional[str] = None
    country: Country = Country.EG
    language: Language = Language.en

    @model_validator(mode="after")
    def _need_input(self) -> "CurriculumImport":
        if not (self.course_id or self.proposal):
            raise ValueError("Provide course_id or proposal")
        return self


class FrameworkInfo(BaseModel):
    framework_id: str
    name: str
    short_name: str
    country: str
    source_id: str
    url: Optional[str] = None
    status: str = "to_verify"
    exam_prep: bool = False
    competencies: int
    note: Optional[str] = None


class CurriculumResearch(BaseModel):
    """What the agent found before generating: shown to the user and stored in the curriculum's metadata."""
    course_title: str
    department: Optional[str] = None
    framework_id: Optional[str] = None
    framework_reason: Optional[str] = None
    related_courses: list[str] = Field(default_factory=list)
    industry_roles: list[str] = Field(default_factory=list)
    industry_skills: list[str] = Field(default_factory=list)
    reference_modules: list[dict[str, Any]] = Field(default_factory=list, description="title, topics, citation")
    notes: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent 4 - Document Generation (owner: Member 3)
# --------------------------------------------------------------------------- #
class DocumentType(str, Enum):
    course_description = "course_description"
    course_proposal = "course_proposal"
    curriculum_pdf = "curriculum_pdf"
    participant_guide = "participant_guide"
    facilitator_guide = "facilitator_guide"
    presentation_outline = "presentation_outline"
    question_bank = "question_bank"
    assessment_rubric = "assessment_rubric"
    capstone_brief = "capstone_brief"
    outcomes_assessment_plan = "outcomes_assessment_plan"


class DocumentRequest(BaseModel):
    doc_type: DocumentType
    course_id: Optional[str] = None
    curriculum_id: Optional[str] = None
    language: Language = Language.en
    audience: Optional[str] = None
    template_id: Optional[str] = None
    email_to: Optional[list[EmailStr]] = Field(None, description="Only sent after the document is approved")
    instructions: Optional[str] = Field(None, max_length=2000, description="Extra wishes for the LLM, e.g. "
                                        "'focus on the banking sector'")

    @model_validator(mode="after")
    def _need_input(self) -> "DocumentRequest":
        if not (self.course_id or self.curriculum_id):
            raise ValueError("Provide course_id or curriculum_id")
        return self


class DocTable(BaseModel):
    columns: list[str]
    rows: list[list[str]]
    widths: Optional[list[float]] = Field(None, description="Relative column widths, e.g. [1, 3, 1]")

    @model_validator(mode="after")
    def _same_width(self) -> "DocTable":
        if any(len(r) != len(self.columns) for r in self.rows):
            raise ValueError("Every table row needs one cell per column")
        if self.widths and len(self.widths) != len(self.columns):
            raise ValueError("widths needs one value per column")
        return self


class DocSection(BaseModel):
    """One block of a generated document. Rendered in this order: heading, paragraphs, bullets, table, notes."""
    key: str = Field(..., description="Template section key; repeated sections get a suffix, e.g. module_notes:3")
    heading: str
    level: int = Field(1, ge=1, le=2)
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    numbered: bool = False
    table: Optional[DocTable] = None
    notes: list[str] = Field(default_factory=list, description="Speaker / facilitator notes, shown in a box")
    locked: bool = Field(False, description="Facts from approved data (hours, fees, weights): the LLM never "
                                            "rewrites them, only a reviewer can")


class DocumentRecord(BaseModel):
    document_id: str
    doc_type: DocumentType
    title: str
    course_id: Optional[str] = None
    curriculum_id: Optional[str] = None
    language: Language
    file_path: str
    download_url: str
    status: ApprovalStatus = ApprovalStatus.draft
    sources: list[SourceRef] = Field(default_factory=list)
    created_at: datetime
    # Sprint 2 additions (optional, backwards compatible)
    version: int = 1
    template_id: Optional[str] = None
    sections: list[DocSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, description="course facts used, generated_by, notes...")
    reviewer_comments: list[str] = Field(default_factory=list)
    emailed_to: list[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None


class DocumentEdit(BaseModel):
    """A reviewer's changes: edited sections (matched by key; a new key is added at the end), sections to remove,
    a new title, and/or plain-language instructions applied by the LLM (when configured)."""
    reviewer: str = Field(..., min_length=1)
    sections: list[DocSection] = Field(default_factory=list)
    remove_sections: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    instructions: Optional[str] = Field(None, max_length=4000)

    @model_validator(mode="after")
    def _need_change(self) -> "DocumentEdit":
        if not (self.sections or self.remove_sections or self.title or self.instructions):
            raise ValueError("Give edited sections, sections to remove, a title or instructions")
        return self


class DocumentEmail(BaseModel):
    to: list[EmailStr] = Field(..., min_length=1)
    sender: str = Field(..., min_length=1, description="Who sends it (logged)")
    message: Optional[str] = Field(None, max_length=4000)


class DocumentPackRequest(BaseModel):
    """Several documents from one source in one call, e.g. after a curriculum is approved (handoff)."""
    curriculum_id: Optional[str] = None
    course_id: Optional[str] = None
    doc_types: list[DocumentType] = Field(default_factory=lambda: [
        DocumentType.course_description, DocumentType.participant_guide, DocumentType.question_bank])
    language: Optional[Language] = None
    audience: Optional[str] = None

    @model_validator(mode="after")
    def _need_input(self) -> "DocumentPackRequest":
        if not (self.course_id or self.curriculum_id):
            raise ValueError("Provide course_id or curriculum_id")
        return self


class DocumentTypeInfo(BaseModel):
    doc_type: DocumentType
    template_id: str
    title: str
    title_ar: str
    reader: str
    description: str
    sections: list[str]
    templates: list[str] = Field(default_factory=list, description="All template ids for this type")


# --------------------------------------------------------------------------- #
# Agent 5 - AI Marketing (owner: Member 1)
# --------------------------------------------------------------------------- #
class Channel(str, Enum):
    social_caption = "social_caption"
    facebook_instagram_ad = "facebook_instagram_ad"
    linkedin_post = "linkedin_post"
    email = "email"
    whatsapp = "whatsapp"


class CampaignRequest(BaseModel):
    """The brief's discovery questions (§10): course, audience, platforms, objective, budget, offer and CTA.
    Only the course is required; everything else comes from stored data (audience, offer from current promotions,
    CTA from the objective) instead of being asked again."""
    course_id: str
    curriculum_id: Optional[str] = None
    target_audience: Optional[str] = None
    channels: list[Channel] = Field(default_factory=lambda: list(Channel))
    objective: str = Field("registrations", examples=["registrations", "awareness", "webinar sign-ups"])
    offer: Optional[str] = None
    cta: Optional[str] = None
    budget: Optional[float] = Field(None, ge=0)
    variations_per_channel: int = Field(2, ge=1, le=5)
    language: Language = Language.en
    # Sprint 2 additions (optional, backwards compatible)
    document_id: Optional[str] = Field(None, description="Approved document handed over by the Document Agent")
    instructions: Optional[str] = Field(None, max_length=2000, description="Extra wishes for the LLM, e.g. "
                                        "'mention the weekend cohort'")

    @field_validator("course_id")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("channels")
    @classmethod
    def _channels(cls, v: list[Channel]) -> list[Channel]:
        if not v:
            raise ValueError("Choose at least one channel")
        return list(dict.fromkeys(v))


class ContentVariant(BaseModel):
    channel: Channel
    variant_no: int
    headline: Optional[str] = None
    body: str
    cta: str
    # Sprint 2 additions (optional, backwards compatible)
    angle: Optional[str] = Field(None, examples=["outcome", "career", "offer", "expert", "question"])
    hashtags: list[str] = Field(default_factory=list)
    char_count: int = 0
    checks: list[str] = Field(default_factory=list, description="Fact / claim / length checks for the reviewer")
    generated_by: str = "rules"

    @property
    def key(self) -> str:
        return f"{self.channel.value}:{self.variant_no}"


class Campaign(BaseModel):
    campaign_id: str
    course_id: str
    strategy: str
    variants: list[ContentVariant]
    course_facts_used: dict[str, Any] = Field(..., description="Snapshot of approved facts the copy relies on")
    status: ApprovalStatus = ApprovalStatus.pending_approval  # never published without approval
    sources: list[SourceRef] = Field(default_factory=list)
    created_at: datetime
    # Sprint 2 additions (optional, backwards compatible)
    title: Optional[str] = None
    curriculum_id: Optional[str] = None
    document_id: Optional[str] = None
    language: Language = Language.en
    objective: Optional[str] = None
    target_audience: Optional[str] = None
    channels: list[Channel] = Field(default_factory=list)
    headlines: list[str] = Field(default_factory=list, description="Campaign headline options")
    ctas: list[str] = Field(default_factory=list, description="Call-to-action options")
    strategy_plan: dict[str, Any] = Field(default_factory=dict, description="audience, channel mix, budget split, "
                                          "timeline, key messages, KPIs, A/B tests, compliance")
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict, description="generated_by, notes, approvals, "
                                     "publications")
    reviewer_comments: list[str] = Field(default_factory=list)
    download_url: Optional[str] = None
    updated_at: Optional[datetime] = None


class CampaignEdit(BaseModel):
    """A reviewer's changes: edited variants (matched by channel + variant_no; a new number is added), variants to
    remove ("linkedin_post:2"), a new strategy text, and/or plain-language instructions applied by the LLM."""
    reviewer: str = Field(..., min_length=1)
    variants: list[ContentVariant] = Field(default_factory=list)
    remove_variants: list[str] = Field(default_factory=list, examples=[["whatsapp:2"]])
    strategy: Optional[str] = None
    instructions: Optional[str] = Field(None, max_length=4000)

    @model_validator(mode="after")
    def _need_change(self) -> "CampaignEdit":
        if not (self.variants or self.remove_variants or self.strategy or self.instructions):
            raise ValueError("Give edited variants, variants to remove, a strategy or instructions")
        return self


class CampaignPublish(BaseModel):
    """Release an APPROVED campaign for distribution. variant_keys picks the variants to publish (default: every
    variant); email_to sends the e-mail variant to consented recipients (SMTP, else an .eml in the outbox)."""
    publisher: str = Field(..., min_length=1)
    variant_keys: list[str] = Field(default_factory=list, examples=[["linkedin_post:1", "email:2"]])
    email_to: list[EmailStr] = Field(default_factory=list)


class ChannelInfo(BaseModel):
    channel: Channel
    name: str
    name_ar: str
    role: str
    headline_max: int
    body_max: int
    hashtags: int
    paid: bool


# --------------------------------------------------------------------------- #
# Orchestrator and workflow (owner: Member 3)
# --------------------------------------------------------------------------- #
class RouteRequest(BaseModel):
    session_id: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    agent: AgentName
    intent: str
    confidence: float = Field(..., ge=0, le=1)
    result: dict[str, Any]


class HandoffEvent(BaseModel):
    """Emitted when one agent's approved output triggers the next (Curriculum -> Document -> Marketing)."""
    event_id: str
    from_agent: AgentName
    to_agent: AgentName
    payload_type: str = Field(..., examples=["curriculum", "document"])
    payload_id: str
    created_at: datetime


class JobRun(BaseModel):
    job_id: str
    agent: AgentName
    status: JobStatus
    input: dict[str, Any] = Field(default_factory=dict)
    output_ref: Optional[str] = None
    error: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


# ---------------------------------------------------------------- Shared RAG (Sprint 2)
class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["how does self-attention work"])
    collection: str = Field("curriculum_refs", examples=["academy_kb", "curriculum_refs", "rfp_docs"])
    k: int = Field(5, ge=1, le=20)
    where: Optional[dict] = Field(None, description='Metadata filter, e.g. {"department": "AI", "lecture": 8}',
                                  examples=[{"department": "AI"}])
    mmr: bool = Field(False, description="Re-rank for diverse results")
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class RAGHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: str
    metadata: dict


class RAGSearchResponse(BaseModel):
    query: str
    collection: str
    hits: list[RAGHit]
    context: str = Field(..., description="Numbered, cited context ready to put in an LLM prompt")
    sources: list[SourceRef]