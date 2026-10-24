-- REX AI database schema (SQLite for the MVP; PostgreSQL-compatible types).
-- Owner: Member 3 (shared schema/migrations). Agent tables co-owned as noted.
-- List-valued fields are stored as JSON text.

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- Source registry (traceability for every record) - all members
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_sources (
    source_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL CHECK (category IN ('internal','tenders','accreditation_and_competency','curriculum_references')),
    country       TEXT,
    url           TEXT,
    access        TEXT,
    status        TEXT NOT NULL CHECK (status IN ('approved','sample','to_verify')),
    owner         TEXT,
    last_checked  TEXT
);

-- ----------------------------------------------------------------------------
-- Course knowledge base - Member 1
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    course_id               TEXT PRIMARY KEY,
    title_en                TEXT NOT NULL,
    title_ar                TEXT NOT NULL,
    department              TEXT NOT NULL,
    industry                TEXT NOT NULL,          -- JSON list
    skill_level             TEXT NOT NULL CHECK (skill_level IN ('Beginner','Intermediate','Advanced')),
    target_audience         TEXT NOT NULL,
    duration_hours          INTEGER NOT NULL,
    duration_weeks          INTEGER NOT NULL,
    formats                 TEXT NOT NULL,          -- JSON list
    language                TEXT NOT NULL,          -- JSON list
    description_en          TEXT NOT NULL,
    description_ar          TEXT NOT NULL,
    learning_outcomes       TEXT NOT NULL,          -- JSON list
    modules                 TEXT NOT NULL,          -- JSON list
    prerequisites           TEXT NOT NULL,          -- JSON list
    career_opportunities    TEXT NOT NULL,          -- JSON list
    tags                    TEXT NOT NULL,          -- JSON list
    certificate             TEXT,
    accreditation_alignment TEXT,
    country                 TEXT NOT NULL,          -- JSON list
    fee                     REAL,
    currency                TEXT NOT NULL DEFAULT 'EGP',
    installments            INTEGER,
    source_id               TEXT NOT NULL REFERENCES data_sources(source_id),
    last_updated            TEXT,
    is_sample               INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS instructors (
    instructor_id    TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    title            TEXT,
    years_experience INTEGER,
    expertise        TEXT,                          -- JSON list
    bio_en           TEXT,
    bio_ar           TEXT
);

CREATE TABLE IF NOT EXISTS course_instructors (
    course_id     TEXT NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    instructor_id TEXT NOT NULL REFERENCES instructors(instructor_id) ON DELETE CASCADE,
    PRIMARY KEY (course_id, instructor_id)
);

CREATE TABLE IF NOT EXISTS schedules (
    schedule_id TEXT PRIMARY KEY,
    course_id   TEXT NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    start_date  TEXT NOT NULL,
    format      TEXT NOT NULL CHECK (format IN ('online_live','onsite','hybrid')),
    days        TEXT,
    time        TEXT,
    location    TEXT,
    seats_left  INTEGER
);

CREATE TABLE IF NOT EXISTS promotions (
    promo_id     TEXT PRIMARY KEY,
    name_en      TEXT NOT NULL,
    name_ar      TEXT NOT NULL,
    discount_pct REAL NOT NULL,
    applies_to   TEXT NOT NULL,                     -- 'all' or JSON list of course_ids
    condition_en TEXT,
    condition_ar TEXT,
    valid_until  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faqs (
    faq_id   TEXT PRIMARY KEY,
    category TEXT,
    q_en TEXT, a_en TEXT, q_ar TEXT, a_ar TEXT
);

-- REX conversations and consented leads - Member 1
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    language    TEXT NOT NULL DEFAULT 'en',
    profile     TEXT NOT NULL DEFAULT '{}',          -- JSON LeadProfile
    stage       TEXT NOT NULL DEFAULT 'discovery',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content     TEXT NOT NULL,
    sources     TEXT,                               -- JSON list of SourceRef
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leads (
    lead_id               TEXT PRIMARY KEY,
    session_id            TEXT,
    name                  TEXT NOT NULL,
    email                 TEXT,
    phone                 TEXT,
    interested_course_ids TEXT,                     -- JSON list
    profile               TEXT,                     -- JSON LeadProfile
    consent               INTEGER NOT NULL CHECK (consent = 1),   -- no consent, no row
    consent_at            TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    CHECK (email IS NOT NULL OR phone IS NOT NULL)
);

-- ----------------------------------------------------------------------------
-- Tender / RFP research - Member 2
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rfps (
    rfp_id                   TEXT PRIMARY KEY,
    title                    TEXT NOT NULL,
    issuing_organization     TEXT NOT NULL,
    country                  TEXT NOT NULL CHECK (country IN ('EG','SA')),   -- strict geographic rule
    deadline                 TEXT,
    eligibility_requirements TEXT,
    training_category        TEXT NOT NULL,
    submission_link          TEXT,
    publication_date         TEXT,
    available_documents      TEXT,                  -- JSON list
    summary                  TEXT,
    qualification            TEXT NOT NULL DEFAULT 'unknown',
    qualification_notes      TEXT,
    status                   TEXT NOT NULL DEFAULT 'new',
    dedupe_hash              TEXT NOT NULL UNIQUE,  -- duplicate detection
    source_id                TEXT NOT NULL REFERENCES data_sources(source_id),
    source_url               TEXT,
    last_checked             TEXT NOT NULL,
    source_title             TEXT,
    relevance_score          REAL NOT NULL DEFAULT 0,
    matched_course_ids       TEXT,                  -- JSON list
    priority                 TEXT NOT NULL DEFAULT 'normal',
    first_seen               TEXT
);
CREATE INDEX IF NOT EXISTS idx_rfps_country_status ON rfps(country, status);
CREATE INDEX IF NOT EXISTS idx_rfps_deadline ON rfps(deadline);

-- Alerts for newly discovered relevant RFPs (Egypt = high priority, sent immediately)
CREATE TABLE IF NOT EXISTS rfp_alerts (
    alert_id    TEXT PRIMARY KEY,
    rfp_id      TEXT NOT NULL REFERENCES rfps(rfp_id) ON DELETE CASCADE,
    country     TEXT NOT NULL,
    priority    TEXT NOT NULL,
    message     TEXT NOT NULL,
    channels    TEXT NOT NULL,                      -- JSON list: dashboard, log, webhook, email
    created_at  TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Curriculum and accreditation alignment - Member 2
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS accreditation_standards (
    standard_id   TEXT PRIMARY KEY,
    framework     TEXT NOT NULL,                    -- e.g. 'SFIA 9', 'ISTQB Foundation'
    country       TEXT,
    competency    TEXT NOT NULL,
    description   TEXT,
    source_id     TEXT NOT NULL REFERENCES data_sources(source_id),
    last_checked  TEXT
);

CREATE TABLE IF NOT EXISTS curricula (
    curriculum_id   TEXT PRIMARY KEY,
    course_id       TEXT REFERENCES courses(course_id),
    course_title    TEXT NOT NULL,
    request         TEXT NOT NULL,                  -- JSON CurriculumRequest
    content         TEXT NOT NULL,                  -- JSON Curriculum
    department      TEXT, target_audience TEXT, skill_level TEXT, industry TEXT,
    duration_hours  REAL, accreditation_framework TEXT, country TEXT,
    status          TEXT NOT NULL DEFAULT 'draft',
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gap_reports (
    report_id      TEXT PRIMARY KEY,
    curriculum_id  TEXT NOT NULL REFERENCES curricula(curriculum_id) ON DELETE CASCADE,
    framework      TEXT NOT NULL,
    content        TEXT NOT NULL,                   -- JSON GapReport
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type  TEXT NOT NULL CHECK (object_type IN ('curriculum','document','campaign')),
    object_id    TEXT NOT NULL,
    action       TEXT NOT NULL CHECK (action IN ('approved','changes_requested','rejected')),
    reviewer     TEXT NOT NULL,
    comment      TEXT,
    created_at   TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Generated documents - Member 3
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    doc_type       TEXT NOT NULL,
    title          TEXT NOT NULL,
    course_id      TEXT REFERENCES courses(course_id),
    curriculum_id  TEXT REFERENCES curricula(curriculum_id),
    language       TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'draft',
    sources        TEXT,                            -- JSON list
    emailed_to     TEXT,                            -- JSON list, only after approval
    created_at     TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Marketing campaigns - Member 1
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id       TEXT PRIMARY KEY,
    course_id         TEXT NOT NULL REFERENCES courses(course_id),
    curriculum_id     TEXT REFERENCES curricula(curriculum_id),
    request           TEXT NOT NULL,                -- JSON CampaignRequest
    strategy          TEXT,
    course_facts_used TEXT NOT NULL,                -- JSON snapshot of approved facts
    status            TEXT NOT NULL DEFAULT 'pending_approval',
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_variants (
    variant_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id  TEXT NOT NULL REFERENCES campaigns(campaign_id) ON DELETE CASCADE,
    channel      TEXT NOT NULL,
    variant_no   INTEGER NOT NULL,
    headline     TEXT,
    body         TEXT NOT NULL,
    cta          TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Orchestration, jobs and logging - Member 3
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_runs (
    job_id       TEXT PRIMARY KEY,
    agent        TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
    input        TEXT,
    output_ref   TEXT,
    error        TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS handoff_events (
    event_id     TEXT PRIMARY KEY,
    from_agent   TEXT NOT NULL,
    to_agent     TEXT NOT NULL,
    payload_type TEXT NOT NULL,
    payload_id   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT NOT NULL,
    agent      TEXT,
    event      TEXT NOT NULL,
    detail     TEXT,
    created_at TEXT NOT NULL
);
