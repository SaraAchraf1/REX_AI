# REX AI — Data Structure & Integration Interfaces (Sprint 1, S1-07)

This is the contract all three members build against. The source of truth in code is
`rex_ai/core/schemas.py` (Pydantic models) and `rex_ai/core/schema.sql` (database). The FastAPI
docs at `http://localhost:8000/docs` are generated from the same models.

**Change rule:** a field is added, renamed or removed only through a Pull Request reviewed by every
owner whose agent uses that model (see the Owner column).

## 1. Shared rules

| Rule | How it is enforced |
| --- | --- |
| Every answer or record is traceable to its data | `SourceRef {source_id, title, url, record_id, retrieved_at}` on every response; `source_id` comes from `data/data_sources.yaml` |
| Only Egypt and Saudi Arabia | `Country` enum = `EG`, `SA`; the `rfps.country` column has a CHECK constraint |
| No lead stored without consent | `LeadCreate` validator + `leads.consent CHECK (consent = 1)` |
| Nothing is published or sent without a human | `ApprovalStatus`: `draft → pending_approval → approved / changes_requested / rejected`; campaigns default to `pending_approval` |
| No accreditation claim | `GapReport.disclaimer` and `Course.accreditation_alignment` describe alignment only |
| One error shape | `ErrorResponse {code, message, details}` for every 4xx/5xx |
| Bilingual | `Language` enum = `en`, `ar`; auto-detected when the request omits it |

## 2. Entities

| Model (schemas.py) | Table (schema.sql) | Owner | Key fields |
| --- | --- | --- | --- |
| `Course` | `courses` | M1 | course_id, titles (en/ar), department, industry, skill_level, target_audience, duration_hours/weeks, formats, outcomes, modules, prerequisites, careers, accreditation_alignment, country, source_id, last_updated |
| `Instructor` | `instructors`, `course_instructors` | M1 | instructor_id, name, title, years_experience, expertise, bio_en/ar |
| `Schedule` | `schedules` | M1 | schedule_id, course_id, start_date, format, days, time, location, seats_left |
| `Promotion` | `promotions` | M1 | promo_id, discount_pct, applies_to, condition, valid_until |
| `FAQ` | `faqs` | M1 | faq_id, category, q/a in en and ar, keywords |
| `LeadProfile` | `chat_sessions.profile` | M1 | background, education_level, experience_years, interests, career_goal, skill_level, preferred_format, budget, weekly_hours |
| `LeadCreate` / `Lead` | `leads` | M1 | name, email or phone, interested_course_ids, profile, consent (must be true) |
| `RFPRecord` | `rfps`, `rfp_alerts` | M2 | title, issuing_organization, country, deadline, eligibility, training_category, submission_link, publication_date, documents, summary, qualification, status, dedupe_hash, source; Sprint 2 adds relevance_score, matched_course_ids, priority, first_seen (existing databases get the columns automatically in `db.init_db`) |
| `CurriculumRequest`, `Curriculum`, `GapReport`, `GapItem`, `ReviewAction`, `CurriculumRevision`, `CurriculumImport` | `curricula`, `curriculum_versions`, `gap_reports`, `reviews`, `accreditation_standards`, `handoff_events` | M2 | objectives, audience, prerequisites, outcomes, modules, hours, exercises, assessments, capstone, metadata (brief table); coverage_pct, covered / partial / missing items with evidence, missing topics/outcomes, recommendations. Sprint 2 adds optional fields only: `course_id` on the request (the other answers become optional with it), `course_id`, `language`, timestamps on `Curriculum`; `report_id`, `items`, `partial`, `recommendations` on `GapReport` |
| `DocumentRequest` / `DocumentRecord` | `documents` | M3 | doc_type (10 types), course_id or curriculum_id, language, file_path, download_url, status |
| `CampaignRequest` / `Campaign` | `campaigns`, `campaign_variants` | M1 | channels, objective, offer, cta, variants, course_facts_used, status |
| `JobRun`, `HandoffEvent` | `job_runs`, `handoff_events`, `app_logs` | M3 | agent, status, output_ref, error; from_agent → to_agent with payload_id |

## 3. Endpoints

| Method + path | Agent | Request → Response | Status |
| --- | --- | --- | --- |
| `GET /health` | Platform | — → `{status, courses, data_status, llm_enabled}` | Built |
| `POST /chat` | REX Assistant | `ChatRequest` → `ChatResponse` | Built |
| `GET /courses?department=&skill_level=` | REX Assistant | — → `Course[]` | Built |
| `GET /courses/{course_id}` | REX Assistant | — → `Course` (404 if unknown) | Built |
| `POST /leads` | REX Assistant | `LeadCreate` → `Lead` (201) | Built |
| `POST /rag/search` | Shared RAG (all agents) | `RAGSearchRequest` → `RAGSearchResponse` (hits, cited context, sources) | Built (Sprint 2) |
| `GET /rag/collections/{name}` | Shared RAG | — → `{chunks, documents, embedder, chunk_methods}` | Built (Sprint 2) |
| `POST /route` | Orchestrator | `RouteRequest` → `RouteResponse` | Keyword router (Sprint 1) |
| `POST /rfp/run` | RFP Research | `RFPRunRequest` → `RFPRunResult` (counts, rejections with reasons, new EG/SA ids) | Built (Sprint 2) |
| `GET /rfp?country=&status=&category=&qualification=&q=&open_only=` | RFP Research | — → `RFPRecord[]` (Egypt first, then deadline) | Built (Sprint 2) |
| `GET /rfp/{rfp_id}` | RFP Research | — → `RFPRecord` (404 if unknown) | Built (Sprint 2) |
| `PATCH /rfp/{rfp_id}/status` | RFP Research | `RFPStatusUpdate {status, reviewer, comment}` → `RFPRecord` | Built (Sprint 2) |
| `GET /rfp/export.csv` | RFP Research | — → CSV (UTF-8 BOM, opens in Excel / Google Sheets) | Built (Sprint 2) |
| `GET /rfp/alerts` | RFP Research | — → `RFPAlert[]` | Built (Sprint 2) |
| `POST /rfp/ask` | RFP Research | `RFPAskRequest` → `RFPAskResponse` (RAG over records + tender documents, cited) | Built (Sprint 2) |
| `POST /rfp/upload` | RFP Research | file → `{file, kind: document/feed}` | Built (Sprint 2) |
| `GET /curriculum/frameworks` | Curriculum | — → `FrameworkInfo[]` | Built (Sprint 2) |
| `POST /curriculum/research` | Curriculum | `CurriculumRequest` → `CurriculumResearch` (preview, nothing stored) | Built (Sprint 2) |
| `POST /curriculum` | Curriculum | `CurriculumRequest` → `Curriculum` (201, draft v1, alignment reports stored) | Built (Sprint 2) |
| `POST /curriculum/import` | Curriculum | `CurriculumImport {course_id or proposal, framework}` → `Curriculum` | Built (Sprint 2) |
| `GET /curriculum?department=&skill_level=&country=&status=&framework=&industry=&audience=&min_hours=&max_hours=&q=` | Curriculum | — → `Curriculum[]` (metadata filtering) | Built (Sprint 2) |
| `GET /curriculum/{id}` | Curriculum | — → `Curriculum` (404 if unknown) | Built (Sprint 2) |
| `POST /curriculum/{id}/gap-report` | Curriculum | `GapReportRequest {framework}` → `GapReport` (404 unknown framework) | Built (Sprint 2) |
| `GET /curriculum/{id}/gap-reports` | Curriculum | — → `GapReport[]` (latest per framework) | Built (Sprint 2) |
| `POST /curriculum/{id}/review` | Curriculum | `ReviewAction` → `Curriculum` (409 on an invalid transition; approved → `HandoffEvent` to document) | Built (Sprint 2) |
| `POST /curriculum/{id}/revise` | Curriculum | `CurriculumRevision {reviewer, instructions or content}` → `Curriculum` (version + 1, draft) | Built (Sprint 2) |
| `GET /curriculum/{id}/history` | Curriculum | — → `{versions, reviews, gap_reports, handoffs}` | Built (Sprint 2) |
| `GET /curriculum/{id}/pdf?include_alignment=` | Curriculum | — → PDF file (DRAFT watermark until approved) | Built (Sprint 2) |
| `POST /documents` | Document | `DocumentRequest` → `DocumentRecord` | Sprint 2 (M3) |
| `GET /documents/{id}/download` | Document | — → PDF file | Sprint 2 (M3) |
| `POST /campaigns` | Marketing | `CampaignRequest` → `Campaign` | Sprint 2 (M1) |
| `POST /campaigns/{id}/review` | Marketing | `ReviewAction` → `Campaign` | Sprint 2 (M1) |

### Example: `POST /chat`

```json
// request
{"session_id": "demo-1", "message": "I'm an accounting graduate, beginner, want data analysis online"}

// response (trimmed)
{
  "session_id": "demo-1",
  "language": "en",
  "stage": "recommendation",
  "intent": "recommendation",
  "reply": "Based on what you told me, these courses fit you best: ...",
  "profile": {"education_level": "graduate", "interests": ["data"], "skill_level": "Beginner", "preferred_format": "online_live"},
  "recommendations": [
    {"course_id": "DS-101", "title": "Data Analysis with Python and Power BI", "score": 0.9,
     "reasons": ["Matches your interest in data analysis", "Pitched at beginner level"],
     "fee": 10500, "currency": "EGP", "next_start": "2026-11-22", "format_options": ["online_live", "onsite", "hybrid"]}
  ],
  "sources": [{"source_id": "SRC-ACAD-COURSES", "title": "Academy course catalogue", "record_id": "DS-101"}],
  "offer_lead_capture": true,
  "llm_used": false
}
```

### Example: `POST /leads`

```json
{"session_id": "demo-1", "name": "Sara Ahmed", "email": "sara@example.com",
 "interested_course_ids": ["DS-101"], "consent": true}
```
`consent: false`, or no email and no phone → `422 {"code": "validation_error", ...}`.

## 4. Connected workflow (agent-to-agent)

```
Curriculum (M2) --approved--> HandoffEvent --> Document (M3) --approved--> HandoffEvent --> Marketing (M1) --> Admin approval
```

1. Curriculum Agent saves a `Curriculum` with `status = draft`; reviewer sends `ReviewAction`.
2. On `approved`, the orchestrator writes `HandoffEvent {from: curriculum, to: document, payload_type: "curriculum", payload_id}`.
3. Document Agent builds the requested PDFs from that `curriculum_id` and stores `DocumentRecord`s.
4. On document approval, a second `HandoffEvent` starts the Marketing Agent with `course_id` + `curriculum_id`.
   Marketing copy may only use facts listed in `Campaign.course_facts_used` (snapshot of approved data).
5. Every agent call writes a `JobRun` (queued → running → succeeded/failed) for status display and logs.

## 5. Knowledge base format

`python scripts/build_kb.py` validates the data files and writes `data/kb/kb_chunks.jsonl`, one line per chunk:

```json
{"chunk_id": "DS-101#overview", "text": "...", "metadata": {"type": "course", "course_id": "DS-101",
 "department": "Data", "skill_level": "Beginner", "target_audience": "...", "industry": ["..."],
 "duration_hours": 48, "country": ["EG", "SA"], "accreditation_alignment": "...",
 "source_id": "SRC-ACAD-COURSES", "last_updated": "2026-10-17"}}
```

Metadata fields match the brief's knowledge-base table, so the retriever can filter by department,
level, audience, industry, duration, framework and country. The prototype uses a bilingual BM25
retriever (no model download); the `Retriever.search(query, k, filters)` interface stays the same
if Sprint 2 switches to Chroma or FAISS.