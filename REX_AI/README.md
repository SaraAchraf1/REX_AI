# REX AI — Multi-Agent Academy Assistant

Graduation project (Oct 10 – Nov 5, 2026). REX AI automates academy operations with five
specialized agents behind one orchestrator:

| # | Agent | Owner | Status |
| --- | --- | --- | --- |
| 1 | REX Course Assistant (recommendation chatbot) | Member 1 | **Sprint 1 prototype working** |
| 2 | Tender / RFP Research (Egypt + Saudi Arabia only) | Member 2 | **Sprint 2 working** (see below) |
| 3 | Curriculum & Accreditation Alignment | Member 2 | **Sprint 2 working** (see below) |
| 4 | Document Generation (PDF) | Member 3 | Contract defined, Sprint 2 |
| 5 | AI Marketing | Member 1 | Contract defined, Sprint 2 |
| — | Orchestrator, API, app shell, integration | Member 3 | Keyword router + API live |

> **Data notice:** everything in `data/` is **sample data** created by the team for the prototype
> (status `sample` in `data/data_sources.yaml`). Replace it with the academy's approved catalogue,
> fees and schedules before any real use. REX labels sample data and never invents course details.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_kb.py                             # validate data + export the knowledge base
uvicorn rex_ai.api.main:app --reload                   # API on http://localhost:8000 (docs at /docs)
streamlit run frontend/streamlit_app.py                # chat UI on http://localhost:8501
pytest -q                                              # 25 tests
```

The UI calls the API when it is running and falls back to running the assistant in-process otherwise.
Set `OPENAI_API_KEY` (see `.env.example`) to have an LLM phrase replies; recommendations, prices and
dates still come only from the data files. Without a key REX uses its deterministic responder.

## What REX Assistant does (Sprint 1)

- Chats in **Arabic or English** (auto-detected, Egyptian dialect understood) with right-to-left display.
- Asks the brief's discovery questions one at a time — interest, background, level, format — and skips
  anything the lead already said (budget and weekly hours are picked up if mentioned).
- Recommends up to 3 courses with a match score and the **reasons** each one fits.
- Answers questions about content, outcomes, duration, format, fees, offers, prerequisites, instructors,
  schedules, certificates and admission, plus FAQs (refunds, payment, recordings, laptops, jobs...).
- Follows references like "the second one" or "who teaches it?".
- Says plainly when a topic is not offered, instead of inventing a course.
- Cites the source record behind every answer (`sources` field / "Sources" expander).
- Saves a lead only with an explicit consent checkbox and an email or phone.

## Repository layout

```
data/                       sample academy data + data_sources.yaml (all approved/candidate sources)
  curriculum/               frameworks, industry skills and reference syllabi for the Curriculum Agent
  courses.json  instructors.json  pricing_promotions.json  schedules.json  faqs.json  admission_requirements.json
docs/interfaces.md          data model, endpoints, agent handoff contract  <- read before coding
rex_ai/
  config.py                 environment settings
  core/schemas.py           Pydantic contracts for ALL agents (the integration interface)
  core/schema.sql           database schema (SQLite now, PostgreSQL-compatible)
  core/db.py                persistence helpers, seeding, logging
  agents/rex/               REX Assistant: knowledge base, retriever, profile extractor, recommender, LLM, prompts
  agents/rfp/               RFP & Tender Research Agent
  agents/curriculum/        Curriculum & Accreditation Alignment Agent
  api/main.py               FastAPI app (+ orchestrator stub)
frontend/streamlit_app.py   chat UI
frontend/pages/              RFP Research and Curriculum pages of the same app
scripts/build_kb.py         validates data, writes data/kb/kb_chunks.jsonl for a vector store
tests/                      knowledge base, recommender and API tests
```

## Team workflow

- One branch per task ID, e.g. `feature/S2-04-rfp-country-filter`; merge to `main` only through a
  reviewed Pull Request with tests passing.
- Commit messages describe the change: `feat(rex): follow up on "the second one"`.
- Contract changes (`rex_ai/core/schemas.py`, `schema.sql`) need review from every owner who uses them.
- The Instructor is added as a repository collaborator.

## Next (Sprint 2, Oct 19–27)

- M1: Marketing Agent; swap BM25 for Chroma if needed; more bilingual test cases.
- M2: RFP collector + EG/SA filter + dedupe; Curriculum generator + gap report.
- M3: Document Agent (3 PDF templates); orchestrator v2 with job runs; app shell.

## Shared RAG pipeline (Sprint 2)

`rex_ai/core/rag.py` is the one retrieval layer every agent uses: PDF / Word / Excel / CSV / TXT loaders,
document-aware → semantic → recursive chunking, multilingual embeddings and a persistent Chroma store.

```python
from rex_ai.core.rag import get_rag

rag = get_rag("curriculum_refs")
hits = rag.search("how does self-attention work", k=5, where={"department": "AI"}, mmr=True)
context = rag.build_context(hits)      # numbered, cited context for the LLM prompt
sources = rag.to_source_refs(hits)     # SourceRef list for the response
```

```bash
python scripts/ingest_mit_ocw.py                                   # MIT 6.7960 lecture notes -> curriculum_refs
python -m rex_ai.core.rag ingest path/to/folder --collection rex_kb --source-id SRC-ACAD-TEMPLATES
python -m rex_ai.core.rag query "graph neural networks" --collection curriculum_refs
```

**REX Assistant** now retrieves FAQs and courses through this pipeline (collection `academy_kb`, indexed
automatically on startup and rebuilt when `data/*.json` change). `REX_RETRIEVER=tfidf` switches back to the
Sprint 1 retriever. Check and tune its match thresholds with `python scripts/eval_retriever.py`.
Other agents and the UI can query any collection over HTTP: `POST /rag/search` (see `/docs`).

Embeddings default to `intfloat/multilingual-e5-small` (downloaded on first run). Set `REX_RAG_EMBEDDINGS=hash`
to run fully offline, or `openai` to use the OpenAI API. A collection must be queried with the same embedder it
was built with; `python -m rex_ai.core.rag reset --collection <name>` rebuilds it. If you already have a local
`rex_ai.db`, delete it once so the new `curriculum_references` source category is accepted.

## Importing the academy course catalogue

REX recommends only the courses in `data/courses.json` (sample data until the academy sends its real
catalogue). `scripts/import_catalog.py` turns a catalogue in PDF, Word or Excel into that data, plus fees and
cohorts. The format is described at the top of the script; `data/catalog/examples/sample_academy_catalog.pdf`
is a 5-course example; like every file under `data/catalog/`, it is imported automatically.

### Adding new data while REX is running (nothing is overwritten)

| Way | What happens |
| --- | --- |
| Drop a catalogue into `data/catalog/` (sub-folders too) | Courses merged into `data/*.json` on the next message |
| Drop any other PDF / Word / Excel / CSV / TXT into `data/documents/` | Indexed; REX answers questions from it and cites the file |
| `POST /data/upload` (or "Add academy data" in the Streamlit sidebar) | Saved to the right folder, then loaded as above |
| `POST /data/courses` with JSON course records | Merged like a catalogue file |
| `POST /data/reload`, `GET /data/status` | Import now / list what REX has loaded |

New course IDs are added, an existing ID is updated, no course is ever removed, and the previous data files are
backed up to `data/backup/` before every merge. An upload with an existing file name is saved as `name-2.pdf`.
`REX_DOC_MIN_SCORE` tunes how close a document passage must be before REX answers from it.

```bash
python scripts/import_catalog.py data/catalog/sample_academy_catalog.pdf --dry-run    # check only
python scripts/import_catalog.py data/catalog/sample_academy_catalog.pdf --replace    # catalogue = all courses
```

The old files are backed up to `data/backup/<timestamp>/` first. Restart the API afterwards; the RAG index
`academy_kb` rebuilds automatically.
## Agent 2 — Educational RFP & Tender Research (Sprint 2)

Finds training-related tenders in **Egypt and Saudi Arabia only**, and turns each one into a reviewed,
traceable pipeline record. Code: `rex_ai/agents/rfp/`, dashboard: the **RFP Research** page of the Streamlit app.

```
sources ──> country filter ──> relevance ──> expiry / filters ──> duplicates ──> qualification ──> summary
(feeds, documents, portals)  (EG/SA only)  (6 categories)       (exact + fuzzy)   (courses + profile)
        ──> rfps table (searchable dashboard, CSV) ──> RAG index (rfp_docs) ──> alerts (EG now, SA digest)
```

| Brief requirement | Where |
| --- | --- |
| Search portals / approved sources | `sources.py`: JSON/CSV/Excel exports in `data/rfp/feeds/`, tender documents in `data/rfp/documents/`, live portals from `data_sources.yaml` (`--live`, robots.txt respected) |
| Training categories | `taxonomy.py` (English + Arabic terms), `extract.classify` |
| Extract title, organisation, country, deadline, eligibility, category, link, publication date, documents | `extract.extract_fields` (label rules, EN/AR), LLM over RAG passages when `OPENAI_API_KEY` is set; Hijri dates converted |
| Strict Egypt / Saudi rule | `extract.detect_country`: explicit country field wins; unknown or other country = rejected with the reason |
| Filter unrelated / expired | supply, works and catering tenders rejected; past deadlines rejected; stored ones auto-marked `expired` |
| Duplicates | `dedupe.py`: sha256 key + fuzzy title/organisation match across portals; gaps filled from the duplicate |
| AI summary | `agent.summarize`: LLM grounded in the tender's own documents (RAG), template otherwise |
| May the academy qualify? | `qualify.py`: matching catalogue courses (tags + REX's `academy_kb` RAG index) and eligibility rules vs `data/rfp/academy_profile.yaml` → likely / possible / unlikely, with notes |
| Searchable dashboard / sheet | `GET /rfp?q=&country=&status=`, `GET /rfp/export.csv`, Streamlit page |
| Alerts | `notify.py`: Egypt = one immediate high-priority alert each (status `flagged_review`); Saudi = one digest (status `monitoring`). Stored for the dashboard; also webhook / e-mail if configured |
| Traceability | every record has `source` (registry `source_id`), the qualification cites course ids and the profile |

```bash
REX_RFP_TODAY=2026-10-24 python scripts/run_rfp.py -v        # sample data is fixed, so pin "today" for the demo
python scripts/run_rfp.py --country EG --days 45 --keyword "ISO 27001"
python scripts/run_rfp.py --ask "which tenders need cybersecurity training?"
python scripts/run_rfp.py --export rfps.csv
pytest -q tests/test_rfp_agent.py                           # 55 test cases covering every rule above
```

The sample feed is made up on purpose to exercise every rule: of 21 notices, 5 are outside EG/SA (UAE, France,
USA, Kenya, Jordan), 2 are not training (air conditioning, catering), 1 is expired and 3 are duplicates
(including a tender document that matches a feed notice), leaving 10. Replace it with real portal exports:
drop a CSV/Excel export (Arabic column names such as `اسم المنافسة`, `الجهة`, `آخر موعد لتقديم العروض` are
understood) into `data/rfp/feeds/`, or upload it with `POST /rfp/upload`.

## Agent 3 — Curriculum Research & Accreditation Alignment (Sprint 2)

Designs a structured curriculum from the discovery answers, checks it against accreditation / competency
frameworks, exports a PDF and takes it through review. Code: `rex_ai/agents/curriculum/`, UI: the **Curriculum**
page of the Streamlit app. Works without an LLM; with `OPENAI_API_KEY` set, the LLM refines the rule-based draft
and the same checks are applied to its output.

```
discovery answers ──> research ─────────────────────────────> curriculum ──> finalize ──> alignment reports ──> draft v1
 (or a catalogue      catalogue courses · reference syllabi      (modules,     hours, weights,  framework + country
  course id)          (RAG, curriculum_refs, metadata filters)    outcomes...)  requested items, checklist + academy
                      industry skills · framework                               no accreditation baseline
                                                                                claims
review: submit ──> approved (HandoffEvent curriculum → document) | changes_requested ──> revise ──> v2 (draft, re-aligned)
```

| Brief requirement | Where |
| --- | --- |
| FR 18 Research with approved, credible sources | `research.py`: catalogue courses, reference syllabi in `data/curriculum/references/` indexed in the shared RAG collection `curriculum_refs` (with the MIT OCW notes), framework summaries; every source is listed in `Curriculum.sources` and registered in `data_sources.yaml` |
| FR 19 Industry skill requirements | `data/curriculum/industry_skills.yaml` (roles, skills, tools per department; sample from EG/SA job postings, ESCO/O*NET to verify) |
| FR 20-21 Structured curriculum: objectives, audience, prerequisites, outcomes, modules & topics, hours, exercises, assessments, capstone | `generator.py`: modules from the catalogue course or the best reference syllabus (plus a second one for cross-domain titles), hours always add up, weights total 100%, outcomes cover the knowledge / intellectual / practical / transferable domains |
| FR 22 Compare with accreditation or competency standards | `alignment.py` + `data/curriculum/frameworks.yaml`: ISTQB CTFL, AWS CLF-C02, PMP ECO, SFIA 9 (by department), Egypt NAQAAE program-specification checklist, Saudi NQF-KSA domains, academy quality baseline |
| FR 23 Missing topics / outcomes | each competency is covered (keyword evidence), partial (semantic match, reviewer confirms) or missing; `GapReport.missing_topics`, `missing_outcomes`, `recommendations` |
| FR 24 Professional PDF | `pdf.py` (reportlab): metadata table, outcomes, module table, assessment plan, capstone, alignment report, sources, review history; Arabic shaped right-to-left; DRAFT watermark until approved |
| FR 25 Review and modifications before approval | `POST /curriculum/{id}/review` and `/revise`: plain-language instructions ("add module ...", "remove topic ...", "add outcome: ...", "set hours to 48") or edited fields → new version; nothing is overwritten (`curriculum_versions`) |
| Metadata table (department, audience, level, industry, duration, outcomes, prerequisites, framework, country, source & update date) | stored in `Curriculum.metadata` and the `curricula` columns; `GET /curriculum?department=&skill_level=&country=&framework=&industry=&audience=&min_hours=&max_hours=&q=`; reference chunks carry the same fields as RAG metadata |
| Never claim accreditation | `generator.guard_claims` rewrites "accredited / certified by ..." in any generated, imported or edited text; every report carries the disclaimer; frameworks are labelled "alignment only" |
| Use stored information instead of asking again | `course_id` fills title, audience, level, hours, outcomes from the catalogue; the chat router asks only the missing discovery questions |

```bash
python scripts/run_curriculum.py --title "Data Analysis for Healthcare Quality" --audience "Hospital quality officers" \
    --level Beginner --hours 40 --industry Healthcare --outcome "Build a KPI dashboard for patient safety" --pdf
python scripts/run_curriculum.py --import-course SW-201 --framework ISTQB     # gap analysis of an existing course
python scripts/run_curriculum.py --title "إدارة المشاريع الاحترافية" --audience "مديرو المشاريع" \
    --level Intermediate --hours 35 --language ar --country SA --pdf       # Arabic curriculum + PDF
python scripts/run_curriculum.py --revise CUR-XXXX --reviewer "Dr. Mona" --instructions "add module Power BI; set hours to 48"
python scripts/run_curriculum.py --approve CUR-XXXX --reviewer Dean
pytest -q tests/test_curriculum_agent.py                                     # 52 test cases
```

Endpoints: `GET /curriculum/frameworks`, `POST /curriculum/research` (preview), `POST /curriculum`,
`POST /curriculum/import` (catalogue course or an instructor's proposal), `GET /curriculum` (metadata filters),
`GET /curriculum/{id}`, `POST /curriculum/{id}/gap-report`, `GET /curriculum/{id}/gap-reports`,
`POST /curriculum/{id}/review`, `POST /curriculum/{id}/revise`, `GET /curriculum/{id}/history`,
`GET /curriculum/{id}/pdf`. `POST /route` sends "design a curriculum for ..." messages here.

**Real model + LLM:** settings are read from `.env` (copy `.env.example`). Set `OPENAI_API_KEY` and
`REX_RAG_EMBEDDINGS=sentence-transformers` (multilingual-e5-small, Arabic + English). A `curriculum_refs` collection
built earlier with another embedder is rebuilt automatically (re-run `scripts/ingest_mit_ocw.py` for the MIT notes).
With e5 the alignment relies on keyword and shared-word evidence; an embedding match alone counts only when it
clearly stands out (score ≥ 0.85 and ≥ 0.045 above that competency's average), because e5 scores every pair 0.79-0.87.

Framework summaries, reference syllabi and industry skills are team-made (`sample` / `to_verify` in
`data_sources.yaml`): check them against the official documents and add the academy's own references to
`data/curriculum/references/` (with a line in `manifest.yaml`) before real use. PDFs need a TTF font with Arabic
glyphs; Arial/Tahoma (Windows), DejaVu Sans or Noto Naskh (Linux) are found automatically, or set `REX_PDF_FONT`.
