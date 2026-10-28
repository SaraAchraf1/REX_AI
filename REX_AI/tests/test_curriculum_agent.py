"""Agent 3 - Curriculum Research & Accreditation Alignment Agent (owner: Member 2).
Run: pytest -q tests/test_curriculum_agent.py

Covers the brief's FR 18-25: research (catalogue, RAG references with metadata filters, industry skills),
structured curriculum, framework comparison and gaps, PDF, review / modification / approval, plus the rule that
the agent never claims a course is accredited. Runs offline (hash embedder, no LLM) - see conftest.py.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rex_ai.agents.curriculum import frameworks as fw
from rex_ai.agents.curriculum.agent import CurriculumAgent, NotFound
from rex_ai.agents.curriculum.alignment import is_measurable, outcome_domain
from rex_ai.agents.curriculum.generator import guard_claims, guard_text, rescale_hours, split_hours
from rex_ai.agents.curriculum.research import ensure_references_indexed, refs_rag
from rex_ai.agents.curriculum.revise import apply_instructions
from rex_ai.core import db
from rex_ai.core.schemas import (ApprovalStatus, CurriculumImport, CurriculumRequest, CurriculumRevision,
                                 ProposedCurriculum, ReviewAction)

HEALTH = dict(course_title="Data Analysis for Healthcare Quality", target_audience="Hospital quality officers and nurses",
              skill_level="Beginner", total_hours=40, industry="Healthcare", country="EG",
              expected_outcomes=["Build a KPI dashboard for patient safety indicators"],
              required_projects=["Patient-safety KPI dashboard for a hospital ward", "Root cause analysis report"])
TESTING = dict(course_title="Software Testing Fundamentals", target_audience="Junior developers",
               skill_level="Intermediate", total_hours=30, accreditation_framework="ISTQB Foundation")


@pytest.fixture(scope="module")
def agent():
    return CurriculumAgent()


@pytest.fixture(scope="module")
def health(agent):
    return agent.generate(CurriculumRequest(**HEALTH))


@pytest.fixture(scope="module")
def testing(agent):
    return agent.generate(CurriculumRequest(**TESTING))


# =========================================================================== discovery questions (brief §10)
def test_request_needs_the_discovery_answers():
    with pytest.raises(ValidationError) as e:
        CurriculumRequest(course_title="Power BI")
    assert "target_audience" in str(e.value) and "total_hours" in str(e.value)


def test_existing_course_fills_missing_answers(agent):
    filled = agent.fill_request(CurriculumRequest(course_id="ds-101"))
    course = agent.kb.courses["DS-101"]
    assert filled["course_title"] == course.title_en and filled["total_hours"] == course.duration_hours
    assert filled["target_audience"] == course.target_audience and filled["expected_outcomes"] == course.learning_outcomes


def test_unknown_course_id(agent):
    with pytest.raises(NotFound):
        agent.generate(CurriculumRequest(course_id="XX-999"))


# =========================================================================== research (FR 18-19) + RAG
def test_reference_syllabi_are_indexed_with_metadata():
    rag = refs_rag()
    ensure_references_indexed(rag, force=False)
    hits = rag.search("software testing techniques", k=5, where={"department": "Software"})
    assert hits and all(h.metadata["department"] == "Software" for h in hits)
    assert {"skill_level", "target_audience", "accreditation_framework", "source_id", "update_date"} <= set(hits[0].metadata)


def test_research_combines_catalogue_references_skills_and_framework(agent):
    r = agent.research(CurriculumRequest(**HEALTH))
    assert r.department == "Healthcare" and r.framework_id == "EG-NAQAAE-PS"
    assert r.industry_skills and r.industry_roles
    titles = [m["title"] for m in r.reference_modules]
    assert "Patient safety" in titles and all(m["citation"] for m in r.reference_modules)
    assert {"SRC-CUR-REFS", "SRC-CUR-INDUSTRY", "SRC-EG-NAQAAE"} <= {s.source_id for s in r.sources}


@pytest.mark.parametrize("title, department, country, expected", [
    ("ISTQB Foundation Exam Prep", "Software", "EG", "ISTQB-CTFL-4"),
    ("AWS Cloud Practitioner Bootcamp", "Cloud", "SA", "AWS-CLF-C02"),
    ("PMP Exam Preparation", "Business", "EG", "PMI-PMP-ECO-2021"),
    ("Data Visualisation with Power BI", "Data", "EG", "SFIA-9"),
    ("Healthcare Quality Basics", "Healthcare", "EG", "EG-NAQAAE-PS"),
    ("Healthcare Quality Basics", "Healthcare", "SA", "SA-NQF-DOMAINS"),
])
def test_framework_suggestion(title, department, country, expected):
    assert fw.suggest_framework(title, department, country)[0].framework_id == expected


def test_find_framework_by_name_or_alias():
    assert fw.find_framework("istqb").framework_id == "ISTQB-CTFL-4"
    assert fw.find_framework("SFIA").framework_id == "SFIA-9"
    assert fw.find_framework("NAQAAE").framework_id == "EG-NAQAAE-PS"
    assert fw.find_framework("ai") is None and fw.find_framework("Unknown body") is None


def test_frameworks_are_stored_for_traceability(agent):
    rows = db.list_standards("ISTQB-CTFL-4")
    assert len(rows) == 6 and all(r["source_id"] == "SRC-INTL-ISTQB" for r in rows)


# =========================================================================== structured curriculum (FR 20-21)
def test_curriculum_has_every_required_part(health):
    for field in ("objectives", "prerequisites", "learning_outcomes", "modules", "practical_exercises",
                  "assessments"):
        assert getattr(health, field), field
    assert health.target_audience == HEALTH["target_audience"] and len(health.capstone_project) > 40
    assert health.status == ApprovalStatus.draft and health.version == 1
    assert all(len(m.topics) >= 3 and m.exercises and m.hours > 0 for m in health.modules)


def test_hours_and_weights_add_up(health, testing):
    for cur in (health, testing):
        assert sum(m.hours for m in cur.modules) == pytest.approx(cur.training_hours)
        assert sum(a.weight_pct for a in cur.assessments) == pytest.approx(100)
    assert health.training_hours == 40 and health.modules[-1].title == "Capstone project"


def test_requested_outcomes_and_projects_are_kept(health):
    assert HEALTH["expected_outcomes"][0] in health.learning_outcomes
    assert health.capstone_project.startswith("Patient-safety KPI dashboard")
    assert "Root cause analysis report" in health.practical_exercises


def test_modules_come_from_the_retrieved_references(health):
    titles = [m.title for m in health.modules]
    assert "Patient safety" in titles                       # healthcare reference syllabus
    assert any("data" in t.lower() for t in titles)         # data analysis reference: cross-domain title
    assert "references" in health.metadata["evidence"]


def test_metadata_follows_the_brief_table(health):
    m = health.metadata
    for key in ("department", "target_audience", "skill_level", "industry", "duration_hours", "learning_outcomes",
                "prerequisites", "accreditation_framework", "country", "data_sources", "update_date"):
        assert m.get(key) is not None, key
    assert {"SRC-CUR-REFS", "SRC-CUR-INDUSTRY"} <= {s.source_id for s in health.sources}


def test_outcomes_cover_every_learning_domain(health):
    domains = set().union(*(outcome_domain(o) for o in health.learning_outcomes))
    assert {"knowledge", "intellectual", "practical", "transferable"} <= domains
    assert all(is_measurable(o) for o in health.learning_outcomes)


def test_existing_course_redesign_uses_its_modules(agent):
    cur = agent.generate(CurriculumRequest(course_id="CY-101", country="SA"))
    course = agent.kb.courses["CY-101"]
    teaching = [m.title for m in cur.modules[:-1]]            # the capstone module is added last
    assert len(teaching) >= 5 and teaching == course.modules[:len(teaching)]
    assert cur.course_id == "CY-101" and cur.metadata["country"] == "SA"


def test_unknown_topic_is_flagged_not_invented(agent):
    cur = agent.generate(CurriculumRequest(course_title="Underwater Basket Weaving", target_audience="Hobbyists",
                                           skill_level="Beginner", total_hours=12))
    assert cur.metadata["evidence"] == ["none"]
    assert any("No reference syllabus" in n for n in cur.metadata["research"]["notes"])
    assert not any("security" in m.title.lower() for m in cur.modules)


def test_arabic_curriculum(agent):
    cur = agent.generate(CurriculumRequest(course_title="إدارة المشاريع الاحترافية", target_audience="مديرو المشاريع",
                                           skill_level="Intermediate", total_hours=35, language="ar", country="SA"))
    assert cur.language.value == "ar" and cur.modules[-1].title == "المشروع التطبيقي الختامي"
    assert any("المخاطر" in m.title for m in cur.modules)
    pmp = next(r for r in agent.gap_reports(cur.curriculum_id) if r.framework_id == "PMI-PMP-ECO-2021")
    assert pmp.coverage_pct == 100                           # Arabic keywords in frameworks.yaml


# =========================================================================== alignment and gaps (FR 22-23)
def test_chosen_framework_is_fully_covered(agent, testing):
    reports = {r.framework_id: r for r in agent.gap_reports(testing.curriculum_id)}
    assert set(reports) == {"ISTQB-CTFL-4", "EG-NAQAAE-PS", "REX-QUALITY"}
    istqb = reports["ISTQB-CTFL-4"]
    assert istqb.coverage_pct == 100 and not istqb.missing_topics
    assert all(i.evidence for i in istqb.items)


def test_suggested_framework_reports_real_gaps(agent):
    cur = agent.generate(CurriculumRequest(course_id="CY-101"))
    sfia = next(r for r in agent.gap_reports(cur.curriculum_id) if r.framework_id == "SFIA-9")
    assert "Penetration testing" in sfia.missing_topics and sfia.coverage_pct < 100
    assert any("Penetration testing" in rec for rec in sfia.recommendations)


def test_imported_catalogue_course_is_compared_with_istqb(agent):
    cur = agent.import_curriculum(CurriculumImport(course_id="SW-201", framework="ISTQB Foundation"))
    reports = {r.framework_id: r for r in agent.gap_reports(cur.curriculum_id)}
    istqb, baseline = reports["ISTQB-CTFL-4"], reports["REX-QUALITY"]
    assert istqb.missing_topics and istqb.coverage_pct < 100
    failed = {i.competency_id for i in baseline.items if i.status == "missing"}
    assert {"Q-ASSESS", "Q-PRAC"} <= failed                  # catalogue has no assessments or exercises
    assert cur.metadata["origin"] == "catalogue course SW-201"


def test_country_checklist_finds_missing_outcome_domains(agent):
    proposal = ProposedCurriculum(course_title="Excel for Accountants", target_audience="Accountants",
                                  learning_outcomes=["Understand pivot tables", "Know Excel formulas"],
                                  modules=[{"title": "Formulas", "topics": ["SUM", "VLOOKUP"], "hours": 6}])
    cur = agent.import_curriculum(CurriculumImport(proposal=proposal, country="SA"))
    nqf = next(r for r in agent.gap_reports(cur.curriculum_id) if r.framework_id == "SA-NQF-DOMAINS")
    assert "Learning outcomes: knowledge and understanding" in nqf.missing_outcomes
    base = next(r for r in agent.gap_reports(cur.curriculum_id) if r.framework_id == "REX-QUALITY")
    assert "Learning outcomes use measurable action verbs" in [i.competency for i in base.items if i.status == "missing"]


def test_gap_report_on_demand_and_unknown_framework(agent, health):
    rep = agent.gap_report(health.curriculum_id, "SFIA 9")
    assert rep.framework_id == "SFIA-9" and rep.curriculum_id == health.curriculum_id
    assert "not evidence of official accreditation" in rep.disclaimer
    with pytest.raises(NotFound):
        agent.gap_report(health.curriculum_id, "Galactic Standards Board")


def test_removing_a_topic_creates_a_gap(agent, testing):
    rev = agent.revise(testing.curriculum_id, CurriculumRevision(reviewer="Dr. Hany", instructions="remove module Static testing"))
    istqb = next(r for r in agent.gap_reports(testing.curriculum_id) if r.framework_id == "ISTQB-CTFL-4")
    assert rev.version == 2 and istqb.curriculum_version == 2
    assert "Static testing" in istqb.missing_topics
    assert any("Static testing" in r for r in istqb.recommendations)


# =========================================================================== never claim accreditation
@pytest.mark.parametrize("text", [
    "This course is accredited by NAQAAE.", "An officially accredited programme.",
    "The course is certified by ISTQB.", "It has official accreditation from ETEC.", "الدورة معتمدة رسمياً من الهيئة",
])
def test_accreditation_claims_are_rewritten(text):
    out, n = guard_text(text)
    assert n >= 1 and "accredited" not in out.lower() and "certified by" not in out.lower() and "معتمدة" not in out


@pytest.mark.parametrize("text", ["This course is not accredited by any body.", "Alignment, not accreditation.",
                                  "ISTQB Certified Tester Foundation Level topics"])
def test_honest_statements_are_kept(text):
    assert guard_text(text) == (text, 0)


def test_imported_claims_are_removed(agent):
    proposal = ProposedCurriculum(course_title="QA Bootcamp", target_audience="Testers",
                                  objectives=["Deliver a course officially accredited by ISTQB"],
                                  modules=[{"title": "Test design", "topics": ["Boundary value analysis"], "hours": 10}])
    cur = agent.import_curriculum(CurriculumImport(proposal=proposal, framework="ISTQB"))
    assert cur.metadata["claims_removed"] == 1 and "accredited" not in cur.objectives[0]


def _llm_curriculum(**overrides):
    base = {"objectives": ["Prepare learners for an officially accredited ISTQB exam"], "prerequisites": ["Basic coding"],
            "learning_outcomes": ["Design test cases with boundary value analysis"],
            "modules": [{"title": "Test design", "topics": ["Boundary values", "Decision tables", "State transitions"],
                         "hours": 20, "exercises": ["Design tests for a login form"]},
                        {"title": "Capstone project", "topics": ["Plan", "Build", "Present"], "hours": 5,
                         "exercises": ["Test a web shop"]}],
            "practical_exercises": ["Design tests for a login form"],
            "assessments": [{"name": "Project", "type": "project", "weight_pct": 60},
                            {"name": "Quiz", "type": "quiz", "weight_pct": 20}],
            "capstone_project": "Test plan, test cases and defect reports for a web shop, presented to a panel."}
    return {**base, **overrides}


def test_llm_output_is_checked_like_rule_output(agent, monkeypatch):
    from rex_ai.agents.curriculum import llm
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "refine", lambda *a, **k: _llm_curriculum())
    cur = agent.generate(CurriculumRequest(**TESTING))
    assert cur.metadata["generated_by"] == "llm" and cur.metadata["claims_removed"] == 1
    assert "accredited" not in cur.objectives[0]
    assert sum(m.hours for m in cur.modules) == 30 and sum(a.weight_pct for a in cur.assessments) == 100


def test_broken_llm_output_falls_back_to_rules(agent, monkeypatch):
    from rex_ai.agents.curriculum import llm
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "refine", lambda *a, **k: _llm_curriculum(modules=[{"title": "No topics", "hours": "x"}]))
    cur = agent.generate(CurriculumRequest(**TESTING))
    assert cur.metadata["generated_by"] == "rules" and len(cur.modules) >= 4


def test_guard_skips_source_titles():
    data, n = guard_claims({"objectives": ["accredited by X"], "sources": [{"title": "Accredited by X list"}]})
    assert n == 1 and data["sources"][0]["title"] == "Accredited by X list"


# =========================================================================== review, modification, approval (FR 25)
def test_review_workflow_and_handoff(agent):
    cur = agent.generate(CurriculumRequest(**{**HEALTH, "course_title": "Patient Safety Analytics"}))
    cid = cur.curriculum_id
    assert agent.review(cid, ReviewAction(action="pending_approval", reviewer="Coordinator")).status.value == "pending_approval"
    agent.review(cid, ReviewAction(action="changes_requested", reviewer="Dr. Mona", comment="More on dashboards"))
    rev = agent.revise(cid, CurriculumRevision(reviewer="Dr. Mona", instructions=
                       "add topic Power BI dashboards to module 1\nadd outcome: Interpret control charts\nmake it fun"))
    assert rev.version == 2 and rev.status == ApprovalStatus.draft
    assert "Power BI dashboards" in rev.modules[0].topics and "Interpret control charts" in rev.learning_outcomes
    assert rev.metadata["last_revision"]["not_applied"] == ["make it fun"]
    approved = agent.review(cid, ReviewAction(action="approved", reviewer="Dean"))
    assert approved.status == ApprovalStatus.approved and approved.metadata["approved_by"] == "Dean"
    handoffs = db.list_handoffs(cid)
    assert handoffs[-1]["from_agent"] == "curriculum" and handoffs[-1]["to_agent"] == "document"
    with pytest.raises(ValueError):
        agent.review(cid, ReviewAction(action="approved", reviewer="Dean"))
    history = agent.history(cid)
    assert [v["version"] for v in history["versions"]] == [1, 2]
    assert [r["action"] for r in history["reviews"]] == ["changes_requested", "changes_requested", "approved"]


def test_revision_with_edited_content(agent, health):
    new_modules = [m.model_dump() for m in health.modules[:3]]
    rev = agent.revise(health.curriculum_id, CurriculumRevision(reviewer="QA", content={"modules": new_modules}))
    assert len(rev.modules) == 3 and rev.training_hours == sum(m["hours"] for m in new_modules)
    with pytest.raises(ValueError):
        agent.revise(health.curriculum_id, CurriculumRevision(reviewer="QA", content={"status": "approved"}))


def test_revision_that_changes_nothing_keeps_the_version(agent, testing):
    before = agent.get(testing.curriculum_id).version
    rev = agent.revise(testing.curriculum_id, CurriculumRevision(reviewer="QA", instructions="please be nicer"))
    assert rev.version == before and rev.status == ApprovalStatus.changes_requested


def test_instruction_rules():
    cur = {"training_hours": 30.0, "learning_outcomes": ["Explain A"], "modules": [
        {"title": "Intro", "topics": ["A"], "hours": 10, "exercises": []},
        {"title": "SQL basics", "topics": ["Joins"], "hours": 14, "exercises": []},
        {"title": "Capstone project", "topics": [], "hours": 6, "exercises": []}]}
    out, ok, ko = apply_instructions(cur, "add module on Online marketing\nadd topic Window functions to module 2\n"
                                          "rename module 1 to Foundations\nset hours to 48\nremove topic joins\n"
                                          "add topic data cleaning in Excel\nremove outcome 1\nadd topic X to module 9",
                                     capstone_title="Capstone project")
    titles = [m["title"] for m in out["modules"]]
    assert titles == ["Foundations", "SQL basics", "Online marketing", "Capstone project"]
    assert out["training_hours"] == 48 and sum(m["hours"] for m in out["modules"]) == pytest.approx(48)
    assert out["modules"][1]["topics"] == ["Window functions"] and out["learning_outcomes"] == []
    assert any("Data cleaning in Excel" in t for m in out["modules"] for t in m["topics"])
    assert ko == ["add topic X to module 9 (module not found)"] and len(ok) == 7
    assert cur["modules"][0]["title"] == "Intro"            # input not modified


def test_hour_helpers():
    assert sum(split_hours(34, 6)) == 34 and len(split_hours(34, 6)) == 6
    mods = [{"hours": 10}, {"hours": 10}, {"hours": 5}]
    rescale_hours(mods, 50)
    assert [m["hours"] for m in mods] == [20, 20, 10]


# =========================================================================== metadata filtering
def test_filter_curricula_by_metadata(agent, health, testing):
    assert health.curriculum_id in {c.curriculum_id for c in agent.search(department="Healthcare")}
    ids = {c.curriculum_id for c in agent.search(department="Software", skill_level="Intermediate", max_hours=30)}
    assert testing.curriculum_id in ids and health.curriculum_id not in ids
    assert {c.curriculum_id for c in agent.search(framework="ISTQB")} >= {testing.curriculum_id}
    assert all(c.metadata["country"] == "SA" for c in agent.search(country="SA"))


# =========================================================================== PDF (FR 24)
def _pdf_text(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def test_pdf_export(agent, health):
    path = agent.export_pdf(health.curriculum_id)
    assert path.read_bytes()[:4] == b"%PDF" and path.name.endswith("_draft.pdf")
    text = _pdf_text(path)
    for part in ("Course information", "Learning outcomes", "Course structure", "Assessment plan", "Capstone project",
                 "Accreditation alignment report", "not evidence of official accreditation", "Sources and traceability",
                 "DRAFT"):
        assert part in text, part


def test_arabic_pdf(agent):
    cur = agent.generate(CurriculumRequest(course_title="تحليل البيانات", target_audience="المحاسبون",
                                           skill_level="Beginner", total_hours=24, language="ar"))
    path = agent.export_pdf(cur.curriculum_id)
    assert path.stat().st_size > 5000


# =========================================================================== API
@pytest.fixture(scope="module")
def client():
    from rex_ai.api.main import app
    return TestClient(app)


def test_api_curriculum_flow(client):
    r = client.post("/curriculum", json=TESTING)
    assert r.status_code == 201
    cid = r.json()["curriculum_id"]
    assert client.get(f"/curriculum/{cid}").json()["course_title"] == TESTING["course_title"]
    assert any(c["curriculum_id"] == cid for c in client.get("/curriculum", params={"department": "Software"}).json())
    rep = client.post(f"/curriculum/{cid}/gap-report", json={"framework": "SFIA 9"}).json()
    assert rep["framework_id"] == "SFIA-9"
    assert client.post(f"/curriculum/{cid}/gap-report", json={"framework": "Nope Board"}).status_code == 404
    assert len(client.get(f"/curriculum/{cid}/gap-reports").json()) >= 3
    rev = client.post(f"/curriculum/{cid}/revise", json={"reviewer": "QA", "instructions": "add outcome: Report defects"})
    assert rev.json()["version"] == 2
    assert client.post(f"/curriculum/{cid}/review", json={"action": "approved", "reviewer": "Dean"}).status_code == 200
    assert client.post(f"/curriculum/{cid}/review", json={"action": "approved", "reviewer": "Dean"}).status_code == 409
    pdf = client.get(f"/curriculum/{cid}/pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert client.get(f"/curriculum/{cid}/history").json()["handoffs"][0]["to_agent"] == "document"


def test_api_validation_and_lookup(client):
    assert client.post("/curriculum", json={"course_title": "Power BI"}).status_code == 422
    assert client.get("/curriculum/CUR-NOPE").status_code == 404
    assert client.post("/curriculum", json={"course_id": "XX-1"}).status_code == 404
    names = {f["framework_id"] for f in client.get("/curriculum/frameworks").json()}
    assert {"ISTQB-CTFL-4", "SFIA-9", "EG-NAQAAE-PS", "SA-NQF-DOMAINS", "REX-QUALITY"} <= names
    research = client.post("/curriculum/research", json=HEALTH).json()
    assert research["department"] == "Healthcare" and research["reference_modules"]


def test_api_import(client):
    r = client.post("/curriculum/import", json={"course_id": "SW-201", "framework": "ISTQB"})
    assert r.status_code == 201 and r.json()["metadata"]["framework_id"] == "ISTQB-CTFL-4"


def test_orchestrator_routes_curriculum_requests(client):
    ask = client.post("/route", json={"session_id": "c1", "message": "Design a curriculum for Power BI"}).json()
    assert ask["agent"] == "curriculum" and ask["result"]["status"] == "need_input"
    assert ask["result"]["understood"]["course_title"] == "Power BI" and len(ask["result"]["questions"]) == 3
    done = client.post("/route", json={"session_id": "c1", "message":
                       "Design a curriculum for Software Testing for junior developers, intermediate, 30 hours, ISTQB"}).json()
    assert done["result"]["status"] == "draft_created"
    assert done["result"]["alignment"][0]["framework_id"] == "ISTQB-CTFL-4"
    ar = client.post("/route", json={"session_id": "c2", "message": "أريد منهج عن تحليل البيانات للمحاسبين، مبتدئ، 24 ساعة"})
    result = ar.json()["result"]
    assert result["status"] == "draft_created" and result["course_title"] == "تحليل البيانات"
