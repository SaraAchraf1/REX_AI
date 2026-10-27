"""Agent 4 - Educational Content & Document Generation Agent (owner: Member 3).
Run: pytest -q tests/test_document_agent.py

Covers the brief's Agent 4 requirements: select a document type (all 10), collect course information (curriculum,
catalogue, RAG reference syllabi), template-based content, PDF export, Arabic and English, review and edit before
approval, downloads and e-mail only after approval - plus the Curriculum -> Document -> Marketing handoffs and the
rule that no document claims accreditation. Runs offline (hash embedder, no LLM) - see conftest.py.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent
from rex_ai.agents.documents import llm
from rex_ai.agents.documents import templates as tpl
from rex_ai.agents.documents.agent import DocumentAgent, NotFound
from rex_ai.agents.documents.builders import bloom_level
from rex_ai.core import db
from rex_ai.core.schemas import (ApprovalStatus, CurriculumRequest, DocSection, DocTable, DocumentEdit,
                                 DocumentEmail, DocumentPackRequest, DocumentRequest, DocumentType, ReviewAction)

TESTING = dict(course_title="Software Testing Fundamentals", target_audience="Junior developers",
               skill_level="Intermediate", total_hours=30, accreditation_framework="ISTQB Foundation")
CLAIM = re.compile(r"\b(?:accredited|certified|endorsed)\s+by\b|officially accredited", re.I)


def _text(doc) -> str:
    parts = [doc.title]
    for s in doc.sections:
        parts += [s.heading, *s.paragraphs, *s.bullets, *s.notes]
        if s.table:
            parts += s.table.columns + [c for r in s.table.rows for c in r]
    return "\n".join(parts)


def _approve(cid: str) -> None:
    CurriculumAgent().review(cid, ReviewAction(action="approved", reviewer="Dean"))


@pytest.fixture(scope="module")
def agent():
    return DocumentAgent()


@pytest.fixture(scope="module")
def curriculum():
    return CurriculumAgent().generate(CurriculumRequest(**TESTING))


@pytest.fixture(scope="module")
def all_docs(agent, curriculum):
    return {t: agent.generate(DocumentRequest(doc_type=t, curriculum_id=curriculum.curriculum_id,
                                              audience="Ministry QA team")) for t in DocumentType}


# =========================================================================== templates (brief: "based on templates")
def test_every_document_type_has_a_template():
    types = {t.doc_type for t in DocumentAgent.types()}
    assert types == set(DocumentType)                                  # brief items 26-35
    for info in DocumentAgent.types():
        assert info.sections and info.title_ar and info.template_id in info.templates


def test_unknown_template_is_rejected(agent, curriculum):
    with pytest.raises(NotFound):
        agent.generate(DocumentRequest(doc_type="question_bank", curriculum_id=curriculum.curriculum_id,
                                       template_id="TPL-NOPE"))


def test_template_order_is_followed(all_docs):
    for t, doc in all_docs.items():
        order = [s.key for s in tpl.get_template(t.value).sections]
        keys = [s.key.split(":")[0] for s in doc.sections if s.level == 1 or ":" not in s.key]
        seen = list(dict.fromkeys(k for k in keys if k in order))
        assert seen == [k for k in order if k in seen], t


# =========================================================================== generation (all 10 types)
def test_all_ten_types_generate_a_pdf(all_docs):
    for t, doc in all_docs.items():
        path = Path(doc.file_path)
        assert path.exists() and path.read_bytes()[:4] == b"%PDF", t
        assert doc.status == ApprovalStatus.draft and doc.version == 1 and "draft" in path.name
        assert doc.sections and doc.download_url == f"/documents/{doc.document_id}/download"
        assert doc.metadata["generated_by"] == "rules" and doc.metadata["type_title"]


def test_documents_are_traceable(all_docs, curriculum):
    for t, doc in all_docs.items():
        ids = {s.source_id for s in doc.sources}
        assert "SRC-DOC-TEMPLATES" in ids and "REX-CURRICULUM" in ids, t
        assert any(s.record_id == curriculum.curriculum_id for s in doc.sources)
        assert doc.metadata["facts"]["curriculum_version"] == curriculum.version


def test_no_document_claims_accreditation(all_docs):
    for t, doc in all_docs.items():
        assert not CLAIM.search(_text(doc)), t


def test_facts_are_copied_not_invented(all_docs, curriculum):
    desc = all_docs[DocumentType.curriculum_pdf]
    structure = next(s for s in desc.sections if s.key == "structure_table")
    assert structure.locked and structure.table.rows[-1][-1] == f"{curriculum.training_hours:g}"
    assert [r[1] for r in structure.table.rows[:-1]] == [m.title for m in curriculum.modules]
    plan = next(s for s in all_docs[DocumentType.outcomes_assessment_plan].sections if s.key == "assessment_table")
    assert plan.table.rows[-1][2] == "100%"
    assert len(next(s for s in all_docs[DocumentType.outcomes_assessment_plan].sections
                    if s.key == "outcome_map").table.rows) == len(curriculum.learning_outcomes)


def test_proposal_is_prepared_for_the_client(all_docs):
    doc = all_docs[DocumentType.course_proposal]
    assert doc.metadata["prepared_for"] == "Ministry QA team"
    assert "Ministry QA team" in _text(doc)
    invest = next(s for s in doc.sections if s.key == "investment")
    assert invest.locked and any("confirmed" in b for b in invest.bullets)


def test_question_bank_answers_are_consistent(all_docs):
    doc = all_docs[DocumentType.question_bank]
    rows = [r for s in doc.sections if s.key.startswith("module_questions") and s.table for r in s.table.rows]
    mcq = [r for r in rows if r[1] == "Multiple choice"]
    assert mcq and len(rows) >= 10
    for r in mcq:
        options = dict(o.split(". ", 1) for o in r[5].split("\n"))
        assert set(options) == set("ABCD") and r[6] in options       # exactly one keyed answer among 4 options
    blueprint = next(s for s in doc.sections if s.key == "blueprint")
    assert blueprint.locked and int(blueprint.table.rows[-1][-1]) == len(rows)
    assert "CONFIDENTIAL" not in doc.title                             # shown in the PDF header instead


def test_rubric_points_match_the_assessment_weights(all_docs, curriculum):
    doc = all_docs[DocumentType.assessment_rubric]
    for a in curriculum.assessments:
        sec = next(s for s in doc.sections if s.heading.endswith(a.name))
        pts = sum(float(re.search(r"\(([\d.]+)\)$", r[0]).group(1)) for r in sec.table.rows)
        assert abs(pts - a.weight_pct) < 0.01, a.name
        assert len(sec.table.columns) == 5                             # criterion + 4 performance levels


def test_guides_and_slides_use_reference_syllabi_from_rag(all_docs):
    guide = all_docs[DocumentType.participant_guide]
    assert guide.metadata["reference_modules"]                         # RAG found passages for the modules
    assert any(s.key.endswith(":reading") for s in guide.sections)
    assert any(s.source_id == "SRC-CUR-REFS" for s in guide.sources)
    slides = [s for s in all_docs[DocumentType.presentation_outline].sections if s.key.startswith("slide:")]
    assert len(slides) > 10 and all(s.notes for s in slides)          # every slide has speaker notes
    assert slides[0].heading.startswith("Slide 1:")


def test_catalogue_course_without_curriculum(agent):
    doc = agent.generate(DocumentRequest(doc_type="course_description", course_id="ds-101"))
    assert doc.metadata["origin"] == "catalogue" and doc.course_id == "DS-101" and doc.curriculum_id is None
    fees = next(s for s in doc.sections if s.key == "fees")
    course = agent.kb.courses["DS-101"]
    assert any(f"{course.fee:,.0f}" in b for b in fees.bullets)       # the catalogue fee, not an estimate
    assert {"SRC-ACAD-COURSES", "SRC-ACAD-PRICING"} <= {s.source_id for s in doc.sources}
    qb = agent.generate(DocumentRequest(doc_type="question_bank", course_id="DS-101"))
    assert qb.metadata["reference_modules"], "catalogue modules get their topics from reference syllabi"


def test_course_id_uses_its_approved_curriculum(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_id="SW-201"))
    _approve(cur.curriculum_id)
    doc = agent.generate(DocumentRequest(doc_type="participant_guide", course_id="SW-201"))
    assert doc.curriculum_id == cur.curriculum_id and doc.metadata["origin"] == "curriculum"


def test_arabic_document(agent):
    doc = agent.generate(DocumentRequest(doc_type="participant_guide", course_id="DS-101", language="ar"))
    assert doc.language.value == "ar" and doc.title.startswith("دليل المتدرب")
    assert next(s for s in doc.sections if s.key == "welcome").heading == "مرحباً بك"
    assert Path(doc.file_path).read_bytes()[:4] == b"%PDF"


def test_language_follows_the_curriculum_unless_asked(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_title="إدارة المشاريع الاحترافية",
                                                       target_audience="مديرو المشاريع", skill_level="Intermediate",
                                                       total_hours=35, language="ar", country="SA"))
    doc = agent.generate(DocumentRequest(doc_type="capstone_brief", curriculum_id=cur.curriculum_id))
    assert doc.language.value == "ar"
    en = agent.generate(DocumentRequest(doc_type="capstone_brief", curriculum_id=cur.curriculum_id, language="en"))
    assert en.language.value == "en" and any("written in ar" in n for n in en.metadata["notes"])


def test_request_validation(agent):
    with pytest.raises(ValidationError):
        DocumentRequest(doc_type="question_bank")                      # needs a course or a curriculum
    with pytest.raises(ValidationError):
        DocumentRequest(doc_type="flyer", course_id="DS-101")
    with pytest.raises(NotFound):
        agent.generate(DocumentRequest(doc_type="question_bank", course_id="XX-999"))
    with pytest.raises(NotFound):
        agent.generate(DocumentRequest(doc_type="question_bank", curriculum_id="CUR-NOPE"))
    with pytest.raises(ValidationError):
        DocTable(columns=["a", "b"], rows=[["only one"]])


# =========================================================================== review, edit, approval
def test_edit_creates_a_new_version(agent, curriculum):
    doc = agent.generate(DocumentRequest(doc_type="course_description", curriculum_id=curriculum.curriculum_id))
    overview = next(s for s in doc.sections if s.key == "overview")
    edited = overview.model_copy(update={"paragraphs": ["A shorter overview written by the reviewer."]})
    new = agent.edit(doc.document_id, DocumentEdit(reviewer="Ms Salma", sections=[edited],
                                                   remove_sections=["careers"], title="QA Course Description"))
    assert new.version == 2 and new.status == ApprovalStatus.draft and new.title == "QA Course Description"
    assert next(s for s in new.sections if s.key == "overview").paragraphs == [edited.paragraphs[0]]
    assert not any(s.key == "careers" for s in new.sections)
    assert "v2" in Path(new.file_path).name
    h = agent.history(doc.document_id)
    assert [v["version"] for v in h["versions"]] == [1, 2] and h["reviews"][0]["action"] == "changes_requested"


def test_edit_keeps_locked_flag_and_guards_claims(agent, curriculum):
    doc = agent.generate(DocumentRequest(doc_type="course_description", curriculum_id=curriculum.curriculum_id))
    fees = next(s for s in doc.sections if s.key == "fees").model_copy(update={"locked": False,
                                                                                "bullets": ["Fee: 9,000 EGP"]})
    claim = DocSection(key="extra", heading="Recognition", paragraphs=["This course is accredited by ISTQB."])
    new = agent.edit(doc.document_id, DocumentEdit(reviewer="QA", sections=[fees, claim]))
    assert next(s for s in new.sections if s.key == "fees").locked      # a reviewer can edit facts, the flag stays
    assert "aligned with" in next(s for s in new.sections if s.key == "extra").paragraphs[0]
    assert new.metadata["last_edit"]["claims_removed"] == 1


def test_instructions_without_llm_are_reported(agent, curriculum):
    doc = agent.generate(DocumentRequest(doc_type="capstone_brief", curriculum_id=curriculum.curriculum_id))
    new = agent.edit(doc.document_id, DocumentEdit(reviewer="QA", instructions="make the scenario about banking"))
    assert new.version == 1 and new.status == ApprovalStatus.changes_requested
    assert "needs the LLM" in new.metadata["last_edit"]["not_applied"][0]


def test_approval_needs_an_approved_curriculum_then_hands_off_to_marketing(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(**TESTING))
    doc = agent.generate(DocumentRequest(doc_type="participant_guide", curriculum_id=cur.curriculum_id))
    assert any("not approved" in n for n in doc.metadata["notes"])
    agent.review(doc.document_id, ReviewAction(action="pending_approval", reviewer="Author"))
    with pytest.raises(ValueError, match="approve the curriculum first"):
        agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))
    _approve(cur.curriculum_id)
    done = agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean", comment="Ready"))
    assert done.status == ApprovalStatus.approved and done.metadata["approved_by"] == "Dean"
    assert "draft" not in Path(done.file_path).name                    # no watermark file suffix once approved
    handoff = agent.history(doc.document_id)["handoffs"][0]
    assert handoff["from_agent"] == "document" and handoff["to_agent"] == "marketing"
    with pytest.raises(ValueError, match="already approved"):
        agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))
    edited = agent.edit(doc.document_id, DocumentEdit(reviewer="Dean", title="Participant Guide v2"))
    assert edited.status == ApprovalStatus.draft and "approved_by" not in edited.metadata


def test_changed_curriculum_blocks_approval(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(**TESTING))
    doc = agent.generate(DocumentRequest(doc_type="question_bank", curriculum_id=cur.curriculum_id))
    from rex_ai.core.schemas import CurriculumRevision
    CurriculumAgent().revise(cur.curriculum_id, CurriculumRevision(reviewer="QA", instructions="set hours to 40"))
    _approve(cur.curriculum_id)
    with pytest.raises(ValueError, match="regenerate"):
        agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))


def test_pending_handoffs_and_pack(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(**TESTING))
    _approve(cur.curriculum_id)
    waiting = {h["curriculum_id"]: h for h in agent.pending_handoffs()}
    assert waiting[cur.curriculum_id]["documents"] == 0
    docs = agent.generate_pack(DocumentPackRequest(curriculum_id=cur.curriculum_id))
    assert [d.doc_type.value for d in docs] == ["course_description", "participant_guide", "question_bank"]
    after = {h["curriculum_id"]: h for h in agent.pending_handoffs()}[cur.curriculum_id]
    assert after["documents"] == 3 and "question_bank" in after["doc_types"]


# =========================================================================== distribution
def test_email_only_after_approval(agent, monkeypatch):
    monkeypatch.setattr(config, "SMTP_HOST", "")
    cur = CurriculumAgent().generate(CurriculumRequest(**TESTING))
    _approve(cur.curriculum_id)
    doc = agent.generate(DocumentRequest(doc_type="course_description", curriculum_id=cur.curriculum_id,
                                         email_to=["coordinator@academy.example"]))
    with pytest.raises(ValueError, match="Only approved"):
        agent.email(doc.document_id, DocumentEmail(to=["a@b.example"], sender="Author"))
    done = agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))
    sent = done.metadata["emails"][0]                                  # pending e-mail sent on approval
    assert sent["channel"] == "outbox" and sent["to"] == ["coordinator@academy.example"]
    eml = Path(sent["file"]).read_bytes()
    assert b"coordinator@academy.example" in eml and b"application/pdf" in eml
    assert done.emailed_to == [] and done.metadata["pending_email"] == []   # outbox is not "sent"


def test_email_via_smtp(agent, monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append(host)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            sent.append(msg["To"])

    monkeypatch.setattr(config, "SMTP_HOST", "smtp.example")
    monkeypatch.setattr("rex_ai.agents.documents.mailer.smtplib.SMTP", FakeSMTP)
    doc = agent.generate(DocumentRequest(doc_type="course_description", course_id="DS-101"))
    agent.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))   # catalogue: no curriculum
    done = agent.email(doc.document_id, DocumentEmail(to=["team@academy.example"], sender="Dean"))
    assert sent == ["smtp.example", "team@academy.example"] and done.emailed_to == ["team@academy.example"]


# =========================================================================== LLM path (stubbed - no key needed)
def test_llm_refinement_cannot_touch_locked_facts(agent, curriculum, monkeypatch):
    def fake_refine(doc_type, title, sections, facts, context, language, wishes=None):
        assert "[R" in context                                          # reference syllabi passed as context
        out = []
        for s in sections:
            if s.locked:
                continue
            d = s.model_dump()
            if d.get("table"):
                d["table"] = {"columns": ["only"], "rows": [["broken shape"]]}   # must be rejected
            else:
                d["paragraphs"] = ["Polished by the LLM. Officially accredited by ISTQB."]
            out.append(d)
        out.append({"key": "fees", "bullets": ["Fee: 1 EGP"]})          # locked/unknown here - ignored
        return out

    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "refine", fake_refine)
    doc = agent.generate(DocumentRequest(doc_type="participant_guide", curriculum_id=curriculum.curriculum_id))
    assert doc.metadata["generated_by"] == "llm"
    welcome = next(s for s in doc.sections if s.key == "welcome")
    assert welcome.paragraphs[0].startswith("Polished by the LLM") and not CLAIM.search(welcome.paragraphs[0])
    glance = next(s for s in doc.sections if s.key == "glance_table")
    assert glance.locked and "Polished" not in str(glance.table.rows)


def test_llm_question_bank_usage_note(agent, curriculum, monkeypatch):
    def fake_refine(doc_type, title, sections, facts, context, language, wishes=None):
        return [{"key": s.key, "table": {"columns": s.table.columns, "rows": [
            ["1", "Multiple choice", "Understand", "LO1", "Which technique finds boundary values?",
             "A. BVA\nB. Code review\nC. Load test\nD. Linting", "A"]]}}
            for s in sections if s.key.startswith("module_questions") and s.table][:1]

    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "refine", fake_refine)
    doc = agent.generate(DocumentRequest(doc_type="question_bank", curriculum_id=curriculum.curriculum_id))
    usage = next(s for s in doc.sections if s.key == "usage")
    assert any("written by the LLM" in p for p in usage.paragraphs)


def test_llm_revision_applies_instructions(agent, curriculum, monkeypatch):
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "revise", lambda doc_type, sections, instructions, language: [
        {"key": "scenario", "paragraphs": ["You work for a bank in Cairo."]}])
    doc = agent.generate(DocumentRequest(doc_type="capstone_brief", curriculum_id=curriculum.curriculum_id))
    new = agent.edit(doc.document_id, DocumentEdit(reviewer="QA", instructions="make the scenario about banking"))
    assert new.version == 2 and next(s for s in new.sections if s.key == "scenario").paragraphs == \
        ["You work for a bank in Cairo."]


def test_bloom_levels():
    assert bloom_level("Explain the testing principles") == "understand"
    assert bloom_level("Design test cases with black-box techniques") == "create"
    assert bloom_level("Learners will be able to compare test levels") == "analyse"
    assert bloom_level("يطبق أساليب التحليل") == "apply"


def test_reference_examples_are_indexed_in_rag():
    report = tpl.ensure_templates_indexed(force=True)
    assert report["files"] + report["skipped"] >= 10 and not report["errors"]
    context, refs = tpl.reference_examples("question_bank", "question bank style")
    assert "Question Bank" in context and refs[0].source_id == "SRC-DOC-TEMPLATES"


# =========================================================================== API + orchestrator
@pytest.fixture(scope="module")
def client():
    from rex_ai.api.main import app
    return TestClient(app)


def test_api_document_flow(client):
    assert len(client.get("/documents/types").json()) == 10
    cur = client.post("/curriculum", json=TESTING).json()
    r = client.post("/documents", json={"doc_type": "facilitator_guide", "curriculum_id": cur["curriculum_id"]})
    assert r.status_code == 201
    did = r.json()["document_id"]
    assert client.get(f"/documents/{did}").json()["doc_type"] == "facilitator_guide"
    assert any(d["document_id"] == did for d in client.get("/documents", params={"doc_type": "facilitator_guide"}).json())
    pdf = client.get(f"/documents/{did}/download")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.post(f"/documents/{did}/review", json={"action": "approved", "reviewer": "D"}).status_code == 409
    assert client.post(f"/documents/{did}/email", json={"to": ["a@b.example"], "sender": "D"}).status_code == 409
    e = client.patch(f"/documents/{did}", json={"reviewer": "QA", "title": "Instructor guide"})
    assert e.status_code == 200 and e.json()["version"] == 2
    assert client.post(f"/curriculum/{cur['curriculum_id']}/review",
                       json={"action": "approved", "reviewer": "Dean"}).status_code == 200
    assert client.post(f"/documents/{did}/review", json={"action": "approved", "reviewer": "D"}).status_code == 200
    assert client.get(f"/documents/{did}/history").json()["handoffs"][0]["to_agent"] == "marketing"
    assert any(h["curriculum_id"] == cur["curriculum_id"] for h in client.get("/documents/handoffs").json())
    pack = client.post("/documents/pack", json={"course_id": "DS-101", "doc_types": ["course_proposal", "assessment_rubric"]})
    assert pack.status_code == 201 and len(pack.json()) == 2


def test_api_errors(client):
    assert client.post("/documents", json={"doc_type": "question_bank"}).status_code == 422
    assert client.post("/documents", json={"doc_type": "question_bank", "course_id": "XX-1"}).status_code == 404
    assert client.get("/documents/DOC-NOPE").status_code == 404
    assert client.get("/documents/DOC-NOPE/download").status_code == 404
    assert client.patch("/documents/DOC-NOPE", json={"reviewer": "x"}).status_code == 422   # nothing to change


def test_orchestrator_routes_document_requests(client):
    ask = client.post("/route", json={"session_id": "d1", "message": "I need a question bank"}).json()
    assert ask["agent"] == "document" and ask["result"]["status"] == "need_input"
    assert ask["result"]["understood"]["doc_type"] == "question_bank" and len(ask["result"]["questions"]) == 1
    done = client.post("/route", json={"session_id": "d1", "message": "Create a participant guide for DS-101"}).json()
    assert done["result"]["status"] == "draft_created" and done["result"]["doc_type"] == "participant_guide"
    ar = client.post("/route", json={"session_id": "d2", "message": "أريد دليل المدرب لدورة DS-101"}).json()
    assert ar["result"]["status"] == "draft_created" and ar["result"]["title"].startswith("دليل المدرب")
    assert db.get_document(ar["result"]["document_id"])["language"] == "ar"
