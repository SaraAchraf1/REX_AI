"""Change requests are applied by Gemini (Curriculum, Document and Marketing agents). Run: pytest -q tests/test_llm_changes.py

With Gemini configured, a plain-language change request goes only to Gemini. If it cannot do it (no credits, bad key,
network, unusable answer) nothing is changed and the reviewer gets the reason. No real call is made: the OpenAI
client (also used for Gemini's OpenAI-compatible endpoint) is replaced by a fake.
"""
import json

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent
from rex_ai.agents.documents.agent import DocumentAgent
from rex_ai.agents.marketing.agent import MarketingAgent
from rex_ai.core import llm_status
from rex_ai.core.llm_status import LLMUnavailable
from rex_ai.core.schemas import (CampaignEdit, CampaignRequest, CurriculumRequest, CurriculumRevision, DocumentEdit,
                                 DocumentRequest)

QUOTA = ("Error code: 429 - {'error': {'message': 'You have no credits remaining.', 'type': 'insufficient_quota', "
         "'code': 'credit_balance_exhausted'}}")
COURSE = dict(course_title="Data Analysis for Retail", target_audience="Store managers", skill_level="Beginner",
              total_hours=30)


class FakeOpenAI:
    """Stands in for openai.OpenAI: answers with `reply` (a dict or a function of the prompt), or raises `error`."""
    reply = None
    error = None
    prompts: list = []
    models: list = []

    def __init__(self, api_key=None, base_url=None):
        FakeOpenAI.models.append(base_url)
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, response_format):
        FakeOpenAI.prompts.append(messages[-1]["content"])
        if FakeOpenAI.error:
            raise Exception(FakeOpenAI.error)
        reply = FakeOpenAI.reply(messages[-1]["content"]) if callable(FakeOpenAI.reply) else FakeOpenAI.reply
        msg = type("M", (), {"content": json.dumps(reply)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


@pytest.fixture(scope="module")
def drafts():
    """Drafts made without the LLM (as in every other test)."""
    cur = CurriculumAgent().generate(CurriculumRequest(**COURSE))
    doc = DocumentAgent().generate(DocumentRequest(doc_type="capstone_brief", curriculum_id=cur.curriculum_id))
    camp = MarketingAgent().generate(CampaignRequest(course_id="DS-101", channels=["linkedin_post"],
                                                     variations_per_channel=1))
    return cur, doc, camp


@pytest.fixture
def llm_on(monkeypatch):
    import openai
    monkeypatch.setattr(config, "USE_LLM", True)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "USE_GEMINI", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gm-test")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.reply, FakeOpenAI.error, FakeOpenAI.prompts, FakeOpenAI.models = None, None, [], []
    return FakeOpenAI


# =========================================================================== the reason, in plain words
def test_reasons_are_explained():
    assert "no credits left" in llm_status.explain(Exception(QUOTA))
    assert "not valid" in llm_status.explain(Exception("Error code: 401 - invalid_api_key"))
    assert "rate-limiting" in llm_status.explain(Exception("Error code: 429 - Rate limit reached"))
    assert "could not be reached" in llm_status.explain(Exception("Connection error."))
    assert "GEMINI_API_KEY" in llm_status.explain(Exception("400 API key not valid"), provider="Gemini")


# =========================================================================== LLM fails -> nothing changes, reason shown
def test_curriculum_change_fails_clearly(drafts, llm_on):
    cur, _, _ = drafts
    llm_on.error = QUOTA
    with pytest.raises(LLMUnavailable, match="no credits left") as e:
        CurriculumAgent().revise(cur.curriculum_id, CurriculumRevision(
            reviewer="QA", instructions="Add a module on Power BI dashboards before the capstone"))
    assert "Nothing was changed" in str(e.value)
    now = CurriculumAgent().get(cur.curriculum_id)
    assert now.version == cur.version and [m.title for m in now.modules] == [m.title for m in cur.modules]
    assert "Power BI dashboards before the capstone" in llm_on.prompts[-1]          # the request went to the LLM


def test_document_change_fails_clearly(drafts, llm_on):
    _, doc, _ = drafts
    llm_on.error = QUOTA
    with pytest.raises(LLMUnavailable, match="no credits left"):
        DocumentAgent().edit(doc.document_id, DocumentEdit(reviewer="QA", instructions="Set the scenario in Cairo"))
    assert DocumentAgent().get(doc.document_id).version == doc.version


def test_campaign_change_fails_clearly(drafts, llm_on):
    _, _, camp = drafts
    llm_on.error = QUOTA
    with pytest.raises(LLMUnavailable, match="no credits left"):
        MarketingAgent().edit(camp.campaign_id, CampaignEdit(reviewer="QA", instructions="Make it shorter"))
    assert MarketingAgent().get(camp.campaign_id).version == camp.version


def test_unusable_answer_changes_nothing(drafts, llm_on):
    cur, _, _ = drafts
    llm_on.reply = {"objectives": ["only one key"]}                                  # not a curriculum
    with pytest.raises(LLMUnavailable, match="required format"):
        CurriculumAgent().revise(cur.curriculum_id, CurriculumRevision(reviewer="QA", instructions="Shorten it"))


# =========================================================================== LLM works -> its change is applied
def test_curriculum_change_applied_by_llm(drafts, llm_on):
    cur, _, _ = drafts
    base = cur.model_dump(mode="json")

    def answer(prompt):
        modules = base["modules"][:-1] + [
            {"title": "Power BI dashboards", "topics": ["Data model", "DAX measures", "Report design"], "hours": 6,
             "exercises": ["Build a weekly sales dashboard"]}, base["modules"][-1]]
        return {**{k: base[k] for k in ("objectives", "prerequisites", "practical_exercises")},
                "learning_outcomes": base["learning_outcomes"] + ["Present sales insights to store managers"],
                "modules": modules, "assessments": [{"name": "Labs", "type": "assignment", "weight_pct": 70},
                                                    {"name": "Capstone", "type": "project", "weight_pct": 50}],
                "capstone_project": "Weekly sales dashboard for a supermarket chain", "training_hours": 36}

    llm_on.reply = answer
    new = CurriculumAgent().revise(cur.curriculum_id, CurriculumRevision(
        reviewer="QA", instructions="Add a Power BI module, make the capstone a supermarket sales dashboard, "
                                    "add an outcome about presenting to store managers, set hours to 36"))
    assert new.version == cur.version + 1 and new.metadata["last_revision"]["applied"][0].startswith("LLM applied")
    assert "Power BI dashboards" in [m.title for m in new.modules] and new.modules[-1].title == cur.modules[-1].title
    assert new.capstone_project.startswith("Weekly sales dashboard")
    assert new.training_hours == 36 and abs(sum(m.hours for m in new.modules) - 36) < 0.01   # hours add up
    assert abs(sum(a.weight_pct for a in new.assessments) - 100) < 0.01                       # weights add up


def test_document_change_applied_by_llm(drafts, llm_on):
    _, doc, _ = drafts
    llm_on.reply = {"sections": [{"key": "scenario", "paragraphs": ["You work for a supermarket chain in Cairo."]}]}
    new = DocumentAgent().edit(doc.document_id, DocumentEdit(reviewer="QA", instructions="Set the scenario in Cairo"))
    assert new.version == doc.version + 1
    assert next(s for s in new.sections if s.key == "scenario").paragraphs == ["You work for a supermarket chain in Cairo."]
    assert llm_on.models == [config.GEMINI_BASE_URL]                                # the change went to Gemini


def test_document_change_can_touch_anything(drafts, llm_on):
    _, doc, _ = drafts
    locked = next(s for s in doc.sections if s.locked)
    gone = next(s for s in doc.sections if not s.locked and s.key != "scenario")
    llm_on.reply = {"title": "Cairo Retail Capstone",
                    "sections": [{"key": locked.key, "heading": locked.heading, "paragraphs": ["Updated by request."]},
                                 {"key": "faq", "heading": "FAQ", "bullets": ["Who marks it? The instructor."],
                                  "after": locked.key}],
                    "remove": [gone.key]}
    new = DocumentAgent().edit(doc.document_id, DocumentEdit(reviewer="QA", instructions="Rename, add an FAQ, ..."))
    keys = [s.key for s in new.sections]
    assert new.title == "Cairo Retail Capstone" and gone.key not in keys
    assert keys[keys.index(locked.key) + 1] == "faq"
    edited = next(s for s in new.sections if s.key == locked.key)
    assert edited.paragraphs == ["Updated by request."] and edited.locked           # locked flag kept


def test_campaign_change_applied_by_llm(drafts, llm_on):
    _, _, camp = drafts
    v = camp.variants[0]
    body = "Ready to turn data into decisions? Ideal for fresh graduates. " + \
        "\n".join(line for line in v.body.splitlines() if line.startswith("#"))
    llm_on.reply = {"variants": [{"channel": v.channel.value, "variant_no": v.variant_no, "headline": v.headline,
                                  "body": body, "cta": v.cta}]}
    new = MarketingAgent().edit(camp.campaign_id, CampaignEdit(reviewer="QA",
                                                               instructions="Shorter, mention fresh graduates"))
    assert new.version == camp.version + 1 and "fresh graduates" in new.variants[0].body


def test_campaign_change_may_use_reviewer_numbers_and_strategy(drafts, llm_on):
    _, _, camp = drafts
    v = camp.variants[0]
    body = "Early-bird: 15% off until 30 November. " + \
        "\n".join(line for line in v.body.splitlines() if line.startswith("#"))
    llm_on.reply = {"variants": [{"channel": v.channel.value, "variant_no": v.variant_no, "headline": v.headline,
                                  "body": body, "cta": v.cta}], "strategy": "Push the early-bird offer."}
    new = MarketingAgent().edit(camp.campaign_id, CampaignEdit(
        reviewer="QA", instructions="Add an early-bird 15% discount until 30 November and focus the strategy on it"))
    assert "15% off" in new.variants[0].body and new.strategy == "Push the early-bird offer."


# =========================================================================== API: 503 with the reason
def test_api_reports_llm_unavailable(drafts, llm_on):
    from rex_ai.api.main import app
    cur, doc, camp = drafts
    llm_on.error = QUOTA
    c = TestClient(app)
    for path, method, body in ((f"/curriculum/{cur.curriculum_id}/revise", "POST",
                                {"reviewer": "QA", "instructions": "add a module on Excel"}),
                               (f"/documents/{doc.document_id}", "PATCH", {"reviewer": "QA", "instructions": "shorter"}),
                               (f"/campaigns/{camp.campaign_id}", "PATCH", {"reviewer": "QA", "instructions": "shorter"})):
        r = c.request(method, path, json=body)
        assert r.status_code == 503 and r.json()["code"] == "llm_unavailable", path
        assert "no credits left" in r.json()["message"]


def test_without_llm_curriculum_keeps_simple_rules(drafts):
    cur, _, _ = drafts                                                               # REX_USE_LLM=0 in tests
    new = CurriculumAgent().revise(cur.curriculum_id, CurriculumRevision(reviewer="QA", instructions="set hours to 40"))
    assert new.training_hours == 40
