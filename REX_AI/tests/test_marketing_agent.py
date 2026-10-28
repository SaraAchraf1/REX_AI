"""Agent 5 - AI Marketing Agent (owner: Member 1).
Run: pytest -q tests/test_marketing_agent.py

Covers the brief's Agent 5 requirements: read approved course details (36), understand the target audience (37),
generate captions, ads, LinkedIn posts, e-mails, WhatsApp messages, headlines and CTAs (38), a simple campaign strategy
(39), several variations (40), the connection with the Curriculum Agent (41) and manual approval before publishing (42)
- plus the Document -> Marketing handoff, the rule that every number comes from approved data, and no accreditation or
job-guarantee claims. Runs offline (hash embedder, no LLM) - see conftest.py.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.curriculum.agent import CurriculumAgent
from rex_ai.agents.documents.agent import DocumentAgent
from rex_ai.agents.marketing import audience as aud
from rex_ai.agents.marketing import checks, llm, rules
from rex_ai.agents.marketing.agent import MarketingAgent, NotFound
from rex_ai.agents.marketing.facts import collect
from rex_ai.agents.marketing.writer import fmt_date, money
from rex_ai.core import db
from rex_ai.core.schemas import (CampaignEdit, CampaignPublish, CampaignRequest, Channel, ContentVariant,
                                 CurriculumRequest, DocumentRequest, ReviewAction)

CLAIM = re.compile(r"\b(?:accredited|certified|endorsed)\s+by\b|officially accredited|guaranteed (?:job|pass)", re.I)
ARABIC = re.compile(r"[؀-ۿ]")


def _text(camp) -> str:
    return "\n".join(f"{v.headline or ''}\n{v.body}\n{v.cta}" for v in camp.variants)


def _approve_curriculum(cid: str) -> None:
    CurriculumAgent().review(cid, ReviewAction(action="approved", reviewer="Dean"))


@pytest.fixture(scope="module")
def agent():
    return MarketingAgent()


@pytest.fixture(scope="module")
def campaign(agent):
    return agent.generate(CampaignRequest(course_id="DS-101", budget=5000))


# =========================================================================== content (items 38, 40)
def test_every_channel_with_variations(campaign):
    assert {v.channel for v in campaign.variants} == set(Channel)                   # captions, ads, LinkedIn, ...
    assert len(campaign.variants) == len(Channel) * 2                               # 2 variations by default
    assert len({v.key for v in campaign.variants}) == len(campaign.variants)
    for ch in Channel:
        mine = [v for v in campaign.variants if v.channel == ch]
        assert len({v.angle for v in mine}) == 2                                    # variations differ (A/B)
        assert len({v.body for v in mine}) == 2
    assert campaign.headlines and campaign.ctas                                     # headlines + CTAs
    assert all(v.cta and v.headline for v in campaign.variants)


def test_channel_subset_and_variation_count(agent):
    c = agent.generate(CampaignRequest(course_id="SW-201", channels=["linkedin_post", "email", "linkedin_post"],
                                       variations_per_channel=3))
    assert [v.key for v in c.variants] == ["linkedin_post:1", "linkedin_post:2", "linkedin_post:3",
                                           "email:1", "email:2", "email:3"]
    assert c.channels == [Channel.linkedin_post, Channel.email]


def test_channel_rules_are_respected(campaign):
    for v in campaign.variants:
        spec = rules.channel(v.channel.value)
        assert len(v.headline) <= spec["headline_max"], v.key
        assert len(v.body) <= spec["body_max"] and v.char_count == len(v.body), v.key
        assert len(v.hashtags) <= spec["hashtags"], v.key
    ads = [v for v in campaign.variants if v.channel == Channel.facebook_instagram_ad]
    assert all(v.cta in rules.channel("facebook_instagram_ad")["ctas"] for v in ads)   # a real Meta button
    assert all("UNSUBSCRIBE" in v.body for v in campaign.variants if v.channel == Channel.email)
    assert all("STOP" in v.body for v in campaign.variants if v.channel == Channel.whatsapp)
    social = [v for v in campaign.variants if v.channel in (Channel.social_caption, Channel.linkedin_post)]
    assert all(v.hashtags and v.hashtags[-1] in v.body for v in social)


# =========================================================================== approved facts only (items 36, 41)
def test_copy_uses_only_approved_facts(campaign, agent):
    facts = campaign.course_facts_used
    allowed = checks.allowed_numbers(facts)
    for v in campaign.variants:
        assert checks.unknown_numbers(v, allowed) == [], v.key                     # nothing invented
        assert not any(c.startswith("numbers not in") for c in v.checks)
    text = _text(campaign)
    course = agent.kb.courses["DS-101"]
    assert course.title_en in text
    assert money(course.fee, course.currency, "en") in text                        # fee copied, not estimated
    if facts["next_start"]:
        assert fmt_date(facts["next_start"]["start_date"], "en") in text
    if facts["promotion"]:
        assert money(facts["best_price"], course.currency, "en") in text


def test_campaign_is_traceable(campaign):
    ids = {s.source_id for s in campaign.sources}
    assert {"SRC-ACAD-COURSES", "SRC-ACAD-PRICING", "SRC-MKT-CHANNELS", "SRC-MKT-AUDIENCES"} <= ids
    assert "SRC-MKT-GUIDE" in ids                                                   # brand guide via RAG
    assert "SRC-ACAD-FAQ" in ids                                                    # objections via RAG
    assert campaign.course_facts_used["course_id"] == "DS-101"
    assert campaign.course_facts_used["data_status"] == "sample"
    assert any("SAMPLE" in n for n in campaign.metadata["notes"])


def test_no_accreditation_or_guarantee_claims(campaign):
    assert not CLAIM.search(_text(campaign))
    for bad, good in [("This course is accredited by NAQAAE.", "aligned with"),
                      ("We guarantee a job after the course.", "prepares you for the job market"),
                      ("100% job placement for graduates", "strong career preparation"),
                      ("The best academy in Egypt", "a trusted academy"),
                      ("نضمن لك وظيفة بعد الدورة", "إعداد عملي لسوق العمل")]:
        text, n = checks.guard_marketing(bad)
        assert n >= 1 and good in text, bad
    kept, n = checks.guard_marketing("We do not guarantee a job.")
    assert n == 0 and "guarantee a job" in kept                                     # honest negation is kept


def test_invented_numbers_and_missing_opt_out_are_caught(campaign):
    allowed = checks.allowed_numbers(campaign.course_facts_used)
    v = ContentVariant(channel="whatsapp", variant_no=9, headline="Hi", body="Only 999 EGP this week!", cta="Go")
    checked, invented = checks.check_variant(v, allowed, "en")
    assert "999" in invented and any("numbers not in" in c for c in checked.checks)
    assert "STOP" in checked.body and "opt-out line added" in checked.checks
    long_ad = ContentVariant(channel="facebook_instagram_ad", variant_no=9, headline="x" * 80, body="y", cta="Sign Up")
    assert len(checks.check_variant(long_ad, allowed, "en")[0].headline) <= 40


def test_arabic_campaign(agent):
    c = agent.generate(CampaignRequest(course_id="DS-101", language="ar", channels=["social_caption", "whatsapp"]))
    course = agent.kb.courses["DS-101"]
    assert c.language.value == "ar"
    assert all(ARABIC.search(v.body) and ARABIC.search(v.cta) for v in c.variants)
    assert course.title_ar in _text(c)
    assert "جنيه" in _text(c) and "#أكاديمية_الأندلس" in _text(c)
    assert all("STOP" in v.body for v in c.variants if v.channel == Channel.whatsapp)


# =========================================================================== audience (item 37)
@pytest.mark.parametrize("text,persona", [
    ("Fresh graduates and university students", "students"),
    ("HR and training managers at banks", "corporate"),
    ("Small business owners and entrepreneurs", "entrepreneurs"),
    ("Project managers and team leads", "managers"),
    ("Accountants who want to switch careers into data", "career_switchers"),
])
def test_audience_personas(text, persona):
    a = aud.resolve(text, "", {}, "en")
    assert a["persona_id"] == persona and a["matched"]
    assert a["pains"] and a["motivations"] and a["preferred_channels"] and a["targeting"]


def test_unknown_audience_falls_back_to_default():
    a = aud.resolve("zzz qqq", "", {}, "en")
    assert not a["matched"] and a["persona_id"] == rules.default_persona()["persona_id"]


def test_audience_drives_angle_and_channel_priority(agent):
    c = agent.generate(CampaignRequest(course_id="BU-101", target_audience="Project managers and team leads"))
    plan = c.strategy_plan
    assert plan["audience"]["persona_en"] == "Managers and team leads"
    assert plan["channel_mix"][0]["channel"] == "linkedin_post"                    # managers: LinkedIn first
    assert c.target_audience == "Project managers and team leads"


def test_offer_matches_the_audience(agent):
    """A student discount is never the headline offer of a campaign for managers."""
    managers = agent.generate(CampaignRequest(course_id="DS-101", target_audience="Department managers",
                                              channels=["linkedin_post"]))
    promo = managers.course_facts_used["promotion"]
    assert promo is None or promo["for"] in (None, "groups")
    assert "student" not in _text(managers).lower()
    students = agent.generate(CampaignRequest(course_id="DS-101", target_audience="University students",
                                              channels=["social_caption"]))
    promo = students.course_facts_used["promotion"]
    assert promo is None or promo["for"] in (None, "students")


def test_pay_before_offer_is_dropped_when_too_late(agent):
    """'Pay in full 3 weeks before the cohort starts': for a cohort starting in 10 days that offer is over."""
    facts = collect(agent.kb, "DS-101")
    today = date.today()
    early = {"promo_id": "X-EARLY", "name": "Early-bird", "discount_pct": 15, "condition": "3 weeks before",
             "valid_until": (today + timedelta(days=90)).isoformat(), "for": None, "lead_days": 21}
    facts.promotions = [early]
    facts.schedules = [{"start_date": (today + timedelta(days=10)).isoformat(), "format": "online_live",
                        "days": "Sun", "time": "19:00-22:00", "location": None, "seats_left": 5}]
    facts.choose_promotion(None, today)
    assert facts.promotion is None and facts.best_price is None
    facts.schedules[0]["start_date"] = (today + timedelta(days=40)).isoformat()
    facts.choose_promotion(None, today)
    assert facts.promotion["deadline"] == (today + timedelta(days=19)).isoformat()


# =========================================================================== strategy (item 39)
def test_strategy_plan(campaign):
    p = campaign.strategy_plan
    assert p["objective"] == "registrations" and p["funnel"] == "conversion"
    assert {m["channel"] for m in p["channel_mix"]} == {c.value for c in Channel}
    assert sorted(m["priority"] for m in p["channel_mix"]) == list(range(1, 6))
    split = p["budget"]["split"]
    assert abs(sum(split.values()) - 5000) < 0.01                                   # budget fully allocated
    assert set(split) <= {"facebook_instagram_ad", "linkedin_post", "social_caption"}   # paid / boostable only
    dates = [t["date"] for t in p["timeline"] if t["date"]]
    assert dates == sorted(dates) and all(d >= date.today().isoformat() for d in dates)
    if campaign.course_facts_used["next_start"]:
        assert dates[-1] == campaign.course_facts_used["next_start"]["start_date"]
    assert "Registrations" in p["kpis"]
    assert len(p["ab_tests"]) == 5 and all("variant 1" in t["test"] for t in p["ab_tests"])
    assert p["objection_handling"] and all(o["answer"] for o in p["objection_handling"])   # FAQs via RAG
    assert any("approves" in x for x in p["compliance"])
    assert "DS-101" in campaign.strategy and "Channel mix" in campaign.strategy
    assert any(m["tips"] for m in p["channel_mix"])                                 # channel playbook via RAG


def test_budget_without_paid_channels(agent):
    c = agent.generate(CampaignRequest(course_id="CY-101", channels=["email", "whatsapp"], budget=1000))
    assert c.strategy_plan["budget"]["split"] == {} and "not allocated" in c.strategy_plan["budget"]["note"]


def test_awareness_objective_leads_with_value(agent):
    c = agent.generate(CampaignRequest(course_id="AI-101", objective="awareness", channels=["social_caption"]))
    assert c.variants[0].angle != "offer"
    assert c.variants[0].cta == "Learn more" and c.strategy_plan["funnel"] == "awareness"


def test_requested_offer_and_cta(agent):
    c = agent.generate(CampaignRequest(course_id="AI-201", offer="Free info session on 5 November",
                                       cta="Book a free call", channels=["whatsapp", "email"]))
    assert all(v.angle == "offer" for v in c.variants if v.variant_no == 1)
    assert "Free info session on 5 November" in _text(c)
    assert all(v.cta == "Book a free call" for v in c.variants)
    assert not any(c_.startswith("numbers not in") for v in c.variants for c_ in v.checks)


# =========================================================================== Curriculum / Document connection (41)
def test_uses_the_approved_curriculum(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_id="SW-201"))
    _approve_curriculum(cur.curriculum_id)
    c = agent.generate(CampaignRequest(course_id="SW-201", channels=["linkedin_post"]))
    f = c.course_facts_used
    assert c.curriculum_id == cur.curriculum_id and f["origin"] == "curriculum"
    assert f["outcomes"] == cur.learning_outcomes and f["curriculum_status"] == "approved"
    assert any(s.source_id == "REX-CURRICULUM" for s in c.sources)
    assert cur.learning_outcomes[0].rstrip(".") in _text(c)
    handoffs = agent.pending_handoffs()
    assert any(h["kind"] == "curriculum" and h["id"] == cur.curriculum_id and h["campaigns"] >= 1 for h in handoffs)


def test_unapproved_curriculum_blocks_approval(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_id="CL-101"))
    c = agent.generate(CampaignRequest(course_id="CL-101", curriculum_id=cur.curriculum_id, channels=["email"]))
    assert any("not approved until the curriculum" in n for n in c.metadata["notes"])
    with pytest.raises(ValueError, match="approve the curriculum first"):
        agent.review(c.campaign_id, ReviewAction(action="approved", reviewer="Admin"))
    _approve_curriculum(cur.curriculum_id)
    assert agent.review(c.campaign_id, ReviewAction(action="approved", reviewer="Admin")).status.value == "approved"


def test_curriculum_of_another_course_is_refused(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_id="CY-101"))
    with pytest.raises(ValueError, match="belongs to"):
        agent.generate(CampaignRequest(course_id="DS-101", curriculum_id=cur.curriculum_id))
    with pytest.raises(NotFound):
        agent.generate(CampaignRequest(course_id="XX-999"))


def test_document_handoff_to_marketing(agent):
    cur = CurriculumAgent().generate(CurriculumRequest(course_id="AI-201"))
    _approve_curriculum(cur.curriculum_id)
    docs = DocumentAgent()
    doc = docs.generate(DocumentRequest(doc_type="course_description", curriculum_id=cur.curriculum_id))
    docs.review(doc.document_id, ReviewAction(action="approved", reviewer="Dean"))   # HandoffEvent -> marketing
    waiting = [h for h in agent.pending_handoffs() if h["kind"] == "document" and h["id"] == doc.document_id]
    assert waiting and waiting[0]["campaigns"] == 0
    c = agent.generate(CampaignRequest(course_id="AI-201", document_id=doc.document_id, channels=["email"]))
    assert c.document_id == doc.document_id and c.curriculum_id == cur.curriculum_id
    assert c.metadata["handoff_event"]
    assert agent.history(c.campaign_id)["handoffs"][0]["to_agent"] == "marketing"
    assert [h for h in agent.pending_handoffs() if h["id"] == doc.document_id][0]["campaigns"] == 1


# =========================================================================== approval before publishing (42)
def test_review_then_publish(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MARKETING_OUTPUT_DIR", tmp_path)
    c = agent.generate(CampaignRequest(course_id="CL-201", channels=["linkedin_post", "email"]))
    assert c.status.value == "pending_approval"                                     # never published by default
    with pytest.raises(ValueError, match="Only approved"):
        agent.publish(c.campaign_id, CampaignPublish(publisher="Admin"))
    with pytest.raises(ValueError):
        agent.review(c.campaign_id, ReviewAction(action="draft", reviewer="Admin"))
    c = agent.review(c.campaign_id, ReviewAction(action="approved", reviewer="Admin", comment="Go"))
    assert c.metadata["approved_by"] == "Admin" and c.reviewer_comments
    with pytest.raises(ValueError, match="already approved"):
        agent.review(c.campaign_id, ReviewAction(action="changes_requested", reviewer="Admin"))
    with pytest.raises(LookupError):
        agent.publish(c.campaign_id, CampaignPublish(publisher="Admin", variant_keys=["whatsapp:1"]))
    with pytest.raises(LookupError, match="No e-mail variant"):
        agent.publish(c.campaign_id, CampaignPublish(publisher="Admin", variant_keys=["linkedin_post:1"],
                                                     email_to=["lead@example.com"]))
    c = agent.publish(c.campaign_id, CampaignPublish(publisher="Admin", variant_keys=["linkedin_post:1", "email:2"],
                                                     email_to=["lead@example.com"]))
    pub = c.metadata["publications"][-1]
    folder = Path(pub["folder"])
    assert pub["variants"] == ["linkedin_post:1", "email:2"]
    assert {"campaign.json", "content.csv", "linkedin_post_1.txt", "email_2.txt"} <= {p.name for p in folder.iterdir()}
    assert json.loads((folder / "campaign.json").read_text(encoding="utf-8"))["status"] == "approved"
    assert pub["email"]["channel"] == "outbox" and Path(pub["email"]["file"]).exists()   # no SMTP in tests
    assert agent.history(c.campaign_id)["publications"]


def test_edit_creates_a_new_version(agent):
    c = agent.generate(CampaignRequest(course_id="PD-101", channels=["whatsapp", "social_caption"]))
    agent.review(c.campaign_id, ReviewAction(action="approved", reviewer="Admin"))
    v = c.variants[0].model_copy(update={"body": c.variants[0].body + "\nCall 01000000000 today!"})
    out = agent.edit(c.campaign_id, CampaignEdit(reviewer="Ms Salma", variants=[v], remove_variants=["social_caption:2"],
                                                 strategy="Focus on WhatsApp leads first."))
    assert out.version == 2 and out.status.value == "pending_approval"            # back to review
    assert "approved_by" not in out.metadata
    assert "social_caption:2" not in [x.key for x in out.variants]
    edited = next(x for x in out.variants if x.key == v.key)
    assert edited.generated_by.startswith("reviewer") and any("01000000000" in c_ for c_ in edited.checks)
    assert out.strategy == "Focus on WhatsApp leads first."
    nothing = agent.edit(c.campaign_id, CampaignEdit(reviewer="Ms Salma", instructions="make it shorter"))
    assert nothing.version == 2 and nothing.metadata["last_edit"]["not_applied"]   # needs the LLM
    versions = agent.history(c.campaign_id)["versions"]
    assert [x["version"] for x in versions] == [1, 2]
    with pytest.raises(ValidationError):
        CampaignEdit(reviewer="x")


def test_changed_course_facts_block_approval(agent, monkeypatch):
    c = agent.generate(CampaignRequest(course_id="SW-101", channels=["email"]))
    course = agent.kb.courses["SW-101"]
    monkeypatch.setitem(agent.kb.courses, "SW-101", course.model_copy(update={"fee": course.fee + 1000}))
    with pytest.raises(ValueError, match="facts changed"):
        agent.review(c.campaign_id, ReviewAction(action="approved", reviewer="Admin"))


def test_brief_pdf(agent, campaign):
    path = agent.pdf(campaign.campaign_id)
    assert path.exists() and path.read_bytes()[:4] == b"%PDF"
    Path(path).unlink()
    assert agent.pdf(campaign.campaign_id).exists()                                 # rebuilt when missing


# =========================================================================== LLM (optional, always checked)
def _fake_llm(monkeypatch, response):
    monkeypatch.setattr(config, "USE_LLM", True)
    monkeypatch.setattr(llm, "_chat", lambda messages, temperature=0.6: response)


def test_llm_rewrite_is_fact_checked(agent, monkeypatch):
    _fake_llm(monkeypatch, {"variants": [
        {"channel": "whatsapp", "variant_no": 1, "headline": "Hi", "body": "Only 999 EGP - guaranteed job!",
         "cta": "Book"},
        {"channel": "whatsapp", "variant_no": 2, "headline": "Your data career starts here",
         "body": "Hello! Learn data analysis with Python and Power BI. Reply 1 to talk to us.", "cta": "Talk to us"},
        {"channel": "email", "variant_no": 1, "body": "unknown channel"}]})
    c = agent.generate(CampaignRequest(course_id="DS-101", channels=["whatsapp"]))
    v1, v2 = c.variants
    assert v1.generated_by == "rules" and "999" not in v1.body                    # invented price rejected
    assert any("999" in r for r in c.metadata["llm_rejected"])
    assert v2.generated_by == "llm" and v2.headline == "Your data career starts here"
    assert "STOP" in v2.body                                                        # opt-out line restored
    assert c.metadata["generated_by"] == "llm+rules"


def test_llm_applies_reviewer_instructions(agent, monkeypatch):
    c = agent.generate(CampaignRequest(course_id="DS-101", channels=["linkedin_post"], variations_per_channel=1))
    _fake_llm(monkeypatch, {"variants": [{"channel": "linkedin_post", "variant_no": 1,
                                          "headline": "Turn spreadsheets into dashboards",
                                          "body": "Shorter post about Data Analysis with Python and Power BI."}]})
    out = agent.edit(c.campaign_id, CampaignEdit(reviewer="Admin", instructions="make it shorter"))
    assert out.version == 2 and out.variants[0].headline == "Turn spreadsheets into dashboards"
    assert "LLM applied" in out.metadata["last_edit"]["applied"][0]


# =========================================================================== leads and RAG
def test_lead_insights_without_personal_data(agent):
    db.save_lead({"session_id": "s1", "name": "Lead Person", "email": "lead.person@example.com",
                  "interested_course_ids": ["BU-201"], "consent": True,
                  "profile": {"skill_level": "Beginner", "career_goal": "Scrum master", "education_level": "graduate"}})
    c = agent.generate(CampaignRequest(course_id="BU-201", channels=["email"]))
    insights = c.course_facts_used["lead_insights"]
    assert insights["leads"] >= 1 and "Scrum master" in insights["goals"]
    dump = json.dumps(c.model_dump(mode="json"), ensure_ascii=False)
    assert "lead.person@example.com" not in dump and "Lead Person" not in dump
    assert any(s.source_id == "SRC-REX-LEADS" for s in c.sources)


def test_marketing_references_are_indexed_in_rag():
    report = rules.ensure_refs_indexed(force=True)
    assert report["chunks"] > 0 and not report["errors"]
    context, refs, tips = rules.guidance(["linkedin_post", "whatsapp"], "data analysis", "en")
    assert context and {r.source_id for r in refs} == {"SRC-MKT-GUIDE"}
    assert "LinkedIn" in tips["linkedin_post"] and "opted in" in tips["whatsapp"]


def test_request_validation():
    with pytest.raises(ValidationError):
        CampaignRequest(course_id="DS-101", channels=[])
    with pytest.raises(ValidationError):
        CampaignRequest(course_id="DS-101", variations_per_channel=9)
    assert CampaignRequest(course_id=" ds-101 ").course_id == "DS-101"


# =========================================================================== API and orchestrator
@pytest.fixture(scope="module")
def client():
    from rex_ai.api.main import app
    return TestClient(app)


def test_api_campaign_flow(client):
    assert len(client.get("/campaigns/channels").json()) == 5
    r = client.post("/campaigns", json={"course_id": "BU-102", "channels": ["social_caption", "facebook_instagram_ad"],
                                        "target_audience": "small business owners", "budget": 2000})
    assert r.status_code == 201, r.text
    cid = r.json()["campaign_id"]
    assert r.json()["status"] == "pending_approval"
    assert any(c["campaign_id"] == cid for c in client.get("/campaigns", params={"course_id": "bu-102"}).json())
    assert client.get(f"/campaigns/{cid}").json()["strategy_plan"]["audience"]["persona_en"].startswith("Entrepreneurs")
    assert client.post(f"/campaigns/{cid}/publish", json={"publisher": "Admin"}).status_code == 409
    v = client.get(f"/campaigns/{cid}").json()["variants"][0]
    r = client.patch(f"/campaigns/{cid}", json={"reviewer": "Admin", "variants": [{**v, "body": v["body"] + " !"}]})
    assert r.status_code == 200 and r.json()["version"] == 2
    r = client.post(f"/campaigns/{cid}/review", json={"action": "approved", "reviewer": "Admin"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert client.post(f"/campaigns/{cid}/publish", json={"publisher": "Admin", "variant_keys": ["x:1"]}).status_code \
        == 400
    assert client.post(f"/campaigns/{cid}/publish", json={"publisher": "Admin"}).status_code == 200
    h = client.get(f"/campaigns/{cid}/history").json()
    assert len(h["versions"]) == 2 and h["publications"]
    pdf = client.get(f"/campaigns/{cid}/pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"


def test_api_errors(client):
    assert client.post("/campaigns", json={"course_id": "XX-000"}).status_code == 404
    r = client.post("/campaigns", json={"course_id": "DS-101", "channels": []})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert client.get("/campaigns/CMP-NOPE").status_code == 404
    assert client.post("/campaigns/CMP-NOPE/review", json={"action": "approved", "reviewer": "x"}).status_code == 404


def test_orchestrator_routes_marketing_requests(client):
    r = client.post("/route", json={"session_id": "m1", "message": "Create a LinkedIn post and a WhatsApp message to "
                                                                     "promote DS-101 for fresh graduates, budget 3000 EGP"})
    out = r.json()
    assert out["agent"] == "marketing" and out["result"]["status"] == "pending_approval"
    assert out["result"]["channels"] == ["linkedin_post", "whatsapp"]
    assert out["result"]["audience"] == "Students and fresh graduates"
    assert "budget" not in out["result"]["assumed"] and "objective" in out["result"]["assumed"]
    r = client.post("/route", json={"session_id": "m2", "message": "Write a marketing campaign please"})
    assert r.json()["agent"] == "marketing" and r.json()["result"]["status"] == "need_input"
    r = client.post("/route", json={"session_id": "m3", "message": "اعمل حملة ترويج لدورة DS-101 على واتساب"})
    assert r.json()["agent"] == "marketing" and r.json()["result"]["channels"] == ["whatsapp"]
    # a lead asking about the Digital Marketing course or an ad they saw stays with REX
    for msg in ("Do you have a digital marketing course?", "I saw your facebook ad, what are the fees?"):
        assert client.post("/route", json={"session_id": "m4", "message": msg}).json()["agent"] == "rex_assistant"
