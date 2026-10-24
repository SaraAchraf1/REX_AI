"""Agent 2 - Educational RFP & Tender Research Agent (owner: Member 2). Run: pytest -q tests/test_rfp_agent.py

The sample feed (data/rfp/feeds/sample_tenders.json) contains 19 notices and data/rfp/documents/ 2 tender
documents; "today" is pinned to 2026-10-24 in conftest.py. Expected outcome of a full run:
    5 outside Egypt/Saudi Arabia, 2 not training, 1 expired, 3 duplicates  ->  10 kept (5 EG, 5 SA)
"""
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.agents.rfp.agent import RFPResearchAgent
from rex_ai.agents.rfp.dedupe import dedupe_hash, is_duplicate
from rex_ai.agents.rfp.extract import RawOpportunity, classify, detect_country, extract_fields, parse_date
from rex_ai.agents.rfp.qualify import Qualifier
from rex_ai.agents.rfp.sources import FeedSource, WebPageSource
from rex_ai.core import db
from rex_ai.core.schemas import (Country, QualificationFit, RFPAskRequest, RFPCategory, RFPRunRequest, RFPStatus)

TODAY = date(2026, 10, 24)


def raw(title="Cybersecurity training for bank staff", **kw):
    return RawOpportunity(title=title, source_id="SRC-RFP-SAMPLE", **kw)


# =========================================================================== country rule (strict)
@pytest.mark.parametrize("country, expected", [
    ("Egypt", Country.EG), ("EG", Country.EG), ("مصر", Country.EG), ("Saudi Arabia", Country.SA),
    ("KSA", Country.SA), ("المملكة العربية السعودية", Country.SA),
    ("United Arab Emirates", None), ("France", None), ("USA", None), ("Jordan", None),
])
def test_country_field(country, expected):
    assert detect_country(raw(country=country))[0] == expected


def test_country_field_wins_over_text():
    # a French tender that mentions an Egyptian partner is still outside scope
    c, why = detect_country(raw(country="France", description="Partner universities in Cairo, Egypt"))
    assert c is None and "France" in why


def test_country_inferred_from_city():
    assert detect_country(raw(description="Workshops at the Riyadh headquarters"))[0] == Country.SA
    assert detect_country(raw(description="التدريب في القاهرة"))[0] == Country.EG


def test_unknown_or_other_country_is_rejected():
    assert detect_country(raw(description="Training for county staff in Nairobi, Kenya"))[0] is None
    c, why = detect_country(raw(description="Online training"))
    assert c is None and "unknown" in why


# =========================================================================== relevance and category
def test_supply_and_service_tenders_are_not_training():
    assert not classify(raw("Supply and installation of AC units for training halls",
                            description="Supply, installation and maintenance")).relevant
    assert not classify(raw("Catering services for the training center", organization="Skills Development Fund",
                            description="Daily catering for trainees")).relevant
    assert not classify(raw("Construction of a new road", description="civil works")).relevant


@pytest.mark.parametrize("title, category", [
    ("Cybersecurity Awareness and SOC Analyst Training", RFPCategory.software_it_training),
    ("Digital transformation skills training for civil servants", RFPCategory.digital_transformation),
    ("PMP project management training for engineers", RFPCategory.professional_development),
    ("Vocational technical training for technicians", RFPCategory.technical_training),
    ("E-learning content and LMS training", RFPCategory.corporate_learning),
    ("تقديم دورات تدريبية في الأمن السيبراني", RFPCategory.software_it_training),
])
def test_category(title, category):
    c = classify(raw(title))
    assert c.relevant and c.category == category and 0 < c.score <= 1


def test_extra_keywords_make_a_tender_relevant():
    r = raw("ISO 27001 lead implementer programme")
    assert not classify(r).relevant
    assert classify(r, extra_keywords=["lead implementer"]).relevant


# =========================================================================== dates
@pytest.mark.parametrize("value, expected", [
    ("2026-11-20", date(2026, 11, 20)), ("20/11/2026", date(2026, 11, 20)), ("15 November 2026", date(2026, 11, 15)),
    ("November 15, 2026", date(2026, 11, 15)), ("١٥ نوفمبر ٢٠٢٦", date(2026, 11, 15)), ("TBD", None), (None, None),
])
def test_parse_date(value, expected):
    assert parse_date(value)[0] == expected


def test_hijri_dates_are_converted_and_marked_approximate():
    d, approx = parse_date("1448/06/20 هـ")
    assert approx and date(2026, 11, 29) <= d <= date(2026, 12, 3)
    assert parse_date("1447/09/01")[0] == date(2026, 2, 18)          # 1 Ramadan 1447


# =========================================================================== duplicates
def test_dedupe_hash_ignores_case_punctuation_and_word_order():
    a = dedupe_hash("Cloud & DevOps Upskilling - Program", "Cairo Software Council", date(2026, 12, 5))
    b = dedupe_hash("program upskilling devops cloud", "CAIRO SOFTWARE COUNCIL", date(2026, 12, 5))
    assert a == b
    assert a != dedupe_hash("program upskilling devops cloud", "Cairo Software Council", date(2026, 12, 6))


def test_fuzzy_duplicate_across_portals():
    a = {"title": "Digital Skills Training Program for 2,000 Government Employees",
         "issuing_organization": "National Digital Government Agency", "country": "EG", "deadline": "2026-11-20"}
    b = {"title": "Digital skills training programme for 2000 government employees",
         "issuing_organization": "National Digital Government Agency (NDGA)", "country": "EG", "deadline": "2026-11-20"}
    assert is_duplicate(a, b)
    assert not is_duplicate(a, {**b, "country": "SA"})
    assert not is_duplicate(a, {**b, "title": "Cybersecurity training for bank staff"})


# =========================================================================== extraction from documents
def test_extract_fields_from_arabic_tender_document():
    text = Path(config.RFP_DIR / "documents" / "SA_data_analytics_training_tender_ar.txt").read_text(encoding="utf-8")
    f = extract_fields(text)
    assert f["organization"] == "الهيئة الوطنية للبيانات المفتوحة"
    assert detect_country(raw(country=f["country"]))[0] == Country.SA
    assert parse_date(f["deadline"])[1] is True
    assert "منصة اعتماد" in f["eligibility"]
    assert f["submission_link"].startswith("https://")
    assert len(f["documents"]) == 2


def test_feed_columns_accept_portal_export_names(tmp_path):
    (tmp_path / "etimad_export.csv").write_text(
        "اسم المنافسة,الجهة,الدولة,آخر موعد لتقديم العروض\nتدريب الموظفين على تحليل البيانات,جهة تجريبية,السعودية,2026-12-30\n",
        encoding="utf-8")
    [r] = FeedSource(tmp_path).fetch()
    assert r.organization == "جهة تجريبية" and r.country == "السعودية" and r.deadline == "2026-12-30"


# =========================================================================== qualification
@pytest.fixture(scope="module")
def qualifier():
    return Qualifier(profile={"registered_countries": ["EG"], "years_in_business": 8,
                              "certifications": ["ISO 9001:2015"], "licences": [], "max_trainees_per_year": 1500},
                     use_rag=False)


def test_qualifies_when_courses_match_and_requirements_met(qualifier):
    text = "Cybersecurity awareness training. Minimum 5 years of experience; ISO 9001 certified"
    a = qualifier.assess(text, "Minimum 5 years of experience; ISO 9001 certified", "EG")
    assert a.fit == QualificationFit.likely and "CY-101" in a.course_ids
    assert any(s.source_id == "SRC-ACAD-COURSES" for s in a.sources)


def test_saudi_registration_blocks_an_egypt_only_academy(qualifier):
    text = "Software testing bootcamp. Saudi commercial registration and an active Etimad account required"
    a = qualifier.assess(text, "Saudi commercial registration and an active Etimad account", "SA")
    assert a.fit == QualificationFit.unlikely and any("Saudi-registered" in n for n in a.notes)


def test_missing_certificate_and_experience_are_blockers(qualifier):
    text = "Cloud training. Bidders need ISO 27001 and at least 10 years of experience"
    a = qualifier.assess(text, text, "EG")
    assert a.fit == QualificationFit.unlikely
    assert any("ISO 27001" in n for n in a.notes) and any("10 years" in n for n in a.notes)


def test_no_matching_course_is_unlikely(qualifier):
    a = qualifier.assess("Nursing clinical skills training", "None", "EG")
    assert a.fit == QualificationFit.unlikely and not a.course_ids


def test_unpublished_eligibility_or_capacity_gap_is_possible(qualifier):
    assert qualifier.assess("Power BI training", None, "EG").fit == QualificationFit.possible
    a = qualifier.assess("Data analysis training for 2,000 employees", "Training providers", "EG")
    assert a.fit == QualificationFit.possible and any("capacity" in n for n in a.notes)


# =========================================================================== full pipeline on the sample data
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs against a fresh database: the first discovers, the second must only refresh."""
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DB_PATH", tmp_path_factory.mktemp("rfpdb") / "rfp.db")
    db.init_db()
    agent = RFPResearchAgent(today=TODAY)
    first = agent.run(RFPRunRequest())
    egypt_id = first.new_flagged_egypt[0]
    db.update_rfp_status(egypt_id, "qualified")                    # a reviewer decision
    second = agent.run(RFPRunRequest())
    yield {"agent": agent, "first": first, "second": second, "reviewed": egypt_id}
    mp.undo()


def test_run_counts(runs):
    r = runs["first"]
    assert (r.found, r.kept) == (21, 10)
    assert (r.rejected_country, r.rejected_irrelevant, r.expired, r.duplicates) == (5, 2, 1, 3)
    assert r.found == r.kept + r.rejected_country + r.rejected_irrelevant + r.expired + r.duplicates
    assert {x["rejected"] for x in r.rejections} == {"rejected_country", "rejected_irrelevant", "expired", "duplicates"}


def test_only_egypt_and_saudi_are_kept(runs):
    records = runs["first"].records
    assert {r.country for r in records} == {Country.EG, Country.SA}
    titles = " ".join(r.title for r in records)
    for outside in ("Dubai", "Public Servants", "Workforce Training", "County Staff", "Vocational Instructors"):
        assert outside not in titles


def test_egypt_flagged_immediately_saudi_monitored(runs):
    r = runs["first"]
    assert len(r.new_flagged_egypt) == 5 and len(r.new_monitoring_saudi) == 5
    for rec in r.records:
        if rec.country == Country.EG:
            assert rec.priority == "high"
        else:
            assert rec.status == RFPStatus.monitoring and rec.priority == "normal"
    assert r.records[0].country == Country.EG                         # Egypt first in the pipeline
    alerts = db.list_rfp_alerts(100)
    assert sum(a["priority"] == "high" for a in alerts) == 5 and r.alerts_sent == 10


def test_every_record_is_traceable_and_summarised(runs):
    for rec in runs["first"].records:
        assert rec.source.source_id in ("SRC-RFP-SAMPLE", "SRC-RFP-DOCS")
        assert rec.summary and rec.issuing_organization in rec.summary
        assert rec.qualification_notes and rec.dedupe_hash and rec.first_seen


def test_document_merged_into_feed_notice(runs):
    cyber = next(r for r in runs["first"].records if r.title.startswith("Cybersecurity Awareness"))
    assert "DIB_cybersecurity_training_RFP.md" in cyber.available_documents
    assert "Egyptian company" in cyber.eligibility_requirements             # filled in from the document
    assert cyber.qualification == QualificationFit.likely and "CY-101" in cyber.matched_course_ids


def test_hijri_deadline_and_document_only_tender(runs):
    ar = next(r for r in runs["first"].records if r.source.source_id == "SRC-RFP-DOCS")
    assert ar.country == Country.SA and ar.deadline and "Hijri" in ar.qualification_notes
    assert ar.qualification == QualificationFit.unlikely                    # needs Saudi registration


def test_second_run_refreshes_without_new_alerts_and_keeps_review(runs):
    first, second = runs["first"], runs["second"]
    assert second.kept == first.kept and second.already_known == first.kept
    assert not second.new_flagged_egypt and second.alerts_sent == 0
    assert db.get_rfp(runs["reviewed"])["status"] == "qualified"


def test_request_filters(runs):
    agent = runs["agent"]
    saudi = agent.run(RFPRunRequest(countries=[Country.SA], notify=False))
    # country is checked first: 5 outside scope + all 10 Egyptian notices (even expired/duplicate ones)
    assert {r.country for r in saudi.records} == {Country.SA} and saudi.rejected_country == 5 + 10
    soon = agent.run(RFPRunRequest(countries=[Country.EG], deadline_to=date(2026, 12, 8), notify=False))
    assert soon.records and all(r.deadline and r.deadline <= date(2026, 12, 8) for r in soon.records)
    assert any("not stated" in x["reason"] for x in soon.rejections)      # Power BI tender has no deadline
    it_only = agent.run(RFPRunRequest(categories=[RFPCategory.software_it_training], notify=False))
    assert {r.training_category for r in it_only.records} == {RFPCategory.software_it_training}


def test_keywords_filter_the_run(runs):
    agent = runs["agent"]
    none = agent.run(RFPRunRequest(keywords=["ISO 27001"], notify=False))
    # filters run before duplicate detection: 10 tenders + their 3 duplicate copies
    assert none.kept == 0 and none.rejected_filters == 13 and none.duplicates == 0
    bi = agent.run(RFPRunRequest(keywords=["power bi", "ذكاء الأعمال"], notify=False))
    assert bi.kept == 2 and all("Power BI" in r.title or "ذكاء الأعمال" in r.title for r in bi.records)


def test_priority_organisations_rank_first(runs):
    res = runs["agent"].run(RFPRunRequest(priority_organizations=["bank"], notify=False))
    assert res.records[0].issuing_organization == "Delta Investment Bank"


def test_stored_records_filtered_like_a_run(runs):
    agent = runs["agent"]
    stored = agent.search()
    saudi = agent.filter_stored(stored, RFPRunRequest(countries=[Country.SA]))
    assert saudi and {r.country for r in saudi} == {Country.SA}
    soon = agent.filter_stored(stored, RFPRunRequest(deadline_to=date(2026, 12, 8)))
    assert soon and all(r.deadline and r.deadline <= date(2026, 12, 8) for r in soon)
    assert agent.filter_stored(stored, RFPRunRequest(keywords=["ISO 27001"])) == []


def test_dashboard_search_and_csv(runs):
    agent = runs["agent"]
    assert {r.country for r in agent.search(country="EG")} == {Country.EG}
    assert any("Power BI" in r.title for r in agent.search(q="Power BI"))
    csv_text = agent.export_csv(agent.search())
    assert csv_text.startswith("\ufeffrfp_id,") and "الهيئة الوطنية" in csv_text


def test_ask_uses_rag_with_citations(runs):
    res = runs["agent"].ask(RFPAskRequest(question="cybersecurity SOC analyst training"))
    assert res.rfp_ids and res.sources
    cyber = next(r for r in runs["first"].records if r.title.startswith("Cybersecurity Awareness"))
    assert cyber.rfp_id in res.rfp_ids


# =========================================================================== live portal source (no network)
class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, pages):
        self.pages, self.headers = pages, {}

    def get(self, url, timeout=None):
        return FakeResponse(*self.pages.get(url, ("", 404)))


LISTING = """<html><body><table>
<tr><td><a href="/t/1">Digital skills training program for ministry employees</a></td><td>Closing 2026-12-01</td></tr>
<tr><td><a href="/t/2">Supply of office furniture for the new building</a></td></tr>
<tr><td><a href="/about">About the portal and its terms of use</a></td></tr>
</table></body></html>"""


def test_web_source_keeps_training_links_and_reads_detail_page():
    pages = {"https://portal.example/list": (LISTING,), "https://portal.example/robots.txt": ("User-agent: *\nAllow: /",),
             "https://portal.example/t/1": ("Issuing Organization: Example Ministry\nDeadline: 2026-12-01",)}
    src = WebPageSource({"source_id": "SRC-EG-ETENDERS", "country": "EG", "url": "https://portal.example/list"},
                        session=FakeSession(pages))
    [item] = src.fetch()
    assert item.submission_link == "https://portal.example/t/1" and item.country == "EG"
    assert item.organization == "Example Ministry" and item.deadline == "2026-12-01"


def test_robots_disallow_is_respected_and_reported():
    pages = {"https://portal.example/list": (LISTING,),
             "https://portal.example/robots.txt": ("User-agent: *\nDisallow: /",)}
    src = WebPageSource({"source_id": "SRC-EG-ETENDERS", "country": "EG", "url": "https://portal.example/list"},
                        session=FakeSession(pages))
    result = RFPResearchAgent(sources=[src], today=TODAY).run(RFPRunRequest(notify=False))
    assert result.found == 0 and result.source_errors[0]["source_id"] == "SRC-EG-ETENDERS"


# =========================================================================== API
@pytest.fixture(scope="module")
def client(runs):
    from rex_ai.api.main import app
    return TestClient(app)


def test_api_run_list_get_and_review(client):
    body = client.post("/rfp/run", json={"countries": ["EG"], "notify": False}).json()
    assert body["kept"] == 5 and body["rejected_country"] > 0
    listed = client.get("/rfp", params={"country": "EG"}).json()
    assert listed and all(r["country"] == "EG" for r in listed)
    rid = listed[0]["rfp_id"]
    assert client.get(f"/rfp/{rid}").json()["rfp_id"] == rid
    r = client.patch(f"/rfp/{rid}/status", json={"status": "rejected", "reviewer": "bid team", "comment": "no bid"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert client.get("/rfp/RFP-XX-NOPE").status_code == 404


def test_api_rejects_countries_outside_scope(client):
    r = client.post("/rfp/run", json={"countries": ["AE"]})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"


def test_api_export_alerts_and_ask(client):
    r = client.get("/rfp/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert isinstance(client.get("/rfp/alerts").json(), list)
    ans = client.post("/rfp/ask", json={"question": "Power BI data analysis"}).json()
    assert ans["answer"] and ans["sources"]


def test_api_upload(client):
    ok = client.post("/rfp/upload", files={"file": ("new_tender.txt", "Tender Title: Test\nCountry: Egypt".encode())})
    assert ok.status_code == 200 and ok.json()["kind"] == "document"
    assert (config.RFP_DIR / "documents" / ok.json()["file"]).exists()
    bad = client.post("/rfp/upload", files={"file": ("virus.exe", b"MZ")})
    assert bad.status_code == 400 and bad.json()["code"] == "bad_file"
    (config.RFP_DIR / "documents" / ok.json()["file"]).unlink()


def test_route_answers_tender_questions(client):
    body = client.post("/route", json={"session_id": "rfp-1", "message": "find training tenders in Saudi Arabia"}).json()
    assert body["agent"] == "rfp_research" and body["result"]["rfp_ids"]
    assert all(db.get_rfp(i)["country"] == "SA" for i in body["result"]["rfp_ids"])
