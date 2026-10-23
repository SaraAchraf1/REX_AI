"""Recommendation logic: grounded, on-topic, and explained."""
from rex_ai.agents.rex.knowledge_base import load_knowledge_base
from rex_ai.agents.rex.profile_extractor import update_profile
from rex_ai.agents.rex.recommender import recommend
from rex_ai.core.schemas import LeadProfile, SkillLevel, TrainingFormat

KB = load_knowledge_base()


def profile_from(*messages):
    p = LeadProfile()
    for m in messages:
        p, _ = update_profile(p, m)
    return p


def test_extracts_profile_fields_english():
    p = profile_from("I'm a graduate, beginner, interested in data analysis, online, budget 12000 EGP, 6 hours a week")
    assert "data" in p.interests
    assert p.skill_level == SkillLevel.beginner
    assert p.preferred_format == TrainingFormat.online_live
    assert p.budget == 12000
    assert p.weekly_hours == 6
    assert p.education_level == "graduate"


def test_extracts_profile_fields_arabic():
    p = profile_from("أنا طالب مبتدئ ومهتم بالذكاء الاصطناعي وأفضل الحضوري وميزانيتي ١٠ الف جنيه")
    assert "ai" in p.interests
    assert p.skill_level == SkillLevel.beginner
    assert p.preferred_format == TrainingFormat.onsite
    assert p.budget == 10000


def test_data_analyst_goal_ranks_data_course_first():
    recs = recommend(KB, profile_from("I want to become a data analyst", "beginner", "online"))
    assert recs[0].course_id == "DS-101"


def test_qa_profile_gets_testing_course():
    recs = recommend(KB, profile_from("3 years experience, interested in QA automation, budget 14000 EGP"))
    assert recs[0].course_id == "SW-201"
    assert any("budget" in r.lower() for r in recs[0].reasons)


def test_recommendations_stay_on_topic():
    recs = recommend(KB, profile_from("I want to work in cybersecurity"))
    assert all(r.course_id.startswith("CY") for r in recs)


def test_every_recommendation_has_reasons_and_real_course():
    recs = recommend(KB, profile_from("interested in cloud and devops, some experience"))
    assert recs
    for r in recs:
        assert r.course_id in KB.courses
        assert r.reasons
        assert r.fee == KB.courses[r.course_id].fee


def test_beginner_is_not_pushed_to_intermediate_first():
    recs = recommend(KB, profile_from("beginner interested in AI"))
    assert recs[0].course_id == "AI-101"
