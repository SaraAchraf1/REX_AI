"""Knowledge base integrity: every course is complete and traceable to a source."""
import yaml

from rex_ai import config
from rex_ai.agents.rex.knowledge_base import load_knowledge_base

KB = load_knowledge_base()
METADATA_FIELDS = ["department", "target_audience", "skill_level", "industry", "duration_hours",
                   "learning_outcomes", "prerequisites", "country", "source_id", "last_updated"]


def test_catalogue_loads():
    assert len(KB.courses) >= 10
    assert len(KB.faqs) >= 10


def test_every_course_has_brief_metadata_fields():
    for c in KB.courses.values():
        for f in METADATA_FIELDS:
            assert getattr(c, f) not in (None, "", []), f"{c.course_id} missing {f}"


def test_every_course_has_price_instructor_and_schedule():
    for cid, c in KB.courses.items():
        assert c.fee and c.fee > 0, cid
        assert KB.instructors_for(cid), cid
        assert any(s.course_id == cid for s in KB.schedules), cid


def test_course_sources_exist_in_registry():
    reg = yaml.safe_load((config.DATA_DIR / "data_sources.yaml").read_text(encoding="utf-8"))
    ids = {s["source_id"] for group in reg.values() for s in group}
    assert {c.source_id for c in KB.courses.values()} <= ids
    for ch in KB.chunks:
        assert ch.metadata["source_id"] in ids


def test_tender_sources_are_egypt_or_saudi_only():
    reg = yaml.safe_load((config.DATA_DIR / "data_sources.yaml").read_text(encoding="utf-8"))
    for s in reg["tenders"]:
        countries = s["country"] if isinstance(s["country"], list) else [s["country"]]
        assert set(countries) <= {"EG", "SA"}, s["source_id"]


def test_no_course_claims_official_accreditation():
    for c in KB.courses.values():
        text = (c.accreditation_alignment or "").lower()
        assert "accredited by" not in text and "officially accredited" not in text, c.course_id


def test_best_price_uses_single_highest_promo():
    best, promo = KB.best_price("AI-101")  # student 20% applies
    assert promo.promo_id == "PROMO-STUDENT"
    assert best == round(KB.courses["AI-101"].fee * 0.8)
