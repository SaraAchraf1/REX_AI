"""Reading a chat request: which agent it is for and which fields it gives (owner: Member 3).

Used by POST /route and by the task orchestrator's rule planner (rex_ai/agents/orchestrator/planner.py).
"""
from __future__ import annotations

from rex_ai.agents.rex.text_utils import contains_any
from rex_ai.agents.rfp.taxonomy import COUNTRY_PLACES
from rex_ai.core import data_sync, db
from rex_ai.core.schemas import AgentName, Channel


def _courses() -> dict:
    return data_sync.get_assistant().kb.courses


DOC_TYPE_HINTS = {
    "course_description": ["course description", "course brochure", "وصف الدورة", "وصف دورة"],
    "course_proposal": ["proposal", "عرض تدريبي", "عرض فني"],
    "curriculum_pdf": ["curriculum pdf", "curriculum document"],
    "participant_guide": ["participant guide", "learner guide", "student guide", "trainee guide", "workbook",
                          "دليل المتدرب", "دليل المشارك"],
    "facilitator_guide": ["facilitator guide", "instructor guide", "trainer guide", "دليل المدرب", "دليل الميسر"],
    "presentation_outline": ["presentation", "slides", "slide deck", "عرض تقديمي", "شرائح"],
    "question_bank": ["question bank", "quiz questions", "exam questions", "بنك الأسئلة", "بنك الاسئلة", "بنك أسئلة"],
    "assessment_rubric": ["rubric", "معايير التقييم", "روبريك"],
    "capstone_brief": ["capstone brief", "project brief", "موجز المشروع"],
    "outcomes_assessment_plan": ["assessment plan", "outcomes plan", "خطة التقييم", "خطة نواتج التعلم"],
}
# Order matters: a document or campaign request often names its curriculum ("participant guide for curriculum
# CUR-...", "campaign for curriculum CUR-..."), so both are checked before the curriculum keywords.
ROUTING_HINTS = {
    AgentName.rfp_research: ["tender", "rfp", "rfq", "مناقصه", "مناقصة", "مناقصات", "منافسة", "منافسات"],
    AgentName.document: ["pdf"] + [h for hints in DOC_TYPE_HINTS.values() for h in hints],
    # Requests only: not bare "marketing", "offer" or "ad" - a lead asking about the Digital Marketing course, current
    # offers or "your Facebook ad" stays with REX
    AgentName.marketing: ["campaign", "ad copy", "linkedin post", "caption", "whatsapp message", "حمله", "حملة",
                          "promote", "marketing content", "marketing copy", "marketing email", "ترويج", "روّج"],
    AgentName.curriculum: ["curriculum", "syllabus design", "accreditation", "منهج", "اعتماد"],
}


CURRICULUM_QUESTIONS = {
    "course_title": "What course would you like to create?",
    "target_audience": "Who is the target audience?",
    "skill_level": "What is the intended skill level (beginner, intermediate, advanced)?",
    "total_hours": "How many training hours are available?",
}
CURRICULUM_QUESTIONS_AR = {
    "course_title": "ما الدورة التي تريد تصميم منهجها؟",
    "target_audience": "من هي الفئة المستهدفة؟",
    "skill_level": "ما المستوى المطلوب (مبتدئ، متوسط، متقدم)؟",
    "total_hours": "كم عدد ساعات التدريب المتاحة؟",
}


def parse_curriculum_message(message: str) -> dict:
    """'Design a curriculum for Power BI for finance staff, beginner, 30 hours, ISTQB' -> request fields."""
    import re
    from rex_ai.agents.curriculum.frameworks import load_frameworks
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": detect_language(message)}
    m = re.search(r"(?:curriculum|syllabus|منهج(?:ا)?)\s+(?:for|on|about|of|لـ?|عن|في)\s+(?:an?\s+|the\s+)?"
                  r"(.+?)(?=\s+for\s+|\s+لـ?\s*|,|\.|;|\s+with\s+|\s+at\s+|\s+\d|$)", message, re.I)
    if m and m.group(1).strip():
        out["course_title"] = m.group(1).strip(" '\"")
    rest = message[m.end():] if m else message
    m = re.search(r"\bfor\s+(?!a\b|an\b|the\b)(.+?)(?=,|\.|;|،|\s+\d|\s+at\s+|\s+with\s+|$)", rest, re.I)
    if m:
        out["target_audience"] = m.group(1).strip()
    elif m := re.search(r"(?:^|\s)لل(\w+)", rest):          # "للمحاسبين" = "for the accountants"
        out["target_audience"] = "ال" + m.group(1)
    if h := re.search(r"(\d{1,4})\s*(?:hours?|hrs?|h\b|ساعة|ساعات)", message, re.I):
        out["total_hours"] = int(h.group(1))
    for level, words in (("Beginner", ["beginner", "مبتدئ"]), ("Intermediate", ["intermediate", "متوسط"]),
                         ("Advanced", ["advanced", "متقدم"])):
        if contains_any(message, words):
            out["skill_level"] = level
    for f in load_frameworks().values():
        if contains_any(message, [f.short_name, *f.aliases]) and f.framework_id != "REX-QUALITY":
            out["accreditation_framework"] = f.framework_id
            break
    for code, places in COUNTRY_PLACES.items():
        if contains_any(message, places):
            out["country"] = code
    return out


DOCUMENT_QUESTIONS = {
    "doc_type": "Which document type do you need? (course description, proposal, curriculum PDF, participant guide, "
                "facilitator guide, presentation outline, question bank, rubrics, capstone brief, outcomes and "
                "assessment plan)",
    "source": "Which course or curriculum should be used? (a course id such as DS-101, or a curriculum id CUR-...)",
}
DOCUMENT_QUESTIONS_AR = {
    "doc_type": "ما نوع المستند المطلوب؟ (وصف الدورة، عرض تدريبي، المنهج PDF، دليل المتدرب، دليل المدرب، مخطط العرض "
                "التقديمي، بنك الأسئلة، معايير التقييم، موجز المشروع الختامي، خطة النواتج والتقييم)",
    "source": "ما الدورة أو المنهج المطلوب؟ (رمز دورة مثل DS-101 أو رقم منهج CUR-...)",
}


def parse_document_message(message: str) -> dict:
    """'Participant guide for DS-101 in Arabic' -> {doc_type, course_id, language}. Uses stored ids and catalogue
    titles, so nothing already known is asked again."""
    import re
    from rex_ai.agents.curriculum.research import _words
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": "ar" if re.search(r"\b(arabic)\b|بالعربي|باللغة العربية", message, re.I)
                 else "en" if re.search(r"\benglish\b|بالإنجليزي|بالانجليزي", message, re.I)
                 else detect_language(message)}
    for doc_type, hints in DOC_TYPE_HINTS.items():
        if contains_any(message, hints):
            out["doc_type"] = doc_type
            break
    if m := re.search(r"\bCUR-[0-9A-F]{8}\b", message, re.I):
        out["curriculum_id"] = m.group(0).upper()
    courses = _courses()
    if m := re.search(r"\b([A-Z]{2,4}-\d{3})\b", message, re.I):
        if m.group(1).upper() in courses:
            out["course_id"] = m.group(1).upper()
    if not (out.get("course_id") or out.get("curriculum_id")):
        words = _words(message)
        best = max(((len(words & _words(f"{c.title_en} {c.title_ar}")), c.course_id) for c in courses.values()),
                   default=(0, None))
        if best[0] >= 2:
            out["course_id"] = best[1]
    if m := re.search(r"\bfor\s+(?:the\s+)?(.+?)(?:\s+(?:team|staff|department))\b", message, re.I):
        out["audience"] = m.group(0)[4:].strip()
    return out


MARKETING_QUESTIONS = {
    "course_id": "Which course are we promoting? (a course id such as DS-101, or its title)",
}
MARKETING_QUESTIONS_AR = {
    "course_id": "ما الدورة التي نروّج لها؟ (رمز دورة مثل DS-101 أو اسمها)",
}
# Asked as follow-ups when not given (brief §10); sensible defaults are used meanwhile and reported as "assumed"


CHANNEL_HINTS = {
    Channel.social_caption: ["caption", "social media post", "instagram post", "facebook post", "منشور", "بوست"],
    Channel.facebook_instagram_ad: ["facebook ad", "instagram ad", "meta ad", "ad copy", "ads", "إعلان", "اعلان"],
    Channel.linkedin_post: ["linkedin", "لينكد"],
    Channel.email: ["email", "e-mail", "newsletter", "بريد", "ايميل", "إيميل"],
    Channel.whatsapp: ["whatsapp", "واتساب", "واتس"],
}
OBJECTIVE_HINTS = {"awareness": ["awareness", "reach", "brand", "الوعي", "انتشار"],
                   "webinar sign-ups": ["webinar", "info session", "ندوة", "جلسة تعريفية"],
                   "leads": ["leads", "lead generation", "inquiries", "عملاء محتملين"],
                   "registrations": ["registration", "enrol", "enroll", "sign up", "تسجيل"]}


def parse_marketing_message(message: str) -> dict:
    """'LinkedIn post and WhatsApp message to promote DS-101 for fresh graduates, budget 5000 EGP' -> request fields.
    Uses stored course ids and catalogue titles, so nothing already known is asked again."""
    import re
    from rex_ai.agents.curriculum.research import _words
    from rex_ai.agents.rex.text_utils import detect_language
    out: dict = {"language": "ar" if re.search(r"\barabic\b|بالعربي|باللغة العربية", message, re.I)
                 else "en" if re.search(r"\benglish\b|بالإنجليزي|بالانجليزي", message, re.I)
                 else detect_language(message)}
    courses = _courses()
    if m := re.search(r"\b([A-Z]{2,4}-\d{3})\b", message, re.I):
        if m.group(1).upper() in courses:
            out["course_id"] = m.group(1).upper()
    if m := re.search(r"\bCUR-[0-9A-F]{8}\b", message, re.I):
        out["curriculum_id"] = m.group(0).upper()
        row = db.get_curriculum(out["curriculum_id"])
        if row and row.get("course_id") and "course_id" not in out:
            out["course_id"] = row["course_id"].upper()
    if "course_id" not in out:
        words = _words(message)
        best = max(((len(words & _words(f"{c.title_en} {c.title_ar}")), c.course_id) for c in courses.values()),
                   default=(0, None))
        if best[0] >= 2:
            out["course_id"] = best[1]
    channels = [ch.value for ch, hints in CHANNEL_HINTS.items() if contains_any(message, hints)]
    if channels:
        out["channels"] = channels
    for objective, hints in OBJECTIVE_HINTS.items():
        if contains_any(message, hints):
            out["objective"] = objective
            break
    if m := re.search(r"budget\s*(?:of|:)?\s*([\d,]+)|([\d,]+)\s*(?:egp|sar|جنيه|ريال)|ميزانية\s*([\d,]+)", message, re.I):
        out["budget"] = float(next(g for g in m.groups() if g).replace(",", ""))
    if m := re.search(r"\b(?:for|targeting|aimed at)\s+(?!the\s+course\b|[A-Z]{2,4}-\d{3}\b|CUR-)(.+?)"
                      r"(?=,|\.|;|\s+(?:with|budget|in\s+arabic|in\s+english|on\s+(?:linkedin|facebook|instagram|"
                      r"whatsapp))\b|$)", message, re.I):
        out["target_audience"] = m.group(1).strip()
    if m := re.search(r"(\d)\s*(?:variations?|variants?|versions?)", message, re.I):
        out["variations_per_channel"] = max(1, min(5, int(m.group(1))))
    return out
