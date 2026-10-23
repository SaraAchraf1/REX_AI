"""Course recommendation scoring (owner: Member 1).

Transparent weighted scoring so every recommendation can be explained:
  topic fit 45% · level fit 20% · format 15% · budget 10% · weekly time 10%
Unknown profile fields score neutral (0.5) instead of penalising the course.
"""
from __future__ import annotations

from rex_ai.agents.rex.knowledge_base import KnowledgeBase
from rex_ai.agents.rex.profile_extractor import TOPIC_TO_TAGS, detect_topics
from rex_ai.core.schemas import Course, CourseRecommendation, LeadProfile, SkillLevel

WEIGHTS = {"topic": 0.45, "level": 0.20, "format": 0.15, "budget": 0.10, "time": 0.10}

FORMAT_LABEL = {
    "en": {"online_live": "live online", "onsite": "on-site", "hybrid": "hybrid"},
    "ar": {"online_live": "أونلاين مباشر", "onsite": "حضوري", "hybrid": "هجين"},
}
TOPIC_LABEL = {
    "en": {"ai": "AI", "data": "data analysis", "web": "web development", "testing": "software testing",
           "cloud": "cloud computing", "devops": "DevOps", "security": "cybersecurity",
           "project_management": "project management", "agile": "agile/Scrum", "marketing": "marketing",
           "training": "training and teaching"},
    "ar": {"ai": "الذكاء الاصطناعي", "data": "تحليل البيانات", "web": "تطوير الويب", "testing": "اختبار البرمجيات",
           "cloud": "الحوسبة السحابية", "devops": "ديف أوبس", "security": "الأمن السيبراني",
           "project_management": "إدارة المشاريع", "agile": "المنهجيات الرشيقة", "marketing": "التسويق",
           "training": "التدريب والتعليم"},
}
LEVEL_LABEL ={"en": {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced"},
               "ar": {"Beginner": "المبتدئين", "Intermediate": "المستوى المتوسط", "Advanced": "المستوى المتقدم"}}


def _topic_score(course: Course, profile: LeadProfile) -> tuple[float, list[str]]:
    topics = list(profile.interests)
    if profile.career_goal:
        topics += [t for t in detect_topics(profile.career_goal) if t not in topics]
    if not topics:
        return 0.5, []
    tags = {t.lower() for t in course.tags}
    best, matched = 0.0, []
    for i, t in enumerate(topics):
        overlap = len(TOPIC_TO_TAGS.get(t, set()) & tags)
        if overlap:
            matched.append(t)
            # More shared tags = the course is ABOUT this topic, not just touching it
            fit = min(1.0, overlap / 3) * (1.0 if i == 0 else 0.85)
            best = max(best, fit)
    if best and profile.career_goal:
        goal = profile.career_goal.lower()
        if any(goal in c.lower() or c.lower() in goal for c in course.career_opportunities):
            best = min(1.0, best + 0.25)
    return best, matched


def _level_score(course: Course, profile: LeadProfile) -> float:
    if profile.skill_level is None:
        return 0.6
    table = {
        SkillLevel.beginner: {"Beginner": 1.0, "Intermediate": 0.25, "Advanced": 0.0},
        SkillLevel.intermediate: {"Beginner": 0.6, "Intermediate": 1.0, "Advanced": 0.5},
        SkillLevel.advanced: {"Beginner": 0.2, "Intermediate": 0.7, "Advanced": 1.0},
    }
    return table[profile.skill_level][course.skill_level.value]


def score_course(kb: KnowledgeBase, course: Course, profile: LeadProfile, lang: str = "en") -> CourseRecommendation:
    reasons: list[str] = []
    ar = lang == "ar"

    topic, matched = _topic_score(course, profile)
    if matched:
        label = TOPIC_LABEL[lang].get(matched[0], matched[0])
        reasons.append(f"يغطي اهتمامك بـ{label}" if ar else f"Matches your interest in {label}")
    if profile.career_goal and topic > 0:
        careers = ", ".join(course.career_opportunities[:2])
        reasons.append(f"يؤهلك لوظائف مثل: {careers}" if ar else f"Leads to roles such as {careers}")

    level = _level_score(course, profile)
    lvl = LEVEL_LABEL[lang][course.skill_level.value]
    if profile.skill_level and level >= 0.6:
        reasons.append(f"مناسب لـ{lvl}" if ar else f"Pitched at {lvl} level, which fits your experience")
    elif profile.skill_level and level < 0.6:
        reasons.append(f"تنبيه: الدورة لـ{lvl}" if ar else f"Note: this is an {lvl}-level course")

    fmts = [f.value for f in course.formats]
    if profile.preferred_format:
        fmt = 1.0 if profile.preferred_format.value in fmts else 0.0
        label = FORMAT_LABEL[lang][profile.preferred_format.value]
        if fmt:
            reasons.append(f"متاح بنظام {label}" if ar else f"Available {label}")
    else:
        fmt = 0.5

    discounted, promo = kb.best_price(course.course_id)
    price_for_budget = discounted if discounted is not None else course.fee
    if profile.budget and course.fee is not None:
        if course.fee <= profile.budget:
            budget = 1.0
            reasons.append(f"ضمن ميزانيتك ({course.fee:,.0f} جنيه)" if ar else
                           f"Within your budget ({course.fee:,.0f} EGP)")
        elif price_for_budget is not None and price_for_budget <= profile.budget:
            budget = 0.7
            name = promo.name_ar if ar else promo.name_en
            reasons.append(f"ضمن ميزانيتك مع {name}" if ar else f"Within budget with the {name.lower()}")
        else:
            budget = 0.0
    else:
        budget = 0.5

    weekly_load = course.duration_hours / max(course.duration_weeks, 1)
    if profile.weekly_hours:
        time = 1.0 if weekly_load <= profile.weekly_hours else 0.3
        if time == 1.0:
            reasons.append(f"حوالي {weekly_load:.0f} ساعات أسبوعيًا تناسب وقتك" if ar else
                           f"About {weekly_load:.0f} hours a week fits your time")
    else:
        time = 0.5

    total = (WEIGHTS["topic"] * topic + WEIGHTS["level"] * level + WEIGHTS["format"] * fmt
             + WEIGHTS["budget"] * budget + WEIGHTS["time"] * time)
    next_runs = kb.schedules_for(course.course_id)
    return CourseRecommendation(
        course_id=course.course_id,
        title=course.title_ar if ar else course.title_en,
        score=round(total, 3),
        reasons=reasons,
        fee=course.fee,
        currency=course.currency,
        next_start=next_runs[0].start_date if next_runs else None,
        format_options=course.formats,
    )


def recommend(kb: KnowledgeBase, profile: LeadProfile, lang: str = "en", top_k: int = 3) -> list[CourseRecommendation]:
    scored = [score_course(kb, c, profile, lang) for c in kb.courses.values()]
    has_topics = bool(profile.interests or (profile.career_goal and detect_topics(profile.career_goal)))
    if has_topics:
        # Never recommend off-topic courses when the lead told us what they want
        on_topic = [r for r in scored if _topic_score(kb.courses[r.course_id], profile)[0] > 0]
        scored = on_topic or scored
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
