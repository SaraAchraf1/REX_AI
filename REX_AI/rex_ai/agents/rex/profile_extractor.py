"""Rule-based extraction of the lead's profile from free text (Arabic + English).

Deterministic on purpose: it works without an API key, it is unit-testable, and
the LLM (when enabled) only phrases replies; it never decides the facts.
"""
from __future__ import annotations

import re
from typing import Optional

from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.core.schemas import LeadProfile, SkillLevel, TrainingFormat

# topic -> keywords. Topic keys are matched against course tags/department by the recommender.
TOPICS: dict[str, list[str]] = {
    "ai": ["artificial intelligence", " ai ", "ai", "machine learning", "deep learning", "ml", "genai",
           "generative", "llm", "chatgpt", "ذكاء اصطناعي", "الذكاء الاصطناعي", "تعلم الاله", "تعلم الآلة"],
    "data": ["data", "analyst", "analysis", "analytics", "power bi", "sql", "excel", "dashboard",
             "بيانات", "تحليل", "محلل"],
    "web": ["web", "website", "frontend", "front-end", "backend", "back-end", "full stack", "full-stack",
            "javascript", "react", "programming", "web developer", "coding", "برمجه", "برمجة", "مبرمج", "مطور ويب", "ويب", "مواقع"],
    "testing": ["testing", "tester", "qa", "quality assurance", "automation testing", "istqb", "selenium",
                "اختبار البرمجيات", "اختبار", "جوده البرمجيات", "ضمان الجودة"],
    "cloud": ["cloud", "aws", "azure", "سحاب", "الحوسبه السحابيه"],
    "devops": ["devops", "docker", "kubernetes", "ci/cd", "ديف اوبس"],
    "security": ["cyber", "security", "soc", "hacking", "network security", "امن سيبراني", "الامن السيبراني", "أمن"],
    "project_management": ["project management", "project manager", "pmp", "pmo", "اداره المشاريع", "إدارة المشاريع", "مدير مشروع"],
    "agile": ["agile", "scrum", "سكرم", "اجايل"],
    "marketing": ["marketing", "social media", "seo", "ads", "advertising", "تسويق", "سوشيال", "اعلانات"],
    "training": ["trainer", "training of trainers", "tot", "teaching", "instructor", "مدرب", "تدريب المدربين", "إعداد المدربين", "تدريس"],
}

TOPIC_TO_TAGS: dict[str, set[str]] = {
    "ai": {"ai", "machine learning", "generative ai", "llm", "rag", "agents"},
    "data": {"data analysis", "power bi", "sql", "data science", "dashboard"},
    "web": {"web development", "javascript", "react", "node", "programming"},
    "testing": {"testing", "qa", "automation", "istqb"},
    "cloud": {"cloud", "aws"},
    "devops": {"devops", "docker", "kubernetes", "ci/cd"},
    "security": {"cybersecurity", "security", "soc"},
    "project_management": {"project management", "pmp", "pmi"},
    "agile": {"agile", "scrum", "scrum master"},
    "marketing": {"marketing", "digital marketing", "social media", "seo", "ads"},
    "training": {"training", "tot", "trainer", "teaching", "learning and development"},
}

LEVEL_WORDS = {
    SkillLevel.beginner: ["beginner", "no experience", "new to", "just starting", "zero", "basic", "fresh",
                          "مبتدئ", "مبتدا", "بدون خبره", "ليس لدي خبره", "معنديش خبره", "من الصفر", "جديد"],
    SkillLevel.intermediate: ["intermediate", "some experience", "a bit of experience", "familiar with",
                              "متوسط", "عندي خبره", "لدي بعض الخبره", "بعض الخبرة"],
    SkillLevel.advanced: ["advanced", "expert", "senior", "متقدم", "خبير"],
}

FORMAT_WORDS = {
    TrainingFormat.hybrid: ["hybrid", "both online and", "mix", "هجين", "مدمج"],
    TrainingFormat.online_live: ["online", "remote", "from home", "virtual", "اونلاين", "أونلاين", "عن بعد", "من البيت"],
    TrainingFormat.onsite: ["onsite", "on-site", "in person", "in-person", "offline", "classroom", "at the academy",
                            "حضوري", "حضورى", "في المقر", "اوفلاين", "أوفلاين"],
}

EDUCATION_WORDS = {
    "student": ["student", "undergraduate", "university", "طالب", "طالبه", "جامعه"],
    "graduate": ["graduate", "graduated", "fresh grad", "bachelor", "خريج", "خريجه", "بكالوريوس", "ليسانس"],
    "postgraduate": ["master", "msc", "mba", "phd", "ماجستير", "دكتوراه"],
}

BACKGROUND_WORDS = [
    "engineer", "engineering", "computer science", "accountant", "accounting", "business", "commerce",
    "teacher", "doctor", "pharmacist", "sales", "human resources", "information technology", "it support", "developer", "designer",
    "مهندس", "هندسه", "حاسبات", "علوم حاسب", "محاسب", "تجاره", "مدرس", "معلم", "صيدلي", "طبيب", "مبيعات", "موارد بشريه",
]

SKIP_WORDS = ["just recommend", "recommend now", "suggest now", "skip", "any", "no preference", "don't mind",
              "doesn't matter", "رشح لي", "اقترح", "لا يهم", "مش فارق", "اي حاجه", "تخطي"]


def detect_topics(text: str) -> list[str]:
    padded = f" {text} "
    return [topic for topic, words in TOPICS.items() if contains_any(padded, words)]


def _number(raw: str) -> float:
    raw = raw.replace(",", "")
    mult = 1000 if raw.lower().endswith("k") else 1
    return float(raw.rstrip("kK")) * mult


def extract_budget(text: str) -> Optional[float]:
    t = normalize(text)
    m = re.search(r"(\d[\d,]*\.?\d*k?)\s*(egp|le|pounds?|جنيه|ج\.م|الف|ألف)", t)
    if m:
        val = _number(m.group(1))
        if m.group(2) in ("الف", "ألف") and val < 1000:
            val *= 1000
        return val
    m = re.search(r"(budget|ميزانيه|ميزانيتي|maximum|max|up to|حتي|لحد)\D{0,15}(\d[\d,]*\.?\d*k?)", t)
    if m:
        return _number(m.group(2))
    return None


def extract_weekly_hours(text: str) -> Optional[float]:
    t = normalize(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|ساعات|ساعه)\s*(a|per|each|every|/|في|كل)?\s*(week|wk|الاسبوع|اسبوع|اسبوعيا)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|ساعات|ساعه)\s*(a|per|each|every|في|كل)?\s*(day|يوم|يوميا)", t)
    if m:
        return float(m.group(1)) * 5
    return None


def extract_experience_years(text: str) -> Optional[float]:
    t = normalize(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(\+)?\s*(years?|yrs?|سنه|سنوات|سنين)\s*(of)?\s*(experience|exp|خبره)?", t)
    if m and (m.group(5) or "experience" in t or "خبره" in t):
        return float(m.group(1))
    return None


def extract_career_goal(text: str) -> Optional[str]:
    patterns = [
        r"(?:become|be|work as|career (?:in|as)|job as|get a job (?:in|as)|role as)\s+(?:an?\s+)?([a-z][a-z /&+\-]{2,40})",
        r"(?:اصبح|أصبح|اكون|أكون|اشتغل|أعمل|اعمل)\s+([؀-ۿ ]{2,40})",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            return m.group(1).strip(" .,!؟?")
    return None


def update_profile(profile: LeadProfile, text: str) -> tuple[LeadProfile, list[str]]:
    """Return the updated profile and the list of fields this message filled."""
    filled: list[str] = []
    data = profile.model_dump()

    topics = detect_topics(text)
    new_topics = [t for t in topics if t not in data["interests"]]
    if new_topics:
        data["interests"] = data["interests"] + new_topics
        filled.append("interests")

    goal = extract_career_goal(text)
    if goal:
        data["career_goal"] = goal
        filled.append("career_goal")

    for level, words in LEVEL_WORDS.items():
        if contains_any(text, words):
            data["skill_level"] = level
            filled.append("skill_level")
            break

    years = extract_experience_years(text)
    if years is not None:
        data["experience_years"] = years
        filled.append("experience_years")
        if "skill_level" not in filled:
            data["skill_level"] = SkillLevel.beginner if years < 1 else (
                SkillLevel.intermediate if years < 5 else SkillLevel.advanced)
            filled.append("skill_level")

    for fmt, words in FORMAT_WORDS.items():
        if contains_any(text, words):
            data["preferred_format"] = fmt
            filled.append("preferred_format")
            break

    budget = extract_budget(text)
    if budget:
        data["budget"] = budget
        filled.append("budget")

    hours = extract_weekly_hours(text)
    if hours:
        data["weekly_hours"] = hours
        filled.append("weekly_hours")

    for level, words in EDUCATION_WORDS.items():
        if contains_any(text, words):
            data["education_level"] = level
            filled.append("education_level")
            break

    if contains_any(f" {text} ", BACKGROUND_WORDS):
        data["background"] = text.strip()[:200]
        filled.append("background")
    elif "education_level" in filled and not data.get("background"):
        data["background"] = text.strip()[:200]
        filled.append("background")

    return LeadProfile(**data), filled


def wants_to_skip(text: str) -> bool:
    return contains_any(text, SKIP_WORDS)
