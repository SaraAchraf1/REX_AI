"""Bilingual (English / Arabic) vocabularies the RFP agent uses to classify tenders.

Owner: Member 2. Every list is matched on normalized text (rex_ai.agents.rex.text_utils.normalize),
so Arabic letter variants and diacritics do not matter. Add terms here, not in the agent code.
"""
from __future__ import annotations

from rex_ai.core.schemas import Country, RFPCategory

# ------------------------------------------------------------------ geography
# Strict rule from the brief: only Egypt and Saudi Arabia. Anything else is rejected.
COUNTRY_ALIASES: dict[Country, list[str]] = {
    Country.EG: ["eg", "egy", "egypt", "arab republic of egypt", "مصر", "جمهورية مصر العربية"],
    Country.SA: ["sa", "ksa", "sau", "saudi arabia", "kingdom of saudi arabia", "saudi",
                 "السعودية", "المملكة العربية السعودية"],
}

# Used only when a notice has no country field: places that put it inside one of the two countries.
COUNTRY_PLACES: dict[Country, list[str]] = {
    Country.EG: ["egypt", "egyptian", "cairo", "giza", "alexandria", "new administrative capital", "port said",
                 "suez", "luxor", "aswan", "mansoura", "tanta", "assiut",
                 "مصر", "المصرية", "المصري", "القاهرة", "الجيزة", "الإسكندرية", "العاصمة الإدارية", "بورسعيد",
                 "السويس", "الأقصر", "أسوان", "المنصورة", "طنطا", "أسيوط"],
    Country.SA: ["saudi", "ksa", "riyadh", "jeddah", "dammam", "makkah", "mecca", "madinah", "medina", "khobar",
                 "neom", "tabuk", "abha",
                 "السعودية", "السعودي", "الرياض", "جدة", "الدمام", "مكة المكرمة", "المدينة المنورة", "الخبر",
                 "نيوم", "تبوك", "أبها"],
}

# Places outside scope. Only used to explain a rejection ("detected: UAE"), never to accept.
OTHER_PLACES = {
    "UAE": ["uae", "united arab emirates", "dubai", "abu dhabi", "sharjah", "الإمارات", "دبي", "أبوظبي"],
    "Jordan": ["jordan", "amman", "الأردن", "عمان الأردن"],
    "Kuwait": ["kuwait", "الكويت"],
    "Qatar": ["qatar", "doha", "قطر", "الدوحة"],
    "Oman": ["oman", "muscat", "سلطنة عمان", "مسقط"],
    "Bahrain": ["bahrain", "manama", "البحرين"],
    "Morocco": ["morocco", "rabat", "casablanca", "المغرب"],
    "Kenya": ["kenya", "nairobi"],
    "Europe": ["europe", "european union", "france", "paris", "germany", "berlin", "italy", "spain", "belgium",
               "brussels", "netherlands"],
    "USA": ["usa", "united states", "u.s.", "washington", "new york", "california", "texas"],
    "UK": ["united kingdom", "london", "england"],
}

# ------------------------------------------------------------------ relevance
# A tender must mention at least one of these to be about training at all.
TRAINING_CORE = [
    "training", "trainees", "course", "courses", "capacity building", "capacity development",
    "upskilling", "reskilling", "learning", "education", "educational", "bootcamp", "workshop", "curriculum",
    "academy", "e-learning", "skills development", "train the trainer", "train-the-trainer", "trainers", "tot",
    "upskill",
    "تدريب", "تدريبية", "تدريبي", "التدريب", "دورات", "دورة", "تعليم", "التعليم", "تعلم", "تأهيل",
    "بناء القدرات", "ورش عمل", "المتدربين", "المدربين",
]

# Phrases where "training" names a place, not the service being bought.
TRAINING_PLACES = ["training center", "training centre", "training centers", "training halls", "training hall",
                   "training facility", "training rooms", "مركز التدريب", "مركز تدريب", "قاعات التدريب",
                   "قاعة التدريب", "مراكز التدريب"]

# Supply / works / services tenders that only mention training in passing.
NOT_TRAINING = [
    "supply", "supplies", "installation", "construction", "civil works", "maintenance", "catering", "cleaning",
    "security guards", "furniture", "vehicles", "medical equipment", "air conditioning", "printing", "fuel",
    "توريد", "تركيب", "إنشاء", "انشاءات", "مقاولات", "صيانة", "إعاشة", "تغذية", "نظافة", "حراسة", "أثاث",
    "سيارات", "تكييف", "طباعة", "وقود",
]

CATEGORY_KEYWORDS: dict[RFPCategory, list[str]] = {
    RFPCategory.education_training: [
        "training program", "training programme", "educational", "education", "capacity building", "workshop",
        "courses", "برنامج تدريبي", "برامج تدريبية", "دورات تدريبية", "تعليم", "بناء القدرات", "ورش عمل"],
    RFPCategory.technical_training: [
        "technical training", "vocational", "tvet", "technical skills", "technicians", "hands-on", "lab",
        "تدريب فني", "تدريب تقني", "التدريب المهني", "التدريب التقني", "مهني", "الفنيين"],
    RFPCategory.professional_development: [
        "professional development", "leadership", "management training", "project management", "pmp", "agile",
        "scrum", "soft skills", "train the trainer", "train-the-trainer", "tot", "upskilling", "reskilling",
        "certification", "التطوير المهني", "القيادة", "القيادات", "إدارة المشروعات", "إدارة المشاريع",
        "تدريب المدربين", "المهارات الناعمة", "شهادات مهنية"],
    RFPCategory.digital_transformation: [
        "digital transformation", "digital skills", "digital literacy", "e-government", "digitization",
        "digital government", "التحول الرقمي", "المهارات الرقمية", "الحكومة الرقمية", "الرقمنة"],
    RFPCategory.software_it_training: [
        "software", "it training", "information technology", "programming", "coding", "cloud", "devops",
        "cybersecurity", "cyber security", "data analysis", "data analytics", "data science",
        "artificial intelligence", "machine learning", "ai", "software testing", "quality assurance",
        "web development", "power bi", "البرمجة", "البرمجيات", "الأمن السيبراني", "الحوسبة السحابية",
        "تحليل البيانات", "علوم البيانات", "الذكاء الاصطناعي", "تقنية المعلومات", "تكنولوجيا المعلومات",
        "ذكاء الأعمال", "اختبار البرمجيات"],
    RFPCategory.corporate_learning: [
        # specific phrases only: "employees"/"staff" alone appear in almost every tender and would outvote the topic
        "corporate training", "corporate learning", "employee training", "staff training", "e-learning",
        "lms", "learning management", "onboarding", "in-house training", "تدريب الموظفين", "تدريب العاملين",
        "التعلم الإلكتروني", "منصة تعليمية"],
}

# Category names a feed may already contain -> enum
CATEGORY_ALIASES: dict[str, RFPCategory] = {
    **{c.value.lower(): c for c in RFPCategory},
    **{c.name: c for c in RFPCategory},
    "it training": RFPCategory.software_it_training, "software training": RFPCategory.software_it_training,
    "corporate training": RFPCategory.corporate_learning, "vocational training": RFPCategory.technical_training,
    "تدريب": RFPCategory.education_training, "التحول الرقمي": RFPCategory.digital_transformation,
}

# Arabic topics -> the English tag used in the course catalogue (courses are tagged in English).
ARABIC_TOPICS = {
    "الأمن السيبراني": "cybersecurity", "أمن المعلومات": "security", "تحليل البيانات": "data analysis",
    "علوم البيانات": "data science", "ذكاء الأعمال": "power bi", "الذكاء الاصطناعي": "ai",
    "تعلم الآلة": "machine learning", "البرمجة": "programming", "الحوسبة السحابية": "cloud",
    "إدارة المشروعات": "project management", "إدارة المشاريع": "project management",
    "اختبار البرمجيات": "testing", "ضمان الجودة": "qa", "التسويق الرقمي": "digital marketing",
    "تدريب المدربين": "tot", "أجايل": "agile", "سكرم": "scrum", "تطوير الويب": "web development",
    "تطبيقات الجوال": "mobile development", "اختبار الاختراق": "penetration testing",
}

# Course tags too generic to prove the academy can deliver a tender on their own.
GENERIC_TAGS = {"beginner", "certification", "business", "management", "training", "software", "python"}

# ------------------------------------------------------------------ eligibility rules
SAUDI_REGISTRATION = [
    "saudi commercial registration", "commercial registration in the kingdom", "registered in saudi",
    "registered in the kingdom", "etimad", "saudi company", "saudi entity", "saudi-registered",
    "سجل تجاري سعودي", "سجل تجاري ساري في المملكة", "منصة اعتماد", "شركة سعودية", "منشأة سعودية",
    "مسجلة في المملكة"]
EGYPT_REGISTRATION = [
    "egyptian company", "registered in egypt", "egyptian commercial register", "tax card", "شركة مصرية",
    "سجل تجاري مصري", "بطاقة ضريبية", "مسجلة في مصر"]
TVTC_LICENCE = ["tvtc", "technical and vocational training corporation", "المؤسسة العامة للتدريب التقني",
                "ترخيص التدريب", "training licence", "training license"]
LOCAL_CONTENT = ["local content", "المحتوى المحلي"]
BID_BOND = ["bid bond", "bid security", "performance bond", "performance guarantee", "تأمين ابتدائي",
            "ضمان ابتدائي", "خطاب ضمان", "تأمين نهائي"]
ACCREDITED_CERTIFICATES = ["accredited certificate", "accredited training", "شهادات معتمدة", "تدريب معتمد"]
