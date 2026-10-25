"""Structured curriculum builder (brief, Agent 3 FR 20-21) - works without an LLM.

Every part of the curriculum comes from the research bundle, in this order of preference:
    modules      the requested catalogue course's modules > the best reference syllabus (RAG) > framework topics
                 > industry skills > a generic outline (flagged evidence = "none" for the reviewer)
    topics       the reference section's bullets > matching reference / skill topics > templated topics
    outcomes     the course owner's expected outcomes (verbatim) + one per module + transferable outcomes, so every
                 learning domain (knowledge, intellectual, practical, transferable) is present
    hours        a capstone share (~15%) + the rest split evenly over the modules; always sums to total_hours
    assessments  weights by skill level (always 100%), plus a mock exam for exam-preparation frameworks
When an LLM is configured, agent.py asks it to refine this draft and passes the result through finalize().

guard_claims() removes any statement that the course is accredited / certified by a body (brief rule).
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from rex_ai.agents.curriculum import alignment
from rex_ai.agents.curriculum.frameworks import named_in
from rex_ai.agents.curriculum.research import ResearchBundle, _words, prerequisite_course
from rex_ai.agents.rex.text_utils import normalize

CAPSTONE_SHARE = 0.15
HOURS_PER_MODULE = 6
HOURS_PER_WEEK = 8

T = {
    "en": {
        "capstone_module": "Capstone project",
        "capstone_topics": ["Project scoping and planning", "Building the solution with instructor mentoring",
                            "Presentation and peer review"],
        "topic_concepts": "Key concepts of {m}", "topic_tools": "Tools and techniques for {m}",
        "topic_apply": "Applying {m} in {ind} settings",
        "generic_modules": ["Foundations of {t}", "Core concepts and terminology", "Methods and tools",
                            "Applied practice", "Quality, ethics and professional practice"],
        "obj_foundation": "Build a solid foundation in {areas} for {aud}",
        "obj_skills": "Develop the practical skills employers in {ind} ask for, such as {skills}",
        "obj_roles": "Prepare learners for roles such as {roles}",
        "obj_framework": "Cover the topics of {fw} to support alignment with it (alignment, not accreditation)",
        "obj_capstone": "Apply everything learned in a capstone project based on a realistic {ind} case",
        "out_knowledge": "Explain the key concepts of {m}", "out_intellectual": "Analyse and compare approaches to {m}",
        "out_practical": "Apply {m} to a realistic {ind} task",
        "out_communicate": "Present project results clearly to technical and non-technical audiences",
        "out_ethics": "Work responsibly and ethically, respecting data privacy and professional standards",
        "pre_basic": "Basic computer literacy", "pre_course": "Completion of {c} ({cid}) or equivalent experience",
        "pre_dept": "Working knowledge of {d} fundamentals", "pre_experience": "Hands-on experience in {d}",
        "ex_lab": "Hands-on lab: apply {x} to a {ind} {artifact}", "ex_case": "Case study: apply {x} to a {ind} scenario",
        "ex_sec": "Guided lab: practise {x} in an isolated virtual environment", "ex_ref": "Exercise: {x}",
        "as_quiz": "Module quizzes (knowledge outcomes)", "as_assign": "Practical assignments and labs (skills outcomes)",
        "as_capstone": "Capstone project and presentation (all outcomes)",
        "as_exam": "Final written exam", "as_mock": "Mock exam on {fw} topics", "as_review": "Technical review / oral defence",
        "capstone": "{t} capstone: learners {d}. It brings together {mods} and is assessed with a rubric "
                    "(quality of the deliverable, correct application of course concepts, presentation).",
        "capstone_given": "{p}. Learners plan, build and present it; it brings together {mods} and is assessed "
                          "with a rubric (quality of the deliverable, correct application of course concepts, "
                          "presentation).",
        "industry_default": "workplace",
    },
    "ar": {
        "capstone_module": "المشروع التطبيقي الختامي",
        "capstone_topics": ["تحديد نطاق المشروع والتخطيط له", "تنفيذ الحل بإشراف المدرب", "العرض والتقييم بين الزملاء"],
        "topic_concepts": "المفاهيم الأساسية في {m}", "topic_tools": "أدوات وتقنيات {m}",
        "topic_apply": "تطبيق {m} في بيئة {ind}",
        "generic_modules": ["أساسيات {t}", "المفاهيم والمصطلحات الرئيسية", "المناهج والأدوات", "التطبيق العملي",
                            "الجودة والأخلاقيات والممارسة المهنية"],
        "obj_foundation": "بناء أساس متين في {areas} لدى {aud}",
        "obj_skills": "تنمية المهارات العملية التي يطلبها أصحاب العمل في {ind} مثل {skills}",
        "obj_roles": "إعداد المتدربين لوظائف مثل {roles}",
        "obj_framework": "تغطية موضوعات {fw} لدعم المواءمة معه (مواءمة وليس اعتماداً)",
        "obj_capstone": "تطبيق ما تم تعلمه في مشروع ختامي قائم على حالة واقعية في {ind}",
        "out_knowledge": "يشرح المفاهيم الأساسية في {m}", "out_intellectual": "يحلل ويقارن الأساليب المختلفة في {m}",
        "out_practical": "يطبق {m} على مهمة واقعية في {ind}",
        "out_communicate": "يعرض نتائج المشروع بوضوح للجمهور الفني وغير الفني",
        "out_ethics": "يعمل بمسؤولية وبشكل أخلاقي مع احترام خصوصية البيانات والمعايير المهنية",
        "pre_basic": "مهارات أساسية في استخدام الحاسب", "pre_course": "إتمام دورة {c} ({cid}) أو خبرة مماثلة",
        "pre_dept": "معرفة عملية بأساسيات {d}", "pre_experience": "خبرة عملية في {d}",
        "ex_lab": "معمل تطبيقي: تطبيق {x} على {artifact} من قطاع {ind}", "ex_case": "دراسة حالة: تطبيق {x} على موقف من قطاع {ind}",
        "ex_sec": "معمل موجه: التدرب على {x} في بيئة افتراضية معزولة", "ex_ref": "تمرين: {x}",
        "as_quiz": "اختبارات قصيرة لكل وحدة (نواتج المعرفة)", "as_assign": "تكليفات ومعامل عملية (نواتج المهارات)",
        "as_capstone": "المشروع الختامي والعرض (جميع النواتج)", "as_exam": "اختبار تحريري نهائي",
        "as_mock": "اختبار تجريبي على موضوعات {fw}", "as_review": "مراجعة فنية / مناقشة شفهية",
        "capstone": "المشروع الختامي لدورة {t}: {d}. يجمع المشروع بين {mods} ويُقيَّم بمعايير تقييم "
                    "(جودة المخرج، والتطبيق الصحيح لمفاهيم الدورة، والعرض).",
        "capstone_given": "{p}. يخطط المتدربون للمشروع وينفذونه ويعرضونه؛ ويجمع بين {mods} ويُقيَّم بمعايير تقييم "
                          "(جودة المخرج، والتطبيق الصحيح لمفاهيم الدورة، والعرض).",
        "industry_default": "العمل",
    },
}

ARTIFACT = {"Data": "dataset", "AI": "dataset", "Software": "application", "Cloud": "workload"}
ARTIFACT_AR = {"Data": "مجموعة بيانات", "AI": "مجموعة بيانات", "Software": "تطبيق", "Cloud": "بيئة سحابية"}
DELIVERABLE = {
    "en": {
        "Data": "analyse a real {ind} dataset end to end and deliver a dashboard with a short report of recommendations",
        "AI": "build a working AI application that solves a {ind} problem and evaluate its accuracy and risks",
        "Software": "build a working application with automated tests, a clean version-control history and a deployment",
        "Cloud": "deploy and monitor a cloud workload with infrastructure as code and a cost estimate",
        "Cybersecurity": "assess the security of a lab environment and deliver findings, risk ratings and a remediation plan",
        "Business": "prepare a complete plan for a {ind} case and present it to a panel",
        "Professional Development": "design and deliver a short training session and evaluate its effectiveness",
        "Healthcare": "run a quality-improvement project for a {ind} unit using the PDSA cycle",
        None: "complete an applied project on a realistic {ind} case and present the results",
    },
    "ar": {
        "Data": "تحليل مجموعة بيانات حقيقية من قطاع {ind} من البداية للنهاية وتقديم لوحة مؤشرات وتقرير توصيات",
        "AI": "بناء تطبيق ذكاء اصطناعي يحل مشكلة في قطاع {ind} وتقييم دقته ومخاطره",
        "Software": "بناء تطبيق يعمل مع اختبارات آلية وسجل منظم في نظام التحكم بالإصدارات ونشره",
        "Cloud": "نشر ومراقبة تطبيق سحابي باستخدام البنية كرمز مع تقدير للتكلفة",
        "Cybersecurity": "تقييم أمن بيئة معملية وتقديم النتائج وتصنيف المخاطر وخطة المعالجة",
        "Business": "إعداد خطة متكاملة لحالة من قطاع {ind} وعرضها أمام لجنة",
        "Professional Development": "تصميم جلسة تدريبية قصيرة وتقديمها وتقييم فعاليتها",
        "Healthcare": "تنفيذ مشروع لتحسين الجودة في وحدة من قطاع {ind} باستخدام دورة PDSA",
        None: "تنفيذ مشروع تطبيقي على حالة واقعية من قطاع {ind} وعرض النتائج",
    },
}
CASE_DEPARTMENTS = {"Business", "Professional Development", "Healthcare"}
INDUSTRY_AR = {"Technology": "التقنية", "Finance": "المالية", "Healthcare": "الرعاية الصحية", "Government": "الحكومي",
               "Retail": "التجزئة", "Telecom": "الاتصالات", "Banking": "البنوك", "E-commerce": "التجارة الإلكترونية",
               "Construction": "الإنشاءات", "Consulting": "الاستشارات", "Education": "التعليم",
               "Corporate training": "تدريب الشركات", "Energy": "الطاقة"}
DEPARTMENT_AR = {"AI": "الذكاء الاصطناعي", "Data": "البيانات", "Software": "البرمجيات", "Cloud": "الحوسبة السحابية",
                 "Cybersecurity": "الأمن السيبراني", "Business": "الأعمال", "Professional Development": "التطوير المهني",
                 "Healthcare": "الرعاية الصحية"}
LEVEL_AR = {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"}


# ============================================================================ hours and weights
def round_half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2


def split_hours(total: float, n: int) -> list[float]:
    """n shares rounded to 0.5 h that add up exactly to total."""
    if n <= 0:
        return []
    base = round_half(total / n)
    hours = [base] * n
    hours[-1] = round(total - base * (n - 1), 2)
    if hours[-1] <= 0:                                     # tiny totals: fall back to equal unrounded shares
        hours = [round(total / n, 2)] * (n - 1) + [round(total - round(total / n, 2) * (n - 1), 2)]
    return hours


def capstone_hours(total: float) -> float:
    if total < 8:
        return round_half(max(1.0, total * 0.2)) if total > 2 else 0.0
    return min(20.0, max(2.0, round_half(total * CAPSTONE_SHARE)))


def rescale_hours(modules: list[dict], total: float) -> None:
    """Make module hours add up to total, keeping their proportions (used after LLM output and revisions)."""
    current = sum(float(m.get("hours") or 0) for m in modules)
    if not modules:
        return
    if current <= 0:
        for m, h in zip(modules, split_hours(total, len(modules))):
            m["hours"] = h
        return
    scaled = [round_half(float(m.get("hours") or 0) * total / current) for m in modules]
    scaled[-1] = round(total - sum(scaled[:-1]), 2)
    if scaled[-1] <= 0:
        scaled = split_hours(total, len(modules))
    for m, h in zip(modules, scaled):
        m["hours"] = h


def normalize_weights(assessments: list[dict]) -> None:
    total = sum(float(a.get("weight_pct") or 0) for a in assessments)
    if not assessments or abs(total - 100) < 0.01:
        return
    if total <= 0:
        for a in assessments:
            a["weight_pct"] = round(100 / len(assessments), 1)
    else:
        for a in assessments:
            a["weight_pct"] = round(float(a.get("weight_pct") or 0) * 100 / total, 1)
    assessments[-1]["weight_pct"] = round(100 - sum(a["weight_pct"] for a in assessments[:-1]), 1)


# ============================================================================ accreditation claim guard
_NEGATION = re.compile(r"(\b(?:not|never|no|without|غير|ليس|لا)\b|n't\b)\W*(\w+\W+){0,3}$", re.I)
_CLAIMS = [
    (re.compile(r"\b(?:officially\s+|fully\s+)?(?:accredited|certified|endorsed)\s+by\b", re.I), "aligned with"),
    (re.compile(r"\bofficial(?:ly)?\s+accreditation\b", re.I), "alignment"),
    (re.compile(r"\baccreditation\s+(?:certificate|status)\b", re.I), "alignment report"),
    (re.compile(r"\b(?:officially\s+|fully\s+)?accredited\b", re.I), "aligned"),
    (re.compile(r"معتمد(?:ة)?\s+(?:رسمي(?:ا|اً)\s+)?من"), "متوافق مع"),
    (re.compile(r"اعتماد(?:اً)?\s+رسمي(?:اً)?"), "مواءمة"),
]


def guard_text(text: str) -> tuple[str, int]:
    """Rewrite accreditation claims as alignment statements. Negated mentions ('not accredited') are kept."""
    changes = 0
    for pattern, repl in _CLAIMS:
        def sub(m: re.Match) -> str:
            nonlocal changes
            if _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                return m.group(0)
            changes += 1
            return repl
        text = pattern.sub(sub, text)
    return text, changes


def guard_claims(obj: Any) -> tuple[Any, int]:
    """Apply guard_text to every string inside a curriculum dict / list."""
    if isinstance(obj, str):
        return guard_text(obj)
    if isinstance(obj, list):
        out, n = [], 0
        for x in obj:
            y, k = guard_claims(x)
            out.append(y)
            n += k
        return out, n
    if isinstance(obj, dict):
        out, n = {}, 0
        for key, x in obj.items():
            if key in ("sources", "metadata"):             # source titles are names, not claims
                out[key] = x
                continue
            y, k = guard_claims(x)
            out[key] = y
            n += k
        return out, n
    return obj, 0


# ============================================================================ builder
def _clean_title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" .:-")
    return s[:1].upper() + s[1:] if s and s[0].isascii() else s


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(min(len(wa), len(wb)), 1)


def _same(a: str, b: str) -> bool:
    return normalize(a) == normalize(b) or _overlap(a, b) >= 0.75


class CurriculumBuilder:
    def __init__(self, req: dict, bundle: ResearchBundle, kb):
        self.req, self.b, self.kb = req, bundle, kb
        self.lang = req.get("language", "en")
        self.t = T[self.lang]
        self.total = float(req["total_hours"])
        self.level = req["skill_level"]
        self.department = bundle.department
        self.industry = req.get("industry") or (bundle.industries[0] if bundle.industries else None)
        ind = INDUSTRY_AR.get(self.industry, self.industry) if self.lang == "ar" else self.industry
        self.ind = self._lower(ind) if ind else self.t["industry_default"]
        self.evidence: list[str] = []
        self.ref_outcomes: dict[str, str] = {}

    def _lower(self, s: str) -> str:
        """'Patient safety' -> 'patient safety' inside an English sentence; acronyms (SQL, AWS) and Arabic untouched."""
        if self.lang != "en" or not s or s[:2].isupper() or not s[0].isascii():
            return s
        return s[0].lower() + s[1:]

    # ---------------------------------------------------------------- modules
    def _course(self):
        cid = self.req.get("course_id")
        return self.kb.courses.get(cid) if cid else None

    def _reference_sections(self, n_max: int):
        """Sections of the best reference document in that document's order. When the title spans two fields
        ("Data Analysis for Healthcare Quality") and another reference document matches the title too, its most
        relevant sections take up to a third of the modules."""
        if not self.b.sections:
            return []
        by_doc: dict[str, list] = {}
        for s in self.b.sections:
            by_doc.setdefault(s.doc or s.citation.split(" — ")[0], []).append(s)
        title = self.req["course_title"]
        ranked = sorted(by_doc.items(), key=lambda kv: -(sum(s.score for s in kv[1]) + _overlap(title, kv[0])))
        primary = sorted(ranked[0][1], key=lambda s: s.order)
        secondary = next((ss for doc, ss in ranked[1:] if _overlap(title, doc) >= 0.5), None)
        if not secondary:
            return primary
        k = min(len(secondary), max(1, n_max // 3))
        extra = sorted(sorted(secondary, key=lambda s: -s.score)[:k], key=lambda s: s.order)
        return primary[: max(n_max - k, 1)] + extra

    def _skeleton(self) -> list[dict]:
        course = self._course()
        teaching = self.total - capstone_hours(self.total)
        n_max = max(3, min(10, round(teaching / HOURS_PER_MODULE)))
        sections = self._reference_sections(n_max)
        # chosen by the user, or named in the title: its topics must be taught. Merely suggested: gaps are reported.
        enforce = self.b.framework and (self.req.get("accreditation_framework")
                                        or named_in(self.req["course_title"], self.b.framework))
        fw_topics = [c for c in self.b.framework.competencies_for(self.department) if c.kind == "topic"] \
            if self.b.framework else []

        if course:
            self.evidence.append("catalogue")
            skeleton = [{"title": m, "topics": []} for m in course.modules]
        elif sections:
            self.evidence.append("references")
            skeleton = [{"title": _clean_title(s.title), "topics": list(s.topics)} for s in sections]
            self.ref_outcomes = {normalize(s.title): s.outcome for s in sections if s.outcome}
        elif fw_topics:
            self.evidence.append("framework")
            skeleton = [{"title": c.name, "topics": [_clean_title(k) for k in c.keywords[:4] if k.isascii()]}
                        for c in fw_topics]
        elif self.b.skills:
            self.evidence.append("industry_skills")
            skeleton = [{"title": s, "topics": []} for s in self.b.skills]
        else:
            skeleton = [{"title": g.format(t=self.req["course_title"]), "topics": []}
                        for g in self.t["generic_modules"]]

        # drop modules the catalogue course used for its own project; the capstone is added separately
        skeleton = [m for m in skeleton if not re.search(r"\b(mini|final)?\s*project\b|capstone", m["title"], re.I)] \
            or skeleton
        if enforce:                          # chosen / exam framework: every topic competency must appear
            for c in fw_topics:
                covering = next((m for m in skeleton
                                 if alignment.keyword_hit(c, f"{m['title']} {' '.join(m['topics'])}")), None)
                if covering:
                    covering["framework"] = c.id                 # protected from trimming below
                else:
                    skeleton.append({"title": c.name, "topics": [_clean_title(k) for k in c.keywords[:4]
                                                                  if k.isascii()], "framework": c.id})
            if "framework" not in self.evidence and fw_topics:
                self.evidence.append("framework")
        if len(skeleton) > n_max:                # keep framework-required modules, trim the rest from the end
            required = [m for m in skeleton if m.get("framework")]
            others = [m for m in skeleton if not m.get("framework")][: max(n_max - len(required), 1)]
            skeleton = [m for m in skeleton if m in others or m in required]
        return skeleton

    def _fill_topics(self, module: dict) -> list[str]:
        topics = [_clean_title(t) for t in module["topics"]][:5]
        if len(topics) >= 3:
            return topics
        pool = [t for s in self.b.sections for t in s.topics] + self.b.skills
        for t in sorted(pool, key=lambda t: -_overlap(module["title"], t)):
            if len(topics) >= 3 or _overlap(module["title"], t) < 0.34:
                break
            if not any(_same(t, x) for x in topics):
                topics.append(_clean_title(t))
        for key in ("topic_concepts", "topic_tools", "topic_apply"):
            if len(topics) >= 3:
                break
            topics.append(self.t[key].format(m=self._lower(module["title"]), ind=self.ind))
        return topics

    def _exercise(self, module: dict) -> str:
        ref = self.ref_outcomes.get(normalize(module["title"]))
        if ref and alignment.outcome_domain(ref) & {"practical", "intellectual"}:   # "Describe..." is no exercise
            return self.t["ex_ref"].format(x=ref.rstrip("."))
        x = self._lower(module["title"])
        if self.department == "Cybersecurity":
            return self.t["ex_sec"].format(x=x)
        if self.department in CASE_DEPARTMENTS or self.department not in ARTIFACT:
            return self.t["ex_case"].format(x=x, ind=self.ind)
        artifact = (ARTIFACT_AR if self.lang == "ar" else ARTIFACT)[self.department]
        return self.t["ex_lab"].format(x=x, ind=self.ind, artifact=artifact)

    def modules(self) -> list[dict]:
        skeleton = self._skeleton()
        for m in skeleton:
            m["topics"] = self._fill_topics(m)
        self._place_expected_outcomes(skeleton)
        cap = capstone_hours(self.total)
        hours = split_hours(self.total - cap, len(skeleton))
        out = [{"title": m["title"], "topics": m["topics"], "hours": h, "exercises": [self._exercise(m)]}
               for m, h in zip(skeleton, hours)]
        if cap:
            out.append({"title": self.t["capstone_module"], "topics": list(self.t["capstone_topics"]), "hours": cap,
                        "exercises": []})
        return out

    def _place_expected_outcomes(self, skeleton: list[dict]) -> None:
        """An outcome the course owner asked for must be taught somewhere: add it as a topic if nothing covers it."""
        for o in self.req.get("expected_outcomes", []):
            text = " ".join(f"{m['title']} {' '.join(m['topics'])}" for m in skeleton)
            if _overlap(o, text) >= 0.5:
                continue
            best = max(skeleton, key=lambda m: _overlap(o, f"{m['title']} {' '.join(m['topics'])}"))
            best["topics"].append(_clean_title(o))

    # ---------------------------------------------------------------- outcomes, objectives, prerequisites
    def outcomes(self, modules: list[dict]) -> list[str]:
        teaching = [m for m in modules if m["title"] != self.t["capstone_module"]]
        given = [_clean_title(o) for o in self.req.get("expected_outcomes", [])]
        out = list(given)
        for i, m in enumerate(teaching):
            ref = self.ref_outcomes.get(normalize(m["title"]))
            if ref:
                cand = ref.rstrip(".")
            elif len(given) >= 4:                # the owner's outcomes already describe the course
                continue
            else:
                key = "out_knowledge" if i == 0 else "out_intellectual" if i == len(teaching) // 2 else "out_practical"
                cand = self.t[key].format(m=self._lower(m["title"]), ind=self.ind)
            if not any(_same(cand, x) for x in out):
                out.append(cand)
        out = out[:max(len(given), 8)]
        out += [self.t["out_communicate"], self.t["out_ethics"]]
        domains = alignment.outcome_domains(out)
        first = self._lower(teaching[0]["title"] if teaching else self.req["course_title"])
        for domain, key in (("knowledge", "out_knowledge"), ("intellectual", "out_intellectual"),
                            ("practical", "out_practical")):
            if not domains.get(domain):
                out.insert(len(out) - 2, self.t[key].format(m=first, ind=self.ind))
        return out

    def objectives(self, modules: list[dict]) -> list[str]:
        t, comma = self.t, "، " if self.lang == "ar" else ", "
        teaching = [m["title"] for m in modules if m["title"] != t["capstone_module"]][:3]
        areas = comma.join(self._lower(x) for x in teaching)
        objs = [t["obj_foundation"].format(areas=areas, aud=self._lower(self.req["target_audience"]))]
        # industry_skills.yaml is English: in an Arabic curriculum these lines would mix languages
        if self.b.skills and self.lang == "en":
            skills = comma.join(self._lower(s) for s in self.b.skills[:3])
            objs.append(t["obj_skills"].format(ind=self.ind, skills=skills))
        if self.b.roles and self.lang == "en":
            objs.append(t["obj_roles"].format(roles=comma.join(self.b.roles[:3])))
        fw = self.b.framework
        if fw and any(c.kind == "topic" for c in fw.competencies):
            objs.append(t["obj_framework"].format(fw=fw.short_name))
        objs.append(t["obj_capstone"].format(ind=self.ind))
        return objs

    def prerequisites(self) -> list[str]:
        course = self._course()
        if course and course.prerequisites:
            return list(course.prerequisites)
        t, dept = self.t, self.department or self.req["course_title"]
        if self.lang == "ar":
            dept = DEPARTMENT_AR.get(dept, dept)
        if self.level == "Beginner":
            return [t["pre_basic"]]
        prev = prerequisite_course(self.kb, self.department, self.level, self.req["course_title"])
        pre = [t["pre_course"].format(c=prev.title_ar if self.lang == "ar" else prev.title_en, cid=prev.course_id)] \
            if prev else [t["pre_dept"].format(d=dept)]
        if self.level == "Advanced":
            pre.append(t["pre_experience"].format(d=dept))
        return pre

    # ---------------------------------------------------------------- assessments and capstone
    def assessments(self) -> list[dict]:
        t, fw = self.t, self.b.framework
        plan = {
            "Beginner": [("as_quiz", "quiz", 20), ("as_assign", "assignment", 35), ("as_capstone", "project", 45)],
            "Intermediate": [("as_quiz", "quiz", 15), ("as_assign", "assignment", 35),
                             ("as_capstone", "project", 40), ("as_exam", "exam", 10)],
            "Advanced": [("as_assign", "assignment", 35), ("as_capstone", "project", 50),
                         ("as_review", "exam", 15)],
        }[self.level]
        out = [{"name": t[k], "type": typ, "weight_pct": float(w)} for k, typ, w in plan]
        if fw and fw.exam_prep:              # the mock exam replaces the final exam; the capstone gives up the rest
            out = [a for a in out if a["type"] != "exam"]
            out.append({"name": t["as_mock"].format(fw=fw.short_name), "type": "exam", "weight_pct": 20.0})
            capstone = next(a for a in out if a["type"] == "project")
            capstone["weight_pct"] -= sum(a["weight_pct"] for a in out) - 100
        normalize_weights(out)
        return out

    def capstone(self, modules: list[dict]) -> str:
        t = self.t
        teaching = [m["title"] for m in modules if m["title"] != t["capstone_module"]]
        mods = ", ".join(teaching[:3]) + ("..." if len(teaching) > 3 else "")
        projects = self.req.get("required_projects") or []
        if projects:
            return t["capstone_given"].format(p=projects[0].rstrip("."), mods=mods)
        d = DELIVERABLE[self.lang].get(self.department, DELIVERABLE[self.lang][None]).format(ind=self.ind)
        return t["capstone"].format(t=self.req["course_title"], d=d, mods=mods)

    # ---------------------------------------------------------------- everything
    def build(self) -> dict:
        modules = self.modules()
        capstone = self.capstone(modules)
        if modules and modules[-1]["title"] == self.t["capstone_module"]:
            modules[-1]["exercises"] = [re.split(r"(?<=[.؛])\s", capstone)[0]]
        exercises = list(dict.fromkeys(e for m in modules[:-1] for e in m["exercises"]))
        exercises += [p for p in (self.req.get("required_projects") or [])[1:] if p not in exercises]
        return {
            "course_title": self.req["course_title"],
            "objectives": self.objectives(modules),
            "target_audience": self.req["target_audience"],
            "prerequisites": self.prerequisites(),
            "learning_outcomes": self.outcomes(modules),
            "modules": modules,
            "training_hours": self.total,
            "practical_exercises": exercises,
            "assessments": self.assessments(),
            "capstone_project": capstone,
            "evidence": self.evidence or ["none"],
        }


def finalize(cur: dict, req: dict, lang: str = "en") -> tuple[dict, int]:
    """Rules every curriculum must satisfy, whoever wrote it (rules, LLM or reviewer):
    hours add up to training_hours, weights to 100 %, requested outcomes and projects are present,
    and no accreditation claim survives. Returns (curriculum, claims_removed)."""
    t = T.get(lang, T["en"])
    total = float(cur.get("training_hours") or req.get("total_hours") or 0)
    modules = cur.get("modules") or []
    if total and modules:
        rescale_hours(modules, total)
    cur["training_hours"] = total or float(sum(m.get("hours", 0) for m in modules))
    normalize_weights(cur.get("assessments") or [])
    outcomes = cur.setdefault("learning_outcomes", [])
    for o in reversed(req.get("expected_outcomes") or []):
        if not any(_same(o, x) for x in outcomes):
            outcomes.insert(0, _clean_title(o))
    projects = req.get("required_projects") or []
    if projects and not _overlap(projects[0], cur.get("capstone_project") or "") >= 0.5:
        cur["capstone_project"] = t["capstone_given"].format(
            p=projects[0].rstrip("."), mods=", ".join(m["title"] for m in modules[:3]))
    exercises = cur.setdefault("practical_exercises", [])
    exercises += [p for p in projects[1:] if not any(_same(p, e) for e in exercises)]
    for m in modules:
        m.setdefault("exercises", [])
    cur, removed = guard_claims(cur)
    return cur, removed
