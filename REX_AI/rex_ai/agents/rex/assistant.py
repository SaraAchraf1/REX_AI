"""REX Course Assistant: conversation logic (owner: Member 1).

Flow per message:
  1. Detect language (Arabic / English) and update the lead profile from the text.
  2. Decide the intent: greeting, catalogue, course question, offers, FAQ, discovery.
  3. During discovery, ask only for missing profile fields, then recommend.
  4. Every answer is built from approved data and returns its sources.
  5. If an LLM key is configured, the LLM rewrites the draft reply (never the facts).
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from rex_ai.agents.rex import llm
from rex_ai.agents.rex.knowledge_base import KnowledgeBase, get_kb, source
from rex_ai.agents.rex.profile_extractor import detect_topics, update_profile, wants_to_skip
from rex_ai.agents.rex.prompts import DISCOVERY_QUESTIONS, SLOT_ORDER, TEXT
from rex_ai.agents.rex.recommender import FORMAT_LABEL, recommend
from rex_ai.agents.rex.retriever import make_retriever
from rex_ai.agents.rex.text_utils import contains_any, detect_language, normalize
from rex_ai.core import db
from rex_ai.core.schemas import (ChatRequest, ChatResponse, ChatStage, Course, CourseRecommendation,
                                 Language, LeadProfile, SourceRef)

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "salam", "مرحبا", "اهلا", "أهلا", "السلام عليكم", "هاي", "صباح الخير", "مساء الخير"]
CATALOG_WORDS = ["what courses", "which courses do you", "list of courses", "all courses", "course list", "catalog", "catalogue",
                 "الدورات المتاحه", "الكورسات المتاحه", "ايه الكورسات", "ما الدورات", "كل الدورات", "قائمه الدورات"]
RECOMMEND_WORDS = ["recommend", "suggest", "which course", "best course", "right course", "what should i take",
                   "رشح", "اقترح", "انسب", "الكورس المناسب", "الدوره المناسبه", "انصحني"]

ASPECTS: dict[str, list[str]] = {
    "offers": ["offer", "discount", "promo", "deal", "sale", "عرض", "عروض", "خصم", "تخفيض"],
    "fees": ["price", "cost", "fee", "fees", "how much", "pay", "installment", "سعر", "تكلفه", "مصاريف", "رسوم", "بكام", "تقسيط"],
    "schedule": ["when does", "when is", "start", "schedule", "next cohort", "dates", "timing", "what days",
                 "موعد", "ميعاد", "امتي", "يبدا", "مواعيد", "متي"],
    "duration": ["duration", "how long", "how many hours", "how many weeks", "length", "مده", "كام ساعه", "كام اسبوع"],
    "instructor": ["instructor", "trainer", "teacher", "who teaches", "who is teaching", "مدرب", "محاضر", "المدرس", "مين بيدرس", "من يدرس"],
    "prerequisites": ["prerequisite", "requirement", "need to know", "admission", "eligible", "placement", "required",
                      "متطلبات", "شروط", "محتاج اعرف", "القبول", "اختبار تحديد"],
    "content": ["content", "syllabus", "module", "topics", "what will i learn", "learning outcomes", "outcomes", "curriculum",
                "محتوي", "هتعلم", "ماذا سأتعلم", "المنهج", "نواتج", "موضوعات", "هدرس ايه"],
    "career": ["career", "jobs", "job opportunities", "who is it for", "target audience", "وظائف", "فرص عمل", "لمن", "مناسبه لمين"],
    "certificate": ["certificate", "certification", "accredited", "accreditation", "شهاده", "معتمد", "اعتماد"],
    "format": ["online", "onsite", "on-site", "hybrid", "in person", "format", "اونلاين", "حضوري", "هجين"],
    "overview": ["tell me about", "more about", "details", "what is", "info about", "information about", "explain",
                 "احكيلي", "تفاصيل", "معلومات عن", "ايه هو", "ما هي", "ما هو", "اشرح"],
}

EXTRA_ALIASES: dict[str, list[str]] = {
    "AI-101": ["intro to ai", "introduction to ai", "ai basics", "مقدمه في الذكاء"],
    "AI-201": ["generative ai", "genai", "llm", "الذكاء الاصطناعي التوليدي"],
    "DS-101": ["power bi", "data analysis course", "تحليل البيانات"],
    "SW-101": ["full stack", "fullstack", "full-stack", "web development", "فول ستاك", "تطوير الويب"],
    "SW-201": ["software testing", "qa automation", "testing course", "اختبار البرمجيات"],
    "CL-101": ["aws", "cloud practitioner"],
    "CL-201": ["devops", "kubernetes", "docker", "ديف اوبس"],
    "CY-101": ["cybersecurity", "cyber security", "الامن السيبراني"],
    "BU-101": ["pmp", "project management course"],
    "BU-102": ["digital marketing", "التسويق الرقمي"],
    "BU-201": ["scrum", "scrum master", "سكرم"],
    "PD-101": ["tot", "train the trainer", "اعداد المدربين"],
}
ORDINALS = [(["first", "1st", "#1", "number 1", "option 1", "الاول", "الاولي", "رقم 1"], 0),
            (["second", "2nd", "#2", "number 2", "option 2", "الثاني", "الثانيه", "التانيه", "رقم 2"], 1),
            (["third", "3rd", "#3", "number 3", "option 3", "الثالث", "الثالثه", "التالته", "رقم 3"], 2)]


QUESTION_STARTS = ["what", "how", "when", "where", "who", "why", "which", "is", "are", "can", "do", "does",
                   "will", "should", "tell me", "ما", "ماذا", "كيف", "متى", "امتى", "أين", "فين", "من", "لماذا", "ليه",
                   "هل", "ايه", "إيه", "ازاي", "إزاي", "عايز اعرف"]


def _t(key: str, lang: str) -> str:
    return TEXT[key][lang]


def _fee(c: Course, lang: str) -> str:
    """Imported courses may have no fee yet."""
    if c.fee is None:
        return _t("fee_na", lang)
    return f"{c.fee:,.0f} جنيه" if lang == "ar" else f"{c.fee:,.0f} EGP"


def _excerpt(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("؟ "))
    return (cut[:end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0]) + " ..."


class RexAssistant:
    def __init__(self, kb: Optional[KnowledgeBase] = None, db_path: Optional[Path] = None):
        self.kb = kb or get_kb()
        self.db_path = db_path
        self.retriever = make_retriever(self.kb.chunks)
        self.aliases = self._build_aliases()

    # ------------------------------------------------------------------ setup
    def _build_aliases(self) -> list[tuple[str, str]]:
        tag_count: dict[str, int] = {}
        for c in self.kb.courses.values():
            for tag in c.tags:
                tag_count[tag] = tag_count.get(tag, 0) + 1
        aliases: list[tuple[str, str]] = []
        for cid, c in self.kb.courses.items():
            names = [cid, cid.replace("-", ""), cid.replace("-", " "), c.title_en, c.title_ar]
            names += [t for t in c.tags if tag_count[t] == 1 and len(t) >= 3]
            names += EXTRA_ALIASES.get(cid, [])
            aliases += [(normalize(n), cid) for n in names]
        return sorted(aliases, key=lambda a: len(a[0]), reverse=True)  # longest match wins

    # ---------------------------------------------------------------- helpers
    def find_course(self, text: str, last_recs: list[str]) -> Optional[str]:
        t = normalize(text)
        for alias, cid in self.aliases:
            if alias.isascii() and len(alias) <= 6:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", t):
                    return cid
            elif alias and alias in t:
                return cid
        for words, idx in ORDINALS:
            if last_recs and idx < len(last_recs) and contains_any(text, words):
                return last_recs[idx]
        return None

    @staticmethod
    def detect_aspect(text: str) -> Optional[str]:
        for aspect, words in ASPECTS.items():
            if contains_any(f" {text} ", words):
                return aspect
        return None

    @staticmethod
    def _is_question(text: str) -> bool:
        return "?" in text or "؟" in text or contains_any(text, ["هل", "is it", "does it", "can i", "do you"])

    @classmethod
    def _asks(cls, text: str) -> bool:
        """A question (as opposed to profile info such as "I'm a student")."""
        t = normalize(text)
        return cls._is_question(text) or any(t == w or t.startswith(w + " ") for w in map(normalize, QUESTION_STARTS))

    @staticmethod
    def _known(profile: LeadProfile, slot: str) -> bool:
        if slot == "interests":
            return bool(profile.interests) or bool(profile.career_goal and detect_topics(profile.career_goal))
        if slot == "background":
            return bool(profile.background or profile.education_level)
        return getattr(profile, slot) is not None

    # ------------------------------------------------------------- main entry
    def handle(self, req: ChatRequest) -> ChatResponse:
        stored = db.load_session(req.session_id, self.db_path)
        state = json.loads(stored["profile"]) if stored else {}
        state.setdefault("profile", {})
        state.setdefault("asked", [])
        state.setdefault("skipped", [])
        state.setdefault("last_recs", [])
        state.setdefault("last_course", None)
        stage = ChatStage(stored["stage"]) if stored else ChatStage.discovery

        lang = (req.language.value if req.language else
                detect_language(req.message) if any(ch.isalpha() for ch in req.message)
                else (stored["language"] if stored else "en"))
        profile = LeadProfile(**{**state["profile"], "language": lang})
        profile, filled = update_profile(profile, req.message)

        reply, intent, recs, sources, facts, stage, offer = self._route(req.message, lang, profile, filled, state, stage)

        history = db.recent_messages(req.session_id, 6, self.db_path)
        polished = llm.polish(reply, facts, lang, history + [{"role": "user", "content": req.message}])
        final = polished or reply

        state["profile"] = profile.model_dump(mode="json")
        db.save_session(req.session_id, lang, state, stage.value, self.db_path)
        db.add_message(req.session_id, "user", req.message, db_path=self.db_path)
        db.add_message(req.session_id, "assistant", final, [s.model_dump(mode="json") for s in sources], self.db_path)

        return ChatResponse(session_id=req.session_id, reply=final, language=Language(lang), stage=stage,
                            intent=intent, profile=profile, recommendations=recs, sources=sources,
                            offer_lead_capture=offer, llm_used=polished is not None)

    # ----------------------------------------------------------------- router
    def _route(self, msg: str, lang: str, profile: LeadProfile, filled: list[str], state: dict, stage: ChatStage):
        explicit_course = self.find_course(msg, state["last_recs"])
        aspect = self.detect_aspect(msg)
        if aspect == "format" and not (explicit_course and self._is_question(msg)):
            aspect = None  # "I prefer online" is profile info, not a question
        words = normalize(msg).split()

        # 1. Greeting on a fresh conversation
        if contains_any(msg, GREETINGS) and len(words) <= 4 and not filled:
            state["asked"] = ["interests"]
            text = f"{_t('welcome', lang)}\n\n{DISCOVERY_QUESTIONS['interests'][lang]}"
            return text, "greeting", [], [], {}, ChatStage.discovery, False

        # 2. Catalogue listing
        if contains_any(msg, CATALOG_WORDS):
            return (*self._catalog(lang), stage, False)

        # 3. General FAQ (refunds, recordings, laptops, accreditation...) when no course is named
        if (not explicit_course and not filled and aspect in (None, "overview", "certificate")
                and not (aspect == "certificate" and state["last_course"])):
            faq = self._faq(msg, lang)
            if faq:
                return (*faq, stage, False)
            # 3b. ...or from the academy's uploaded documents (policies, brochures, handbooks...)
            docs = self._documents(msg, lang) if self._asks(msg) else None
            if docs:
                return (*docs, stage, False)

        # 4. Asking for a topic we don't teach ("courses on cooking")
        if (not explicit_course and not detect_topics(msg)
                and re.search(r"(courses?|programs?|training|دورات|دوره|كورس|كورسات)\s+(on|in|about|for|في|عن)\b",
                              normalize(msg))):
            depts = ", ".join(sorted({c.department for c in self.kb.courses.values()}))
            return _t("not_in_catalog", lang).format(depts=depts), "topic_not_offered", [], \
                [source("SRC-ACAD-COURSES")], {}, stage, False

        # 4b. A question nothing in the data answers: say so instead of guessing or changing the subject
        no_course_context = not (state["last_course"] or state["last_recs"])
        if (self._asks(msg) and not explicit_course and not detect_topics(msg)
                and (aspect is None or (aspect == "overview" and no_course_context))
                and set(filled) <= {"career_goal"} and not contains_any(msg, RECOMMEND_WORDS)
                and not wants_to_skip(msg)):
            docs = self._documents(msg, lang)
            if docs:
                return (*docs, stage, False)
            return _t("not_found", lang), "unknown", [], [], {}, stage, False

        # 5. Question about a specific course (explicit, ordinal, or the one being discussed)
        in_qa = stage in (ChatStage.recommendation, ChatStage.qa)
        if (aspect and (explicit_course or not filled or in_qa)) or (explicit_course and (not filled or self._is_question(msg))):
            course_id = explicit_course or state["last_course"] or (state["last_recs"][0] if state["last_recs"] else None)
            if course_id:
                state["last_course"] = course_id
                text, srcs, facts = self._course_answer(course_id, aspect or "overview", lang)
                return text, f"course_question:{aspect or 'overview'}", [], srcs, facts, ChatStage.qa, bool(state["last_recs"])
            if aspect in ("fees", "offers"):
                return (*self._offers(lang), stage, False)
            return _t("which_course", lang), "clarify_course", [], [], {}, stage, False

        # 5. Discovery and recommendation
        force = contains_any(msg, RECOMMEND_WORDS)
        if wants_to_skip(msg) and state["asked"]:
            state["skipped"].append(state["asked"][-1])
            force = force or contains_any(msg, ["just recommend", "recommend now", "رشح لي"])

        if filled or force or wants_to_skip(msg) or stage == ChatStage.discovery:
            if filled and stage in (ChatStage.recommendation, ChatStage.qa) and state["last_recs"]:
                force = True  # new info after recommendations -> refresh them
            return self._discover_or_recommend(msg, lang, profile, filled, state, force)

        # 6. Fallback: closest course in the knowledge base, else say we don't know
        hits = self.retriever.search(msg, k=1, filters={"type": "course"},
                                     min_score=self.retriever.thresholds["course_min"])
        if hits:
            cid = hits[0].chunk.metadata["course_id"]
            state["last_course"] = cid
            text, srcs, facts = self._course_answer(cid, "overview", lang)
            return text, "retrieval_fallback", [], srcs, facts, ChatStage.qa, False
        docs = self._documents(msg, lang)
        if docs:
            return (*docs, stage, False)
        return _t("not_found", lang), "unknown", [], [], {}, stage, False

    # ------------------------------------------------------- discovery / recs
    def _discover_or_recommend(self, msg, lang, profile, filled, state, force):
        others = sum(self._known(profile, s) for s in SLOT_ORDER[1:])
        has_topic = self._known(profile, "interests")
        asked = len(state["asked"])
        ready = has_topic and (others >= 3 or asked >= 4 or force)
        if not has_topic and (force or asked >= 3):
            ready = True  # recommend on what we have rather than looping forever

        if ready:
            recs = recommend(self.kb, profile, lang)
            state["last_recs"] = [r.course_id for r in recs]
            state["last_course"] = recs[0].course_id if recs else None
            text, srcs, facts = self._format_recs(recs, profile, lang)
            return text, "recommendation", recs, srcs, facts, ChatStage.recommendation, True

        nxt = next((s for s in SLOT_ORDER if not self._known(profile, s)
                    and s not in state["skipped"] and s not in state["asked"][-1:]), None)
        if nxt is None:
            nxt = next((s for s in SLOT_ORDER if not self._known(profile, s) and s not in state["skipped"]), None)
        if nxt is None:
            recs = recommend(self.kb, profile, lang)
            state["last_recs"] = [r.course_id for r in recs]
            text, srcs, facts = self._format_recs(recs, profile, lang)
            return text, "recommendation", recs, srcs, facts, ChatStage.recommendation, True
        state["asked"].append(nxt)
        prefix = (_t("got_it", lang) + " ") if filled else ""
        return prefix + DISCOVERY_QUESTIONS[nxt][lang], "discovery", [], [], {}, ChatStage.discovery, False

    def _format_recs(self, recs: list[CourseRecommendation], profile: LeadProfile, lang: str):
        ar = lang == "ar"
        on_topic = any(r.reasons and r.score > 0.45 for r in recs)
        lines = [_t("rec_header", lang) if on_topic or not profile.interests else _t("rec_none_topic", lang), ""]
        facts: dict = {"recommendations": []}
        for i, r in enumerate(recs, 1):
            c = self.kb.courses[r.course_id]
            fmts = " / ".join(FORMAT_LABEL[lang][f.value] for f in c.formats)
            start = r.next_start.isoformat() if r.next_start else ("غير محدد" if ar else "TBA")
            if ar:
                lines.append(f"{i}. **{c.title_ar}** ({c.course_id}): {c.duration_hours} ساعة، {fmts}، "
                             f"الرسوم {_fee(c, lang)}، أقرب بداية {start}")
                lines.append(f"   لماذا: {'، '.join(r.reasons) or c.description_ar}")
                if c.prerequisites and c.prerequisites != ["None"]:
                    lines.append(f"   المتطلبات: {', '.join(c.prerequisites)}")
            else:
                lines.append(f"{i}. **{c.title_en}** ({c.course_id}): {c.duration_hours} hours, {fmts}, "
                             f"{_fee(c, lang)}, next start {start}")
                lines.append(f"   Why: {'; '.join(r.reasons) or c.description_en}")
                if c.prerequisites and c.prerequisites != ["None"]:
                    lines.append(f"   Prerequisites: {', '.join(c.prerequisites)}")
            facts["recommendations"].append({**c.model_dump(mode="json"), "score": r.score, "reasons": r.reasons,
                                             "next_start": str(r.next_start)})
        lines += ["", _t("rec_footer", lang)]
        srcs = [source("SRC-ACAD-COURSES", r.course_id) for r in recs]
        srcs += [source("SRC-ACAD-PRICING"), source("SRC-ACAD-SCHEDULES")]
        return "\n".join(lines), srcs, facts

    # -------------------------------------------------------- course answers
    def _course_answer(self, cid: str, aspect: str, lang: str) -> tuple[str, list[SourceRef], dict]:
        c: Course = self.kb.courses[cid]
        ar = lang == "ar"
        title = f"**{c.title_ar if ar else c.title_en}** ({cid})"
        srcs = [source("SRC-ACAD-COURSES", cid)]
        facts = {"course": c.model_dump(mode="json")}
        fmts = " / ".join(FORMAT_LABEL[lang][f.value] for f in c.formats)

        if aspect in ("fees", "offers"):
            srcs.append(source("SRC-ACAD-PRICING", cid))
            promos = self.kb.promotions_for(cid)
            best, promo = self.kb.best_price(cid)
            facts["promotions"] = [p.model_dump(mode="json") for p in promos]
            if c.fee is None:
                return (f"{title}: {_fee(c, lang)}. " + ("يُرجى التواصل مع فريق القبول." if ar else
                                                         "Please contact the admissions team.")), srcs, facts
            if ar:
                lines = [f"رسوم {title}: {c.fee:,.0f} جنيه." + (" متاح التقسيط حتى 3 أقساط." if c.installments else " تُدفع كاملة عند التسجيل.")]
                for p in promos:
                    lines.append(f"- {p.name_ar}: خصم {p.discount_pct:.0f}% ({p.condition_ar})، ساري حتى {p.valid_until}")
                if promo:
                    lines.append(f"أفضل سعر مع {promo.name_ar}: {best:,.0f} جنيه (عرض واحد لكل تسجيل).")
            else:
                lines = [f"{title} costs {c.fee:,.0f} EGP." + (" Installments are available (up to 3)." if c.installments else " It is paid in full at registration.")]
                for p in promos:
                    lines.append(f"- {p.name_en}: {p.discount_pct:.0f}% off ({p.condition_en}), valid until {p.valid_until}")
                if promo:
                    lines.append(f"Best price with the {promo.name_en.lower()}: {best:,.0f} EGP (one offer per registration).")
            return "\n".join(lines), srcs, facts

        if aspect in ("schedule", "format"):
            srcs.append(source("SRC-ACAD-SCHEDULES", cid))
            runs = self.kb.schedules_for(cid)
            facts["schedules"] = [s.model_dump(mode="json") for s in runs]
            head = (f"{title} متاحة بنظام: {fmts}." if ar else f"{title} runs {fmts}.")
            if not runs:
                return f"{head}\n{_t('no_schedule', lang)}", srcs, facts
            rows = []
            for s in runs:
                where = f"، {s.location}" if (ar and s.location) else (f", {s.location}" if s.location else "")
                label = FORMAT_LABEL[lang][s.format.value]
                rows.append(f"- {s.start_date}: {label}{where}، {s.days} {s.time}، المقاعد المتبقية {s.seats_left}" if ar else
                            f"- {s.start_date}: {label}{where}, {s.days} {s.time}, {s.seats_left} seats left")
            intro = "المواعيد القادمة:" if ar else "Upcoming cohorts:"
            return "\n".join([head, intro] + rows), srcs, facts

        if aspect == "duration":
            weekly = c.duration_hours / max(c.duration_weeks, 1)
            return ((f"{title} مدتها {c.duration_hours} ساعة على {c.duration_weeks} أسابيع (حوالي {weekly:.0f} ساعات أسبوعيًا)، بنظام {fmts}.")
                    if ar else
                    (f"{title} is {c.duration_hours} hours over {c.duration_weeks} weeks (about {weekly:.0f} hours a week), delivered {fmts}.")), srcs, facts

        if aspect == "instructor":
            srcs.append(source("SRC-ACAD-INSTRUCTORS", cid))
            ins = self.kb.instructors_for(cid)
            facts["instructors"] = [i.model_dump() for i in ins]
            rows = [f"- {i.name}، {i.title}، خبرة {i.years_experience} سنة: {i.bio_ar}" if ar else
                    f"- {i.name}, {i.title}, {i.years_experience} years of experience: {i.bio_en}" for i in ins]
            return "\n".join([f"مدربو {title}:" if ar else f"Instructors for {title}:"] + rows), srcs, facts

        if aspect == "prerequisites":
            srcs.append(source("SRC-ACAD-ADMISSION"))
            lvl = self.kb.admission["by_level"][c.skill_level.value]
            gen = self.kb.admission["general"][lang]
            facts["admission"] = self.kb.admission
            if ar:
                return (f"متطلبات {title}: {', '.join(c.prerequisites)}.\n{lvl['ar']}\n"
                        f"متطلبات التسجيل العامة: {'، '.join(gen)}."), srcs, facts
            return (f"Prerequisites for {title}: {', '.join(c.prerequisites)}.\n{lvl['en']}\n"
                    f"General admission requirements: {'; '.join(gen)}."), srcs, facts

        if aspect == "content":
            mods = "، ".join(c.modules) if ar else ", ".join(c.modules)
            outs = "\n".join(f"- {o}" for o in c.learning_outcomes)
            return ((f"وحدات {title}: {mods}.\nنواتج التعلم (بالإنجليزية):\n{outs}") if ar else
                    (f"{title} modules: {mods}.\nBy the end you will be able to:\n{outs}")), srcs, facts

        if aspect == "career":
            return ((f"{title} مناسبة لـ: {c.target_audience}.\nفرص العمل: {', '.join(c.career_opportunities)}.") if ar else
                    (f"{title} is designed for {c.target_audience.lower()}.\nCareer paths: {', '.join(c.career_opportunities)}.")), srcs, facts

        if aspect == "certificate":
            text = f"{title}: {c.certificate}."
            if c.accreditation_alignment:
                text += f"\n{c.accreditation_alignment}. {_t('accreditation_note', lang)}"
            return text, srcs, facts

        # overview
        best, promo = self.kb.best_price(cid)
        runs = self.kb.schedules_for(cid)
        start = runs[0].start_date.isoformat() if runs else ("غير محدد" if ar else "TBA")
        srcs += [source("SRC-ACAD-PRICING", cid), source("SRC-ACAD-SCHEDULES", cid)]
        if ar:
            return (f"{title}\n{c.description_ar}\nالمستوى: {c.skill_level.value} | المدة: {c.duration_hours} ساعة | "
                    f"النظام: {fmts} | الرسوم: {_fee(c, lang)} | أقرب بداية: {start}"), srcs, facts
        return (f"{title}\n{c.description_en}\nLevel: {c.skill_level.value} | Duration: {c.duration_hours} hours | "
                f"Format: {fmts} | Fee: {_fee(c, lang)} | Next start: {start}"), srcs, facts

    # ------------------------------------------------------- other answers
    def _faq(self, msg: str, lang: str, min_score: Optional[float] = None, margin: Optional[float] = None):
        """Answer from an FAQ only when one entry is a clear match (score and lead over the runner-up)."""
        th = self.retriever.thresholds
        min_score = th["faq_min"] if min_score is None else min_score
        margin = th["faq_margin"] if margin is None else margin
        hits = self.retriever.search(msg, k=2, filters={"type": "faq"})
        if not hits or hits[0].score < min_score:
            return None
        if len(hits) > 1 and hits[0].score - hits[1].score < margin:
            return None
        fid = hits[0].chunk.metadata["faq_id"]
        f = next(x for x in self.kb.faqs if x.faq_id == fid)
        text = f.a_ar if lang == "ar" else f.a_en
        return text, f"faq:{f.category}", [], [source("SRC-ACAD-FAQ", fid)], {"faq": f.model_dump()}

    def _documents(self, msg: str, lang: str):
        """Answer from the passages of uploaded documents that clearly match the question."""
        try:
            hits = self.retriever.search_docs(msg, k=3, min_score=self.retriever.thresholds["doc_min"])
        except Exception:  # vector store unavailable -> behave as before
            return None
        if not hits:
            return None
        from rex_ai.core.rag import RAGPipeline
        lines = [_t("docs_intro", lang)]
        lines += [f"> {_excerpt(h.text)}\n({h.citation})" for h in hits[:2]]
        return ("\n\n".join(lines), "documents", [], RAGPipeline.to_source_refs(hits),
                {"documents": RAGPipeline.build_context(hits)})

    def _catalog(self, lang: str):
        ar = lang == "ar"
        rows = [_t("catalog_header", lang)]
        by_dept: dict[str, list[Course]] = {}
        for c in self.kb.courses.values():
            by_dept.setdefault(c.department, []).append(c)
        for dept, cs in by_dept.items():
            rows.append(f"\n**{dept}**")
            rows += [f"- {c.course_id}: {c.title_ar if ar else c.title_en} ({c.skill_level.value}, {c.duration_hours}h)" for c in cs]
        rows.append("\n" + DISCOVERY_QUESTIONS["interests"][lang])
        return "\n".join(rows), "catalog", [], [source("SRC-ACAD-COURSES")], {"count": len(self.kb.courses)}

    def _offers(self, lang: str):
        ar = lang == "ar"
        promos = [p for p in self.kb.promotions if p.valid_until >= date.today()]
        rows = [_t("offers_header", lang)]
        for p in promos:
            scope = ("كل الدورات" if ar else "all courses") if p.applies_to == "all" else ", ".join(p.applies_to)
            rows.append(f"- {p.name_ar}: {p.discount_pct:.0f}% ({p.condition_ar}) - {scope}، حتى {p.valid_until}" if ar else
                        f"- {p.name_en}: {p.discount_pct:.0f}% off ({p.condition_en}), {scope}, until {p.valid_until}")
        return "\n".join(rows), "offers", [], [source("SRC-ACAD-PRICING")], {"promotions": [p.model_dump(mode="json") for p in promos]}