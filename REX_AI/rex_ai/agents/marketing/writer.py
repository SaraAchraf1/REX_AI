"""Rule-based campaign copy (brief, Agent 5 items 38 and 40): social captions, Facebook / Instagram ads, LinkedIn
posts, e-mails, WhatsApp messages, headlines and calls to action, in English or Arabic, with several variations.

Every sentence is assembled from the collected facts (MarketingFacts) and the audience persona, so the rule-based
copy is correct by construction; the optional LLM only rewrites it (llm.py) and is checked again (checks.py).

Each variation takes a different ANGLE, so the reviewer can A/B test them:
    outcome   what the learner will be able to do          career   the roles the course leads to
    offer     the current promotion / fee and its deadline  expert   the instructor's experience
    question  the audience's main pain point as a hook
Variant 1 of every channel carries the campaign's key message (the first angle); variant 2 is the A/B alternative.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from rex_ai.agents.marketing import rules
from rex_ai.agents.marketing.facts import MarketingFacts
from rex_ai.core.schemas import ContentVariant

ACADEMY = {"en": "Andalusia Academy", "ar": "أكاديمية الأندلس"}
ACADEMY_TAG = {"en": "#AndalusiaAcademy", "ar": "#أكاديمية_الأندلس"}
CURRENCY = {"en": {}, "ar": {"EGP": "جنيه", "SAR": "ريال"}}
MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          "ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر",
                 "ديسمبر"]}
DAYS_AR = {"Sat": "السبت", "Sun": "الأحد", "Mon": "الاثنين", "Tue": "الثلاثاء", "Wed": "الأربعاء", "Thu": "الخميس",
           "Fri": "الجمعة"}
AD_BUTTONS_AR = {"Sign Up": "سجّل الآن", "Learn More": "اعرف المزيد", "Contact Us": "تواصل معنا",
                 "Book Now": "احجز الآن", "Apply Now": "قدّم الآن", "Get Offer": "احصل على العرض",
                 "Send WhatsApp Message": "أرسل رسالة واتساب"}
# Angle order per persona: what that audience responds to first
PERSONA_ANGLES = {
    "students": ["offer", "career", "outcome", "expert", "question"],
    "career_switchers": ["career", "question", "outcome", "offer", "expert"],
    "professionals": ["outcome", "expert", "career", "offer", "question"],
    "managers": ["outcome", "expert", "offer", "question", "career"],
    "corporate": ["outcome", "expert", "offer", "question", "career"],
    "entrepreneurs": ["outcome", "offer", "question", "expert", "career"],
}

T = {
    "en": {
        "formats": {"online_live": "live online", "onsite": "on-site", "hybrid": "hybrid (online + on-site)"},
        "hook": {
            "outcome": "In {hours} hours, learn to {outcome_l}.",
            "career": "Ready for a role as {career}? Build the skills for it with {title}.",
            "offer": "Save {pct}% on {title} with the {promo} - until {until}.",
            "offer_fee": "{title}: {fee}{installments}.",
            "offer_custom": "{title}: {offer}.",
            "expert": "Learn {title} from {name}, {ititle} with {years} years of experience.",
            "question": "{pain}? {title} is built for {persona}.",
        },
        "head": {
            "outcome": ["{outcome}", "Learn to {outcome_l}", "{title}"],
            "career": ["Become a {career}", "Start your career as {career}", "{title}"],
            "offer": ["Save {pct}% on {title}", "Save {pct}% - {promo}", "Save {pct}% this month", "{title}"],
            "offer_fee": ["{title}", "{title} - {hours} hours"],
            "offer_custom": ["{offer}", "{title}"],
            "expert": ["Learn from {a_years}-year expert", "Learn from {name}", "{title}"],
            "question": ["{pain}?", "{title}"],
        },
        "head_short": "{hours} hours of practical training",
        "ad_short": "{title}: {hours} hours of practical training.",
        "price_short": "Now {best} instead of {fee} (until {until}).",
        "installments": " - installments available",
        "starts": "Next cohort: {date}, {format}{when}{where}.",
        "when": " ({days} {time})",
        "where": " in {location}",
        "no_date": "Next cohort dates will be announced soon.",
        "seats": "Only {seats} seats left.",
        "price_promo": "Fee: {best} instead of {fee} with the {promo} ({condition}; until {until}).",
        "price": "Fee: {fee}{installments}.",
        "price_custom": "Offer: {offer}.",
        "no_price": "Ask us for the fee and current offers.",
        "you_will": "In this {hours}-hour {level} course you will:",
        "taught_by": "Taught by {name}, {ititle} ({years} years of experience).",
        "project": "You finish with a hands-on project: {capstone}.",
        "for_who": "Who it is for: {audience}.",
        "careers": "Roles it prepares you for: {careers}.",
        "hello": "Hello,",
        "learn": "What you will learn:",
        "details": "Course details:",
        "d_start": "Start: {date} - {format}{when}",
        "d_duration": "Duration: {hours} hours over {weeks} weeks",
        "d_fee": "Fee: {fee}",
        "d_offer": "Offer: {best} with the {promo} - {condition}, until {until}",
        "d_custom": "Offer: {offer}",
        "d_cert": "Certificate: {certificate}",
        "d_installments": "Installments available",
        "sign": "Best regards,\nThe {academy} team",
        "unsubscribe": "You receive this e-mail because you asked us about our courses. Reply UNSUBSCRIBE to stop.",
        "wa_hello": "Hello from {academy}!",
        "wa_reply": "Reply 1 and our advisor will help you {cta_l}.",
        "wa_stop": "Reply STOP to unsubscribe.",
    },
    "ar": {
        "formats": {"online_live": "أونلاين مباشر", "onsite": "حضورياً", "hybrid": "مدمج (أونلاين وحضوري)"},
        "hook": {
            "outcome": "خلال {hours} ساعة تدريب: {description}",
            "career": "هل تستعد للعمل كـ {career}؟ ابنِ المهارات المطلوبة مع دورة {title}.",
            "offer": "وفّر {pct}% على دورة {title} مع {promo} - حتى {until}.",
            "offer_fee": "دورة {title}: {fee}{installments}.",
            "offer_custom": "دورة {title}: {offer}.",
            "expert": "تعلّم {title} مع {name}، {ititle}، بخبرة {years} عاماً.",
            "question": "{pain}؟ دورة {title} مصممة لـ {persona}.",
        },
        "head": {
            "outcome": ["{title}", "{title} في {hours} ساعة"],
            "career": ["ابدأ طريقك كـ {career}", "{title}"],
            "offer": ["وفّر {pct}% على {title}", "وفّر {pct}% - {promo}", "خصم {pct}% لفترة محدودة", "{title}"],
            "offer_fee": ["{title}", "{title} - {hours} ساعة"],
            "offer_custom": ["{offer}", "{title}"],
            "expert": ["تعلّم من خبير بخبرة {years} عاماً", "تعلّم مع {name}", "{title}"],
            "question": ["{pain}؟", "{title}"],
        },
        "head_short": "{hours} ساعة تدريب عملي",
        "ad_short": "دورة {title}: {hours} ساعة تدريب عملي.",
        "price_short": "الآن {best} بدلاً من {fee} (حتى {until}).",
        "installments": " - مع إمكانية التقسيط",
        "starts": "الدفعة القادمة: {date}، {format}{when}{where}.",
        "when": " ({days} {time})",
        "where": " في {location}",
        "no_date": "سيتم الإعلان عن مواعيد الدفعة القادمة قريباً.",
        "seats": "متبقٍ {seats} مقاعد فقط.",
        "price_promo": "الرسوم: {best} بدلاً من {fee} مع {promo} ({condition}؛ حتى {until}).",
        "price": "الرسوم: {fee}{installments}.",
        "price_custom": "العرض: {offer}.",
        "no_price": "تواصل معنا لمعرفة الرسوم والعروض الحالية.",
        "you_will": "في هذه الدورة ({hours} ساعة، مستوى {level}) ستتعلم:",
        "taught_by": "يقدمها {name}، {ititle} (خبرة {years} عاماً).",
        "project": "تنهي الدورة بمشروع عملي: {capstone}.",
        "for_who": "لمن هذه الدورة: {audience}.",
        "careers": "الوظائف التي تؤهلك لها: {careers}.",
        "hello": "مرحباً،",
        "learn": "ماذا ستتعلم:",
        "details": "تفاصيل الدورة:",
        "d_start": "البداية: {date} - {format}{when}",
        "d_duration": "المدة: {hours} ساعة على مدى {weeks} أسابيع",
        "d_fee": "الرسوم: {fee}",
        "d_offer": "العرض: {best} مع {promo} - {condition}، حتى {until}",
        "d_custom": "العرض: {offer}",
        "d_cert": "الشهادة: {certificate}",
        "d_installments": "إمكانية التقسيط متاحة",
        "sign": "مع أطيب التحيات،\nفريق {academy}",
        "unsubscribe": "تصلك هذه الرسالة لأنك استفسرت عن دوراتنا. أرسل \"إلغاء\" لإيقاف الرسائل.",
        "wa_hello": "مرحباً من {academy}!",
        "wa_reply": "أرسل 1 وسيساعدك مستشارنا: {cta_l}.",
        "wa_stop": "أرسل STOP لإيقاف الرسائل.",
    },
}
LEVELS_AR = {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"}


# ---------------------------------------------------------------------------- formatting helpers
def num(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def money(x: Optional[float], currency: str, lang: str) -> str:
    return f"{num(x)} {CURRENCY[lang].get(currency, currency)}"


def fmt_date(iso: str, lang: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[lang][d.month - 1]} {d.year}"


def days(text: str, lang: str) -> str:
    if lang != "ar" or not text:
        return text
    return re.sub(r"\b(Sat|Sun|Mon|Tue|Wed|Thu|Fri)\w*", lambda m: DAYS_AR[m.group(1)], text).replace(",", "،")


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and s[:2] != s[:2].upper() else s


def _sentence(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".!?؟" else s + "."


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?؟])\s+", text.strip())
    return parts[0] if parts else text


def fit(text: str, limit: int) -> tuple[str, bool]:
    """Cut at a word boundary to the limit. Returns (text, was_cut)."""
    text = text.strip()
    if len(text) <= limit:
        return text, False
    cut = text[:limit - 1].rsplit(" ", 1)[0].rstrip(" ,.;:-–")
    return cut + "…", True


_NOT_ACRONYMS = {"web", "app", "ads", "big", "the", "and", "for", "new", "lab"}


def hashtag(tag: str) -> str:
    """'power bi' -> #PowerBI, 'digital marketing' -> #DigitalMarketing, Arabic words joined with _."""
    words = re.findall(r"\w+", tag)
    if not tag.isascii():
        return "#" + "_".join(words)
    return "#" + "".join(w.upper() if len(w) <= 3 and w.lower() not in _NOT_ACRONYMS else w[:1].upper() + w[1:]
                         for w in words)


class CopyWriter:
    def __init__(self, facts: MarketingFacts, audience: dict, objective: str, objective_cfg: dict,
                 offer: Optional[str] = None, cta: Optional[str] = None):
        self.f = facts
        self.lang = "ar" if facts.language == "ar" else "en"
        self.L = T[self.lang]
        self.aud = audience
        self.objective, self.ocfg = objective, objective_cfg
        self.offer = (offer or "").strip() or None
        self.cta = (cta or "").strip() or objective_cfg["cta"][self.lang]
        self.angles = self._angles()

    # ------------------------------------------------------------------ facts as text
    def _vals(self) -> dict:
        f, c, lang = self.f, self.f.course, self.lang
        promo, ins = f.promotion or {}, f.instructors[0] if f.instructors else {}
        outcome = f.outcomes[0].rstrip(".") if f.outcomes else first_sentence(f.description).rstrip(".")
        return {
            "title": f.title, "hours": num(f.hours), "weeks": f.weeks, "description": first_sentence(f.description),
            "outcome": outcome, "outcome_l": _lower_first(outcome),
            "career": c.career_opportunities[0] if c.career_opportunities else f.title,
            "careers": ", ".join(c.career_opportunities[:3]),
            "fee": money(c.fee, c.currency, lang) if c.fee is not None else "",
            "best": money(f.best_price, c.currency, lang) if f.best_price is not None else "",
            "installments": self.L["installments"] if c.installments else "",
            "pct": num(promo.get("discount_pct", 0)), "promo": promo.get("name", ""),
            "condition": _lower_first(promo.get("condition", "")) if lang == "en" else promo.get("condition", ""),
            "until": fmt_date(promo.get("deadline") or promo["valid_until"], lang) if promo.get("valid_until") else "",
            "name": ins.get("name", ""), "ititle": ins.get("title", ""), "years": ins.get("years_experience", ""),
            "a_years": ("an " if str(ins.get("years_experience", "")).startswith("8") or
                        ins.get("years_experience") in (11, 18) else "a ") + str(ins.get("years_experience", "")),
            "pain": (self.aud["pains"] or [""])[0], "persona": _lower_first(self.aud["persona"]) if lang == "en"
            else self.aud["persona"], "offer": self.offer or "", "level": LEVELS_AR.get(c.skill_level.value)
            if lang == "ar" else c.skill_level.value.lower(), "audience": c.target_audience,
            "certificate": c.certificate, "capstone": f.capstone or "", "academy": ACADEMY[lang],
            "cta": self.cta, "cta_l": _lower_first(self.cta) if lang == "en" else self.cta,
        }

    def _angles(self) -> list[str]:
        order = list(PERSONA_ANGLES.get(self.aud["persona_id"], PERSONA_ANGLES["professionals"]))
        if self.objective == "awareness":                    # awareness leads with value, not price
            order.sort(key=lambda a: a == "offer")
        if self.offer and "offer" in order:                  # an offer the requester set is the key message
            order.remove("offer")
            order.insert(0, "offer")
        f = self.f
        ok = {"outcome": True, "career": bool(f.course.career_opportunities), "question": bool(self.aud["pains"]),
              "expert": bool(f.instructors and f.instructors[0].get("years_experience")),
              "offer": bool(self.offer or f.promotion or f.course.fee is not None)}
        return [a for a in order if ok[a]] or ["outcome"]

    def _offer_kind(self) -> str:
        return "offer_custom" if self.offer else "offer" if self.f.promotion else "offer_fee"

    def hook(self, angle: str) -> str:
        key = self._offer_kind() if angle == "offer" else angle
        return self.L["hook"][key].format(**self._vals())

    def headline(self, angle: str, limit: int) -> tuple[str, bool]:
        key = self._offer_kind() if angle == "offer" else angle
        options = [h.format(**self._vals()) for h in self.L["head"][key]] + [self.L["head_short"].format(**self._vals())]
        for h in options:
            if len(h) <= limit:
                return h, False
        return fit(options[-1], limit)

    def starts_line(self) -> str:
        s = self.f.next_start
        if not s:
            return self.L["no_date"]
        when = self.L["when"].format(days=days(s["days"], self.lang), time=s["time"]) if s.get("days") else ""
        where = self.L["where"].format(location=s["location"]) if s.get("location") else ""
        return self.L["starts"].format(date=fmt_date(s["start_date"], self.lang),
                                       format=self.L["formats"].get(s["format"], s["format"]), when=when, where=where)

    def seats_line(self) -> Optional[str]:
        s = self.f.next_start
        return self.L["seats"].format(seats=s["seats_left"]) if s and s.get("seats_left") and s["seats_left"] <= 10 \
            else None

    def price_line(self) -> str:
        v = self._vals()
        if self.offer:
            return self.L["price_custom"].format(**v)
        if self.f.promotion and self.f.best_price is not None:
            return self.L["price_promo"].format(**v)
        if self.f.course.fee is not None:
            return self.L["price"].format(**v)
        return self.L["no_price"]

    def value(self, angle: str) -> Optional[str]:
        """The course description sentence, unless the hook already says it."""
        desc = first_sentence(self.f.description)
        return None if desc.rstrip(".") in self.hook(angle) else _sentence(desc)

    def hashtags(self, n: int) -> list[str]:
        if n <= 0:
            return []
        tags = [hashtag(t) for t in self.f.course.tags if t.lower() not in ("business", "beginner")]
        out = list(dict.fromkeys(tags))[:max(n - 1, 0)] + [ACADEMY_TAG[self.lang]]
        return out[:n]

    def ad_button(self) -> str:
        allowed = rules.channel("facebook_instagram_ad").get("ctas") or []
        match = next((b for b in allowed if self.cta.lower() == b.lower()), None)
        button = match or self.ocfg.get("ad_cta") or "Learn More"
        return AD_BUTTONS_AR.get(button, button) if self.lang == "ar" else button

    # ------------------------------------------------------------------ channels
    def variant(self, channel: str, variant_no: int) -> ContentVariant:
        angle = self.angles[(variant_no - 1) % len(self.angles)]
        spec = rules.channel(channel)
        headline, cut = self.headline(angle, spec["headline_max"])
        body, cta, tags = getattr(self, f"_{channel}")(angle, spec)
        checks = [f"headline shortened to {spec['headline_max']} characters"] if cut else []
        return ContentVariant(channel=channel, variant_no=variant_no, headline=headline, body=body, cta=cta,
                              angle=angle, hashtags=tags, char_count=len(body), checks=checks)

    @staticmethod
    def _lines(*parts: Optional[str]) -> str:
        """None = left out, "" = blank line (never two in a row)."""
        text = "\n".join(p for p in parts if p is not None)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _social_caption(self, angle: str, spec: dict):
        tags = self.hashtags(spec["hashtags"])
        body = self._lines(self.hook(angle), "", self.value(angle), self.starts_line(),
                           self.price_line() if angle != "offer" or self.offer else None, self.seats_line(),
                           "", f"👉 {self.cta}", "", " ".join(tags))
        return body, self.cta, tags

    def _facebook_instagram_ad(self, angle: str, spec: dict):
        s = self.f.next_start
        short_start = (fmt_date(s["start_date"], self.lang) + " · " + self.L["formats"].get(s["format"], s["format"])) \
            if s else self.L["no_date"]
        hook = self.hook(angle)
        if len(hook) > spec["body_target"] * 1.6:           # the first 125 characters must carry the message
            hook = self.L["ad_short"].format(**self._vals())
        price = None
        if angle != "offer" and self.offer:
            price = self.L["price_custom"].format(**self._vals())
        elif angle != "offer" and self.f.promotion and self.f.best_price is not None:
            price = self.L["price_short"].format(**self._vals())
        body = self._lines(hook, short_start + ("" if s is None else "."), price)
        return body, self.ad_button(), []

    def _linkedin_post(self, angle: str, spec: dict):
        v = self._vals()
        tags = self.hashtags(spec["hashtags"])
        outcomes = [f"• {o.rstrip('.')}" for o in self.f.outcomes[:4]]
        ins = self.L["taught_by"].format(**v) if self.f.instructors and v["years"] else None
        careers = self.L["careers"].format(**v) if angle == "career" and v["careers"] else None
        body = self._lines(self.hook(angle), "", self.value(angle), "", self.L["you_will"].format(**v), *outcomes, "",
                           self.L["project"].format(**v) if self.f.capstone else None, ins, careers,
                           self.L["for_who"].format(**v), "", self.starts_line(), self.price_line(), self.seats_line(),
                           "", f"{self.cta} 👉", "", " ".join(tags))
        return body, self.cta, tags

    def _email(self, angle: str, spec: dict):
        v = self._vals()
        s = self.f.next_start
        details = []
        if s:
            when = self.L["when"].format(days=days(s["days"], self.lang), time=s["time"]) if s.get("days") else ""
            details.append(self.L["d_start"].format(date=fmt_date(s["start_date"], self.lang),
                                                    format=self.L["formats"].get(s["format"], s["format"]), when=when))
        else:
            details.append(self.L["no_date"])
        details.append(self.L["d_duration"].format(**v))
        if self.f.course.fee is not None:
            details.append(self.L["d_fee"].format(**v))
        if self.offer:
            details.append(self.L["d_custom"].format(**v))
        elif self.f.promotion and self.f.best_price is not None:
            details.append(self.L["d_offer"].format(**v))
        if self.f.course.installments:
            details.append(self.L["d_installments"])
        if self.f.course.certificate:
            details.append(self.L["d_cert"].format(**v))
        careers = self.L["careers"].format(**v) if angle == "career" and v["careers"] else None
        body = self._lines(self.L["hello"], "", self.hook(angle), "", self.value(angle), "", self.L["learn"], *[f"• {o.rstrip('.')}" for o in self.f.outcomes[:4]], careers, "",
                           self.L["details"], *[f"- {d}" for d in details], self.seats_line(), "",
                           f"→ {self.cta}", "", self.L["sign"].format(**v), "", self.L["unsubscribe"])
        return body, self.cta, []

    def _whatsapp(self, angle: str, spec: dict):
        v = self._vals()
        body = self._lines(self.L["wa_hello"].format(**v), self.hook(angle), self.starts_line(),
                           self.price_line() if angle != "offer" or self.offer else None, self.seats_line(), "",
                           self.L["wa_reply"].format(**v), self.L["wa_stop"])
        return body, self.cta, []

    # ------------------------------------------------------------------ campaign-level options
    def headline_options(self, n: int = 5) -> list[str]:
        out = []
        for a in self.angles:
            out.append(self.headline(a, 60)[0])
        return list(dict.fromkeys(out))[:n]

    def cta_options(self) -> list[str]:
        opts = [self.cta, self.ocfg["cta"][self.lang], self.ad_button(),
                "Ask us on WhatsApp" if self.lang == "en" else "راسلنا على واتساب"]
        return list(dict.fromkeys(opts))
