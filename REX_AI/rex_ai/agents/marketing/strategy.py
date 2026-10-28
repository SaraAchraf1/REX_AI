"""A simple campaign strategy (brief, Agent 5 item 39), built from the facts, the audience and the request:

  objective -> funnel stage and KPIs              audience persona -> channel priority, tone, targeting
  channels.yaml budget weights -> budget split     next cohort start / promotion end -> dated timeline
  variants per channel -> A/B tests                academy FAQs (RAG) -> objection handling for the sales team

The plan is written for the academy's administrator (English); the copy itself is in the campaign's language.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from rex_ai.agents.marketing import rules
from rex_ai.agents.marketing.facts import MarketingFacts
from rex_ai.agents.marketing.writer import CopyWriter, num

AB_METRIC = {"social_caption": "engagement rate", "facebook_instagram_ad": "click-through rate and cost per result",
             "linkedin_post": "engagement and profile visits", "email": "open and click rate",
             "whatsapp": "reply rate"}
PHASES = [  # (phase, days before the cohort starts, channels, action)
    ("Launch", 28, ["social_caption", "facebook_instagram_ad", "linkedin_post"],
     "Publish variant 1 everywhere; start the A/B test on ads"),
    ("Nurture", 14, ["email", "facebook_instagram_ad"],
     "E-mail consented leads; retarget page visitors with the winning ad"),
    ("Last call", 3, ["whatsapp", "email", "social_caption"],
     "Remind warm leads (WhatsApp opt-in only) with the start date and seats left"),
]


def _budget(budget: Optional[float], channels: list[str], currency: str) -> Optional[dict]:
    if not budget:
        return None
    weights = {ch: float(rules.channel(ch).get("budget_weight") or 0) for ch in channels}
    weights = {ch: w for ch, w in weights.items() if w > 0}
    if not weights:
        return {"total": budget, "currency": currency, "split": {},
                "note": "No paid channel selected (e-mail and WhatsApp cost no media budget): budget not allocated"}
    total_w = sum(weights.values())
    split = {ch: round(budget * w / total_w) for ch, w in weights.items()}
    last = list(split)[-1]
    split[last] = round(budget - sum(v for ch, v in split.items() if ch != last), 2)
    return {"total": budget, "currency": currency, "split": split,
            "note": "Split by channel weight (channels.yaml); move budget to the winning ad after the A/B test"}


def _timeline(facts: MarketingFacts, channels: list[str], today: date) -> list[dict]:
    start = date.fromisoformat(facts.next_start["start_date"]) if facts.next_start else None
    out = []
    for n, (phase, before, chans, action) in enumerate(PHASES):
        used = [c for c in chans if c in channels]
        if not used:
            continue
        if start:
            when = start - timedelta(days=before)
            if when < today:
                if n and out and out[-1]["date"] == today.isoformat():
                    continue                         # too late for this phase: the previous one covers it
                when = today
            out.append({"phase": phase, "date": when.isoformat(), "channels": used, "action": action})
        else:
            out.append({"phase": phase, "date": None, "week": n + 1, "channels": used, "action": action})
    promo = facts.promotion
    if promo and start:
        end = min(date.fromisoformat(promo.get("deadline") or promo["valid_until"]), start)
        if end >= today:
            out.append({"phase": "Offer ends" if end < start else "Offer deadline for this cohort",
                        "date": end.isoformat(), "channels": [],
                        "action": f"{promo['name']} ({num(promo['discount_pct'])}%) ends - mention the deadline"})
    if start:
        out.append({"phase": "Cohort starts", "date": start.isoformat(), "channels": [],
                    "action": "Stop conversion ads; thank-you message to registered learners"})
    return sorted(out, key=lambda x: (x["date"] or "", x.get("week", 0)))


def build(facts: MarketingFacts, audience: dict, writer: CopyWriter, objective: str, ocfg: dict,
          channels: list[str], variations: int, budget: Optional[float], tips: dict,
          today: Optional[date] = None) -> tuple[dict, str]:
    today = today or date.today()
    preferred = audience["preferred_channels"]
    ordered = sorted(channels, key=lambda c: (preferred.index(c) if c in preferred else len(preferred),
                                              channels.index(c)))
    budget_plan = _budget(budget, channels, facts.course.currency)
    mix = []
    for n, ch in enumerate(ordered, start=1):
        spec = rules.channel(ch)
        mix.append({"priority": n, "channel": ch, "name": spec["name"]["en"], "role": spec["role"]["en"],
                    "why": "preferred by this audience" if ch in preferred else "supporting channel",
                    "paid": bool(spec["paid"]), "variants": variations,
                    "budget": (budget_plan or {}).get("split", {}).get(ch),
                    "tips": tips.get(ch)})
    angles = writer.angles[:max(variations, 1)]
    ab = [{"channel": ch, "test": " vs ".join(f"variant {i + 1} ({a})" for i, a in enumerate(angles[:2])),
           "metric": AB_METRIC[ch]} for ch in ordered if variations >= 2 and len(angles) >= 2]
    kpis = list(ocfg["kpis"]) + [f"{rules.channel(ch)['name']['en']}: {AB_METRIC[ch]}" for ch in ordered]
    offer = ({"type": "requested", "text": writer.offer} if writer.offer else
             {"type": "promotion", **facts.promotion, "price_after": facts.best_price} if facts.promotion else
             {"type": "fee", "fee": facts.course.fee, "installments": facts.course.installments})
    compliance = [
        "Nothing is published or sent before an academy administrator approves the campaign",
        "Facts (fees, offers, dates, hours, instructors) come only from approved academy data - see sources",
        "E-mail and WhatsApp only to people who consented; every message has an opt-out line",
        "No accreditation, job or pass-rate guarantees",
    ]
    if facts.course.is_sample:
        compliance.append("Course data is SAMPLE data: replace with approved data before publishing")
    plan = {
        "objective": objective, "funnel": ocfg["funnel"],
        "audience": {k: audience[k] for k in ("persona", "persona_en", "secondary_persona", "text", "from", "pains",
                                               "motivations", "tone", "tone_en", "targeting", "lead_insights")},
        "key_messages": [writer.hook(a) for a in writer.angles[:3]],
        "channel_mix": mix, "budget": budget_plan, "timeline": _timeline(facts, ordered, today), "offer": offer,
        "cta": writer.cta, "kpis": kpis, "ab_tests": ab, "objection_handling": facts.objections,
        "compliance": compliance,
    }
    return plan, summary(plan, facts)


def summary(plan: dict, facts: MarketingFacts) -> str:
    c = facts.course
    a = plan["audience"]
    lines = [f"Objective: {plan['objective']} ({plan['funnel']} stage) for {c.title_en} ({c.course_id}).",
             f"Audience: {a['persona_en']} ({a['from']}: \"{a['text']}\"). Tone: {a['tone_en']}.",
             f"Key message: {plan['key_messages'][0]}" if plan["key_messages"] else ""]
    leads = (a.get("lead_insights") or {}).get("leads")
    if leads:
        lines.append(f"{leads} consented lead(s) already asked about this course - start with e-mail/WhatsApp to them.")
    lines.append("Channel mix: " + "; ".join(
        f"{m['priority']}. {m['name']} - {m['role']}" + (f" (budget {num(m['budget'])} {c.currency})"
                                                         if m.get("budget") else "") for m in plan["channel_mix"]))
    if plan["budget"] and not plan["budget"]["split"]:
        lines.append(plan["budget"]["note"] + ".")
    if plan["timeline"]:
        lines.append("Timeline: " + "; ".join(f"{t['date'] or 'week ' + str(t['week'])} {t['phase']}"
                                              for t in plan["timeline"]) + ".")
    o = plan["offer"]
    lines.append("Offer: " + (o["text"] if o["type"] == "requested" else
                              f"{o['name']} {num(o['discount_pct'])}% until {o.get('deadline') or o['valid_until']} "
                              f"({o['condition']})"
                              if o["type"] == "promotion" else
                              f"fee {num(o['fee'])} {c.currency}" if o.get("fee") is not None else "ask for the fee")
                 + f". CTA: {plan['cta']}.")
    lines.append("KPIs: " + ", ".join(plan["kpis"][:4]) + ".")
    if plan["ab_tests"]:
        lines.append("A/B test: " + plan["ab_tests"][0]["test"] + " on every channel.")
    lines.append("Approval: an administrator reviews every variant before anything is published.")
    return "\n".join(x for x in lines if x)
