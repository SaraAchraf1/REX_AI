"""Campaign brief PDF for the administrator's review: strategy, audience, channel mix, budget, timeline, KPIs, A/B
tests, objection handling and every variant with its checks. Reuses the Document Agent's renderer (same look, Arabic
shaping, DRAFT watermark until the campaign is approved)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rex_ai.agents.documents.pdf import render_document_pdf
from rex_ai.agents.marketing import rules
from rex_ai.core.schemas import Campaign

_EMOJI = re.compile("[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F]")      # PDF fonts have no emoji glyphs


def _sec(key: str, heading: str, level: int = 1, **kw) -> dict:
    return {"key": key, "heading": heading, "level": level, **kw}


def _table(columns: list[str], rows: list[list], widths: Optional[list[float]] = None) -> dict:
    return {"columns": columns, "rows": [["" if c is None else str(c) for c in r] for r in rows], "widths": widths}


def brief_sections(c: Campaign) -> list[dict]:
    p = c.strategy_plan
    a = p.get("audience") or {}
    out = [_sec("strategy", "Campaign strategy", paragraphs=c.strategy.splitlines()),
           _sec("audience", "Target audience", paragraphs=[f"{a.get('persona_en')} - {a.get('text')}"],
                bullets=[f"Pain point: {x}" for x in a.get("pains") or []] +
                [f"Motivation: {x}" for x in a.get("motivations") or []] +
                [f"Targeting: {k.replace('_', ' ')} = {', '.join(v) if isinstance(v, list) else v}"
                 for k, v in (a.get("targeting") or {}).items()])]
    if c.headlines or c.ctas:
        out.append(_sec("headlines", "Headlines and calls to action",
                        bullets=[f"Headline: {h}" for h in c.headlines] + [f"CTA: {x}" for x in c.ctas]))
    if p.get("channel_mix"):
        out.append(_sec("mix", "Channel mix", table=_table(
            ["#", "Channel", "Role", "Paid", "Budget"],
            [[m["priority"], m["name"], m["role"], "yes" if m["paid"] else "no", m.get("budget")]
             for m in p["channel_mix"]], [0.4, 1.6, 3, 0.6, 0.9])))
    if p.get("timeline"):
        out.append(_sec("timeline", "Timeline", table=_table(
            ["Date", "Phase", "Channels", "Action"],
            [[t.get("date") or f"week {t.get('week')}", t["phase"], ", ".join(t["channels"]), t["action"]]
             for t in p["timeline"]], [1, 1, 1.6, 3])))
    if p.get("kpis"):
        out.append(_sec("kpis", "KPIs", bullets=p["kpis"]))
    if p.get("ab_tests"):
        out.append(_sec("ab", "A/B tests", table=_table(["Channel", "Test", "Metric"],
                                                        [[t["channel"], t["test"], t["metric"]] for t in p["ab_tests"]])))
    if p.get("objection_handling"):
        out.append(_sec("objections", "Objection handling (academy FAQs)", table=_table(
            ["Question", "Approved answer"], [[o["question"], o["answer"]] for o in p["objection_handling"]],
            [1, 2])))
    for ch in dict.fromkeys(v.channel.value for v in c.variants):
        out.append(_sec(f"channel:{ch}", rules.channel(ch)["name"][c.language.value]))
        for v in (x for x in c.variants if x.channel.value == ch):
            notes = [f"CTA: {v.cta}", f"Angle: {v.angle} · {v.char_count} characters · by {v.generated_by}"]
            notes += [f"Check: {x}" for x in v.checks]
            out.append(_sec(f"variant:{v.key}", f"{v.variant_no}. {v.headline or ''}".strip(), level=2,
                            paragraphs=[x for x in v.body.split("\n") if x.strip()], notes=notes))
    out.append(_sec("compliance", "Compliance", bullets=p.get("compliance") or []))
    return out


def render_campaign_pdf(c: Campaign, path: Path, reviews: Optional[list[dict]] = None) -> Path:
    doc = {"document_id": c.campaign_id, "doc_type": "campaign_brief", "title": c.title or c.campaign_id,
           "language": c.language.value, "status": c.status.value, "version": c.version,
           "metadata": {"type_title": "Campaign brief", "course_title": c.course_facts_used.get("title")},
           "sections": brief_sections(c), "sources": [s.model_dump(mode="json") for s in c.sources]}
    return render_document_pdf(doc, path, reviews=reviews)
