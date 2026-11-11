"""HTML versions of the RFP Research Agent's e-mails (owner: Member 2). The plain-text version is still sent with each
one for mail clients that do not show HTML.

E-mail clients ignore most CSS, so the layout is tables with inline styles, 600 px wide. Every value comes from a tender
- often scraped from a public portal - so all of it is escaped, and only http(s) links become buttons.
"""
from __future__ import annotations

from datetime import date
from html import escape
from typing import Optional

from rex_ai import config

COUNTRY_NAMES = {"EG": "Egypt", "SA": "Saudi Arabia"}
FONT = "font-family:Segoe UI,Helvetica,Arial,sans-serif"
INK, MUTED, LINE, BG = "#1f2933", "#616e7c", "#e4e7eb", "#f5f7fa"
ACCENT = {"qualified": "#0b7a4b", "high": "#b4441c", "normal": "#1d5fa8"}
FIT_COLORS = {"likely": ("#e3f6ec", "#0b7a4b"), "possible": ("#fff4dc", "#8a5a00"),
              "unlikely": ("#fde8e8", "#a61b1b"), "unknown": ("#eef1f4", "#52606d")}


def _e(value) -> str:
    return escape(str(value), quote=True)


def _today() -> date:
    try:
        return date.fromisoformat(config.RFP_TODAY) if config.RFP_TODAY else date.today()
    except ValueError:
        return date.today()


def _day(value) -> Optional[str]:
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d %b %Y")
    except ValueError:
        return None


def _deadline(value) -> str:
    """'12 Nov 2026 · 9 days left' (red when a week or less is left)."""
    if not value:
        return "Not stated"
    if not (shown := _day(value)):
        return _e(value)
    d = date.fromisoformat(str(value)[:10])
    left = (d - _today()).days
    note = "passed" if left < 0 else "today" if left == 0 else f"{left} day{'s' if left != 1 else ''} left"
    color = "#a61b1b" if left <= 7 else MUTED
    return f"{shown} <span style=\"color:{color}\">· {note}</span>"


def _link(url: Optional[str]) -> Optional[str]:
    return url if isinstance(url, str) and url.lower().startswith(("https://", "http://")) else None


def _fit(fit: str) -> str:
    bg, fg = FIT_COLORS.get(fit, FIT_COLORS["unknown"])
    return (f'<span style="background:{bg};color:{fg};border-radius:10px;padding:2px 10px;font-size:12px;'
            f'font-weight:600">{_e(fit).capitalize()} fit</span>')


def _button(url: str, label: str, color: str) -> str:
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin:16px 0 4px"><tr>'
            f'<td style="background:{color};border-radius:6px"><a href="{_e(url)}" style="display:inline-block;'
            f'padding:10px 18px;color:#ffffff;text-decoration:none;font-weight:600;{FONT};font-size:14px">'
            f'{_e(label)}</a></td></tr></table>')


def _facts(rows: list[tuple[str, str]]) -> str:
    cells = "".join(f'<tr><td style="padding:6px 12px 6px 0;color:{MUTED};font-size:13px;white-space:nowrap;'
                    f'vertical-align:top;width:150px">{_e(k)}</td><td dir="auto" style="padding:6px 0;color:{INK};font-size:14px">'
                    f'{v}</td></tr>' for k, v in rows)
    return f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%">{cells}</table>'


def _section(title: str, text: Optional[str]) -> str:
    if not text:
        return ""
    body = _e(text).replace("\n", "<br>")
    return (f'<p style="margin:18px 0 4px;color:{MUTED};font-size:12px;font-weight:700;letter-spacing:.04em;'
            f'text-transform:uppercase">{_e(title)}</p><p dir="auto" style="margin:0;color:{INK};font-size:14px;line-height:1.55">'
            f'{body}</p>')


def _tender(r: dict, color: str, full: bool) -> str:
    """One tender: title, key facts, submission button; full=True adds summary, eligibility and documents."""
    rows = [("Issuing organisation", _e(r.get("issuing_organization") or "-")),
            ("Country", _e(COUNTRY_NAMES.get(r.get("country"), r.get("country") or "-"))),
            ("Category", _e(r.get("training_category") or "-")),
            ("Deadline", _deadline(r.get("deadline"))),
            ("Academy fit", _fit(r.get("qualification") or "unknown"))]
    if full and r.get("publication_date"):
        rows.insert(4, ("Published", _day(r["publication_date"]) or _e(r["publication_date"])))
    if r.get("matched_course_ids"):
        rows.append(("Matching courses", _e(", ".join(r["matched_course_ids"]))))
    html = (f'<h2 dir="auto" style="margin:0 0 10px;color:{INK};font-size:18px;line-height:1.35">{_e(r.get("title") or "")}</h2>'
            + _facts(rows))
    if url := _link(r.get("submission_link")):
        html += _button(url, "Open the tender", color)
    if full:
        html += _section("Summary", r.get("summary"))
        html += _section("Eligibility", r.get("eligibility_requirements"))
        html += _section("Academy fit notes", r.get("qualification_notes"))
        if r.get("available_documents"):
            items = "".join(f'<li style="margin:2px 0">{_e(d)}</li>' for d in r["available_documents"])
            html += (f'<p style="margin:18px 0 4px;color:{MUTED};font-size:12px;font-weight:700;letter-spacing:.04em;'
                     f'text-transform:uppercase">Documents</p><ul style="margin:0;padding-left:20px;color:{INK};'
                     f'font-size:14px">{items}</ul>')
    elif r.get("summary"):
        html += f'<p dir="auto" style="margin:12px 0 0;color:{MUTED};font-size:13px;line-height:1.5">{_e(r["summary"])}</p>'
    return html


def _page(color: str, label: str, heading: str, intro: str, cards: list[str], footer: str) -> str:
    blocks = "".join(f'<tr><td style="padding:24px 28px;border-top:1px solid {LINE}">{c}</td></tr>' for c in cards)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:{BG};{FONT}">
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:{BG}"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:600px;background:#ffffff;border-radius:10px;border:1px solid {LINE};border-top:5px solid {color}">
<tr><td style="padding:24px 28px 18px">
<p style="margin:0 0 6px;color:{color};font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase">{_e(label)}</p>
<h1 style="margin:0 0 8px;color:{INK};font-size:22px;line-height:1.3">{_e(heading)}</h1>
<p style="margin:0;color:{MUTED};font-size:14px;line-height:1.5">{intro}</p>
</td></tr>
{blocks}
</table>
<p style="margin:16px 0 0;color:{MUTED};font-size:12px">{footer}</p>
</td></tr></table>
</body></html>"""


def qualified(r: dict, reviewer: str, comment: str = "") -> str:
    color = ACCENT["qualified"]
    card = _tender(r, color, full=True)
    if comment:
        card = (f'<div dir="auto" style="background:{BG};border-left:3px solid {color};padding:10px 14px;margin:0 0 18px;'
                f'color:{INK};font-size:14px"><strong>{_e(reviewer)}:</strong> {_e(comment)}</div>') + card
    return _page(color, "Tender qualified", "Prepare a bid for this tender",
                 f"<strong>{_e(reviewer)}</strong> marked this tender as qualified.", [card],
                 f"Reference {_e(r.get('rfp_id', ''))} · Sent by REX AI, RFP Research Agent")


def egypt_alert(r: dict) -> str:
    color = ACCENT["high"]
    return _page(color, "Egypt · high priority", "New tender flagged for review",
                 "A new relevant Egyptian tender was found. Please review it and decide whether to bid.",
                 [_tender(r, color, full=False)],
                 f"Reference {_e(r.get('rfp_id', ''))} · Sent by REX AI, RFP Research Agent")


def saudi_digest(records: list[dict]) -> str:
    color = ACCENT["normal"]
    n = len(records)
    return _page(color, "Saudi Arabia · monitoring", f"{n} new tender{'s' if n != 1 else ''} added to monitoring",
                 "These Saudi tenders were added to the monitored pipeline.",
                 [_tender(r, color, full=False) for r in records], "Sent by REX AI, RFP Research Agent")
