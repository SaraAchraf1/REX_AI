"""Alerts for newly discovered relevant RFPs (owner: Member 2).

Business rule from the brief: Egypt opportunities are flagged for review IMMEDIATELY (one high-priority alert
each); Saudi opportunities join the monitored pipeline (one normal-priority digest per run).
Every alert is stored in rfp_alerts (dashboard + GET /rfp/alerts) and written to the log. It is also POSTed
to REX_RFP_WEBHOOK_URL (Slack / Teams / n8n) and e-mailed to REX_RFP_ALERT_EMAILS when those are configured.
A tender a reviewer marks qualified is e-mailed to the address given with the decision (send_qualified).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from rex_ai import config
from rex_ai.agents.rfp import email_html
from rex_ai.core import db, mailer
from rex_ai.mcp import client as tools

log = logging.getLogger("rex.rfp.alerts")

COUNTRY_NAMES = {"EG": "Egypt", "SA": "Saudi Arabia"}


def _line(r: dict) -> str:
    deadline = r.get("deadline") or "deadline not stated"
    return (f"{r['title']} - {r['issuing_organization']} ({COUNTRY_NAMES[r['country']]}), {deadline}, "
            f"fit: {r['qualification']}. {r.get('submission_link') or ''}").strip()


def _send_webhook(text: str) -> bool:
    if not config.RFP_WEBHOOK_URL:
        return False
    try:                                                 # through the MCP tool server
        return tools.call("post_webhook", url=config.RFP_WEBHOOK_URL, payload={"text": text})["ok"]
    except tools.ToolCallError as exc:
        log.warning("RFP webhook failed: %s", exc)
        return False


def _record_sent(rfp_ids: list[str], to: list[str]) -> None:
    """Mark the tenders as e-mailed (columns emailed_at / emailed_to). Runs after the e-mail went out, and a
    database problem here never turns a sent e-mail into an error."""
    try:
        db.mark_rfp_emailed(rfp_ids, to)
    except Exception as exc:
        log.warning("E-mail sent, but it could not be recorded on %s: %s", ", ".join(rfp_ids), exc)


def _send_email(subject: str, text: str, html: str, rfp_ids: list[str]) -> bool:
    if not (mailer.configured() and config.RFP_ALERT_EMAILS):
        return False
    try:
        mailer.send(config.RFP_ALERT_EMAILS, subject, text, html=html)
        _record_sent(rfp_ids, config.RFP_ALERT_EMAILS)
        return True
    except mailer.MailError as exc:
        log.warning("RFP alert e-mail failed: %s", exc)
        return False


def _deliver(subject: str, text: str, html: str, rfp_ids: list[str]) -> list[str]:
    log.warning("%s\n%s", subject, text)       # WARNING so it stands out in the API console
    channels = ["dashboard", "log"]
    if _send_webhook(f"*{subject}*\n{text}"):
        channels.append("webhook")
    if _send_email(subject, text, html, rfp_ids):
        channels.append("email")
    return channels


def notify(new_records: list[dict]) -> list[dict]:
    """new_records: RFPRecord dicts (mode='json') discovered in this run. Returns the stored alerts."""
    alerts = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    egypt = [r for r in new_records if r["country"] == "EG"]
    saudi = [r for r in new_records if r["country"] == "SA"]

    for r in egypt:                                     # one immediate alert per Egyptian tender
        message = _line(r)
        channels = _deliver("[REX RFP] Egypt tender flagged for review", message, email_html.egypt_alert(r),
                            [r["rfp_id"]])
        alerts.append({"alert_id": "ALR-" + uuid.uuid4().hex[:10], "rfp_id": r["rfp_id"], "country": "EG",
                       "priority": "high", "message": message, "channels": channels, "created_at": now})

    if saudi:                                           # one digest for the monitored Saudi pipeline
        text = "\n".join(f"- {_line(r)}" for r in saudi)
        channels = _deliver(f"[REX RFP] {len(saudi)} new Saudi tender(s) added to monitoring", text,
                            email_html.saudi_digest(saudi), [r["rfp_id"] for r in saudi])
        for r in saudi:
            alerts.append({"alert_id": "ALR-" + uuid.uuid4().hex[:10], "rfp_id": r["rfp_id"], "country": "SA",
                           "priority": "normal", "message": _line(r), "channels": channels, "created_at": now})

    for a in alerts:
        db.save_rfp_alert(a)
    return alerts


def send_qualified(r: dict, to: str, reviewer: str, comment: str = "") -> dict:
    """E-mail a tender the reviewer just marked qualified (raises mailer.MailError if it is not sent)."""
    lines = [f"{reviewer} marked this tender as QUALIFIED - prepare a bid.", "",
             f"Title: {r['title']}", f"Issuing organisation: {r['issuing_organization']}",
             f"Country: {COUNTRY_NAMES[r['country']]}", f"Category: {r['training_category']}",
             f"Deadline: {r.get('deadline') or 'not stated'}", f"Published: {r.get('publication_date') or '-'}",
             f"Academy fit: {r['qualification']}"]
    if r.get("matched_course_ids"):
        lines.append(f"Matching courses: {', '.join(r['matched_course_ids'])}")
    if r.get("submission_link"):
        lines.append(f"Submission link: {r['submission_link']}")
    if comment:
        lines += ["", f"Reviewer comment: {comment}"]
    for label, key in (("Summary", "summary"), ("Eligibility", "eligibility_requirements"),
                       ("Academy fit notes", "qualification_notes")):
        if r.get(key):
            lines += ["", f"{label}:", r[key]]
    if r.get("available_documents"):
        lines += ["", "Documents: " + "; ".join(r["available_documents"])]
    lines += ["", f"Reference: {r['rfp_id']}", "-- Sent by REX AI (RFP Research Agent)"]
    result = mailer.send([to], f"[REX RFP] Qualified: {r['title'][:120]}", "\n".join(lines),
                         html=email_html.qualified(r, reviewer, comment))
    _record_sent([r["rfp_id"]], [to])                                 # mailer.send raised if it was not sent
    db.save_rfp_alert({"alert_id": "ALR-" + uuid.uuid4().hex[:10], "rfp_id": r["rfp_id"], "country": r["country"],
                       "priority": "qualified", "message": f"Qualified by {reviewer} - e-mailed to {to}: {_line(r)}",
                       "channels": ["email"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return result
