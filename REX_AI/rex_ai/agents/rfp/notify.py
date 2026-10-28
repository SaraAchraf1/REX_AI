"""Alerts for newly discovered relevant RFPs (owner: Member 2).

Business rule from the brief: Egypt opportunities are flagged for review IMMEDIATELY (one high-priority alert
each); Saudi opportunities join the monitored pipeline (one normal-priority digest per run).
Every alert is stored in rfp_alerts (dashboard + GET /rfp/alerts) and written to the log. It is also POSTed
to REX_RFP_WEBHOOK_URL (Slack / Teams / n8n) and e-mailed to REX_RFP_ALERT_EMAILS when those are configured.
"""
from __future__ import annotations

import logging
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage

from rex_ai import config
from rex_ai.core import db

log = logging.getLogger("rex.rfp.alerts")

COUNTRY_NAMES = {"EG": "Egypt", "SA": "Saudi Arabia"}


def _line(r: dict) -> str:
    deadline = r.get("deadline") or "deadline not stated"
    return (f"{r['title']} - {r['issuing_organization']} ({COUNTRY_NAMES[r['country']]}), {deadline}, "
            f"fit: {r['qualification']}. {r.get('submission_link') or ''}").strip()


def _send_webhook(text: str) -> bool:
    if not config.RFP_WEBHOOK_URL:
        return False
    try:
        import requests
        r = requests.post(config.RFP_WEBHOOK_URL, json={"text": text}, timeout=10)
        return r.status_code < 400
    except Exception as exc:
        log.warning("RFP webhook failed: %s", exc)
        return False


def _send_email(subject: str, text: str) -> bool:
    if not (config.SMTP_HOST and config.RFP_ALERT_EMAILS):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.SMTP_FROM, ", ".join(config.RFP_ALERT_EMAILS)
    msg.set_content(text)
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
            s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as exc:
        log.warning("RFP alert e-mail failed: %s", exc)
        return False


def _deliver(subject: str, text: str) -> list[str]:
    log.warning("%s\n%s", subject, text)       # WARNING so it stands out in the API console
    channels = ["dashboard", "log"]
    if _send_webhook(f"*{subject}*\n{text}"):
        channels.append("webhook")
    if _send_email(subject, text):
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
        channels = _deliver("[REX RFP] Egypt tender flagged for review", message)
        alerts.append({"alert_id": "ALR-" + uuid.uuid4().hex[:10], "rfp_id": r["rfp_id"], "country": "EG",
                       "priority": "high", "message": message, "channels": channels, "created_at": now})

    if saudi:                                           # one digest for the monitored Saudi pipeline
        text = "\n".join(f"- {_line(r)}" for r in saudi)
        channels = _deliver(f"[REX RFP] {len(saudi)} new Saudi tender(s) added to monitoring", text)
        for r in saudi:
            alerts.append({"alert_id": "ALR-" + uuid.uuid4().hex[:10], "rfp_id": r["rfp_id"], "country": "SA",
                           "priority": "normal", "message": _line(r), "channels": channels, "created_at": now})

    for a in alerts:
        db.save_rfp_alert(a)
    return alerts
