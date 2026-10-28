"""Release an approved campaign for distribution (brief, Agent 5 item 42: "allow manual approval before sending or
publishing"; example workflow "... Admin Reviews -> Content Approved for Distribution").

The agent does not post to social networks itself: it writes a distribution package the marketing team (or an n8n /
Zapier flow) picks up, and e-mails the e-mail variant to consented recipients:

    outputs/campaigns/<campaign_id>/v<version>/
        campaign.json        chosen variants, strategy plan, facts used, sources, approval
        content.csv          one row per variant (UTF-8 with BOM: opens in Excel / Google Sheets)
        <channel>_<n>.txt    copy-paste ready text per variant
    outputs/campaigns/outbox/*.eml   the e-mail when REX_SMTP_HOST is not set
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from rex_ai import config
from rex_ai.core.schemas import Campaign, ContentVariant

log = logging.getLogger("rex.marketing.publish")


def _text(v: ContentVariant) -> str:
    parts = [f"[{v.channel.value} - variant {v.variant_no} - {v.angle or ''}]"]
    if v.headline:
        parts.append(f"Headline: {v.headline}")
    parts += ["", v.body, "", f"CTA: {v.cta}"]
    return "\n".join(parts) + "\n"


def write_package(camp: Campaign, variants: list[ContentVariant], publisher: str) -> dict:
    folder = config.MARKETING_OUTPUT_DIR / camp.campaign_id / f"v{camp.version}"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for v in variants:
        path = folder / f"{v.channel.value}_{v.variant_no}.txt"
        path.write_text(_text(v), encoding="utf-8")
        files.append(path.name)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["campaign_id", "course_id", "channel", "variant_no", "angle", "headline", "body", "cta", "hashtags"])
    for v in variants:
        w.writerow([camp.campaign_id, camp.course_id, v.channel.value, v.variant_no, v.angle, v.headline or "",
                    v.body, v.cta, " ".join(v.hashtags)])
    (folder / "content.csv").write_text("﻿" + buf.getvalue(), encoding="utf-8")
    payload = camp.model_dump(mode="json")
    payload["variants"] = [v.model_dump(mode="json") for v in variants]
    payload["published_by"] = publisher
    (folder / "campaign.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"folder": str(folder), "files": ["campaign.json", "content.csv", *files]}


def send_email(to: list[str], v: ContentVariant) -> dict:
    """The e-mail variant as a plain-text message: SMTP when configured, else an .eml file in the outbox."""
    msg = EmailMessage()
    msg["Subject"] = v.headline or v.cta
    msg["From"], msg["To"] = config.SMTP_FROM or "rex-ai@academy.local", ", ".join(to)
    msg.set_content(v.body)
    if config.SMTP_HOST:
        try:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
                s.starttls()
                if config.SMTP_USER:
                    s.login(config.SMTP_USER, config.SMTP_PASSWORD)
                s.send_message(msg)
        except Exception as exc:
            raise RuntimeError(f"E-mail could not be sent: {exc}") from exc
        return {"sent": True, "channel": "smtp", "to": to}
    outbox = config.MARKETING_OUTPUT_DIR / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = outbox / f"{stamp}_{re.sub(r'[^A-Za-z0-9_-]+', '_', v.key)}.eml"
    path.write_bytes(bytes(msg))
    log.info("SMTP not configured - campaign e-mail saved to %s", path)
    return {"sent": False, "channel": "outbox", "to": to, "file": str(Path(path))}
