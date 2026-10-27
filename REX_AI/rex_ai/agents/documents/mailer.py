"""E-mail an approved document as a PDF attachment (brief, Agent 4: "optionally distribute approved documents by
email"). With REX_SMTP_HOST set the message is sent; without it the message is saved as an .eml file in
outputs/documents/outbox/ (open it in any mail client), so the workflow can be demonstrated without a mail server."""
from __future__ import annotations

import logging
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from rex_ai import config

log = logging.getLogger("rex.documents.mailer")


def send_document(to: list[str], subject: str, body: str, attachment: Path) -> dict:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.SMTP_FROM or "rex-ai@academy.local", ", ".join(to)
    msg.set_content(body)
    msg.add_attachment(Path(attachment).read_bytes(), maintype="application", subtype="pdf",
                       filename=Path(attachment).name)
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
    outbox = config.DOCUMENT_OUTPUT_DIR / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = outbox / f"{stamp}_{re.sub(r'[^A-Za-z0-9_-]+', '_', Path(attachment).stem)}.eml"
    path.write_bytes(bytes(msg))
    log.info("SMTP not configured - e-mail saved to %s", path)
    return {"sent": False, "channel": "outbox", "to": to, "file": str(path)}
