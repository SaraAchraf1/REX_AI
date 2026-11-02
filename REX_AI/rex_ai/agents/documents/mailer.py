"""E-mail an approved document as a PDF attachment (brief, Agent 4: "optionally distribute approved documents by
email"). The message is really sent over SMTP (rex_ai/core/mailer.py, REX_SMTP_* settings); if SMTP is not
configured or the server refuses it, MailError (a RuntimeError) says why and nothing is recorded as sent."""
from __future__ import annotations

from pathlib import Path

from rex_ai.core import mailer


def send_document(to: list[str], subject: str, body: str, attachment: Path) -> dict:
    return mailer.send(to, subject, body, [Path(attachment)])
