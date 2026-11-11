"""Real e-mail over SMTP for every agent (owner: Member 3; used by the Document and RFP agents).

Settings: REX_SMTP_HOST, REX_SMTP_PORT (587 = STARTTLS, 465 = SSL), REX_SMTP_USER, REX_SMTP_PASSWORD, REX_SMTP_FROM.
For Gmail: host smtp.gmail.com, port 587, the Gmail address as user and a 16-character App Password
(https://myaccount.google.com/apppasswords - the normal account password is refused).
Nothing is queued: if SMTP is not configured or the server refuses the message, MailError says why.

Agents call send(), which goes through the MCP tool server (tool send_email, rex_ai/mcp/server.py); the server
runs deliver(), the only code that talks to the mail server.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

from rex_ai import config

log = logging.getLogger("rex.mailer")


class MailError(RuntimeError):
    """The e-mail was not sent; the message says why."""


def configured() -> bool:
    return bool(config.SMTP_HOST)


def _login(s: smtplib.SMTP) -> None:
    if not config.SMTP_USER:
        return
    password = config.SMTP_PASSWORD
    if "gmail" in config.SMTP_HOST.lower():
        password = password.replace(" ", "")       # App Passwords are shown as "abcd efgh ijkl mnop"
    try:
        s.login(config.SMTP_USER.strip(), password)
    except smtplib.SMTPServerDisconnected as exc:
        # smtplib tries the next AUTH method after a refused login, but Gmail has already hung up - so the real
        # answer (535 Username and Password not accepted) surfaces as "Connection unexpectedly closed"
        raise smtplib.SMTPAuthenticationError(535, b"login refused (server closed the connection)") from exc


def send(to: list[str], subject: str, body: str, attachments: Iterable[Path] = (), html: str = "") -> dict:
    """Send through the MCP tool server; raises MailError with the reason if the e-mail is not sent.
    html = an HTML version; body is still sent as the plain-text alternative."""
    from rex_ai.mcp import client as tools
    files = [{"filename": Path(p).name, "content_base64": base64.b64encode(Path(p).read_bytes()).decode()}
             for p in attachments]
    try:
        return tools.call("send_email", to=list(to), subject=subject, body=body, attachments=files, html=html)
    except tools.ToolCallError as exc:
        raise MailError(str(exc)) from exc


def deliver(to: list[str], subject: str, body: str, files: Iterable[tuple[str, bytes]] = (), html: str = "") -> dict:
    """SMTP delivery (runs inside the MCP tool server)."""
    if not configured():
        raise MailError("E-mail not sent: SMTP is not configured - set REX_SMTP_HOST, REX_SMTP_PORT, REX_SMTP_USER "
                        "and REX_SMTP_PASSWORD in .env and restart the API")
    if not (config.SMTP_USER and config.SMTP_PASSWORD) and "gmail" in config.SMTP_HOST.lower():
        raise MailError("E-mail not sent: set REX_SMTP_USER (your Gmail address) and REX_SMTP_PASSWORD (a Gmail App "
                        "Password from https://myaccount.google.com/apppasswords) in .env and restart the API")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.SMTP_FROM or config.SMTP_USER, ", ".join(to)
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    for name, content in files:
        maintype, subtype = (mimetypes.guess_type(name)[0] or "application/octet-stream").split("/", 1)
        msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=name)
    try:
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
                _login(s)
                s.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
                s.starttls()
                _login(s)
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        hint = (" - Gmail needs a 16-character App Password (https://myaccount.google.com/apppasswords, 2-Step "
                "Verification must be on), not the normal Gmail password" if "gmail" in config.SMTP_HOST.lower() else "")
        raise MailError(f"E-mail not sent: the mail server refused the login (check REX_SMTP_USER / "
                        f"REX_SMTP_PASSWORD){hint}") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise MailError(f"E-mail not sent: the mail server refused the recipient(s) {', '.join(exc.recipients)}") \
            from exc
    except Exception as exc:
        raise MailError(f"E-mail not sent: {exc}") from exc
    log.info("E-mail '%s' sent to %s", subject, ", ".join(to))
    return {"sent": True, "channel": "smtp", "to": to}


def check_login() -> dict:
    """Connect and log in to the mail server without sending anything (orchestrator deep check)."""
    if not configured():
        raise MailError("SMTP is not configured - set REX_SMTP_HOST, REX_SMTP_PORT, REX_SMTP_USER and REX_SMTP_PASSWORD")
    try:
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
                _login(s)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
                s.starttls()
                _login(s)
    except smtplib.SMTPAuthenticationError as exc:
        hint = (" - Gmail needs a 16-character App Password (https://myaccount.google.com/apppasswords)"
                if "gmail" in config.SMTP_HOST.lower() else "")
        raise MailError(f"the mail server refused the login (check REX_SMTP_USER / REX_SMTP_PASSWORD){hint}") from exc
    except Exception as exc:
        raise MailError(f"the mail server could not be reached: {exc}") from exc
    return {"ok": True, "host": config.SMTP_HOST, "port": config.SMTP_PORT}
