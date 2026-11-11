"""Gemini client for change requests in the Curriculum, Document and Marketing agents (owner: Member 3).

Gemini is called through the MCP tool server (tool llm_chat, provider gemini - its OpenAI-compatible endpoint). Returns the
parsed JSON answer, or None when Gemini is not configured or the call fails (the reason is kept in llm_status).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from rex_ai import config
from rex_ai.core import llm_status

log = logging.getLogger("rex.gemini")


def enabled() -> bool:
    return config.USE_GEMINI


def chat_json(messages: list[dict], temperature: float = 0.2) -> Optional[dict]:
    llm_status.clear()
    if not config.USE_GEMINI:
        return None
    try:
        from rex_ai.mcp import client as tools
        text = tools.chat("gemini", messages, temperature=temperature, json_mode=True).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)        # Gemini sometimes fences its JSON
        return json.loads(text)
    except Exception as exc:
        log.warning("Gemini change request failed, nothing changed: %s", exc)
        llm_status.record(exc, provider="Gemini")
        return None
