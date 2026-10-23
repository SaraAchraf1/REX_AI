"""Optional LLM layer. If OPENAI_API_KEY is not set (or the call fails), REX
falls back to its deterministic replies, so the prototype always works."""
from __future__ import annotations

import json
import logging
from typing import Optional

from rex_ai import config
from rex_ai.agents.rex.prompts import SYSTEM_PROMPT

log = logging.getLogger("rex.llm")


def polish(draft: str, facts: dict, language: str, history: list[dict]) -> Optional[str]:
    if not config.USE_LLM:
        return None
    try:
        from openai import OpenAI  # imported lazily so the package stays optional
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            language_name="Arabic" if language == "ar" else "English")}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        messages.append({"role": "user", "content":
                         f"DRAFT:\n{draft}\n\nFACTS:\n{json.dumps(facts, ensure_ascii=False, default=str)[:6000]}\n\n"
                         "Rewrite the DRAFT as the final reply to the user's last message."})
        resp = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=0.3)
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # network, quota, missing package...
        log.warning("LLM polish failed, using deterministic reply: %s", exc)
        return None
