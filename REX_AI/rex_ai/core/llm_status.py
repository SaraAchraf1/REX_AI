"""Why the last LLM call failed, in words a reviewer can act on (owner: Member 3; used by every agent's llm.py).

Change requests ("make the capstone about retail", "shorten the LinkedIn post") are applied by Gemini. When it is
configured but cannot do it - no credits, wrong key, rate limit, network, an unusable answer - the agent raises
LLMUnavailable with the reason and changes nothing, instead of silently guessing with rules.
"""
from __future__ import annotations

import threading
from typing import Optional

_state = threading.local()                         # per request thread (the API serves requests in parallel)


class LLMUnavailable(ValueError):
    """The LLM could not apply a change request; nothing was changed."""


def clear() -> None:
    _state.reason = None


def record(exc: Exception, provider: str = "OpenAI") -> str:
    _state.reason = explain(exc, provider)
    return _state.reason


def reason() -> Optional[str]:
    return getattr(_state, "reason", None)


def explain(exc: Exception, provider: str = "OpenAI") -> str:
    text = str(exc)
    low = text.lower()
    gemini = provider == "Gemini"
    key, model = ("GEMINI_API_KEY", "GEMINI_MODEL") if gemini else ("OPENAI_API_KEY", "OPENAI_MODEL")
    if ("insufficient_quota" in low or "credit_balance" in low or "no credits" in low
            or (gemini and "resource_exhausted" in low and "quota" in low)):
        where = ("https://aistudio.google.com/" if gemini
                 else "https://platform.openai.com/settings/organization/billing")
        return f"the {provider} account has no credits left (or its quota is used up) - check {where} and try again"
    if ("invalid_api_key" in low or "incorrect api key" in low or "api key not valid" in low
            or "api_key_invalid" in low or "401" in low or (gemini and "permission_denied" in low)):
        return f"the {provider} API key is not valid - check {key} in .env"
    if "model_not_found" in low or "does not exist" in low or "no longer available" in low or             (gemini and "404" in low and "model" in low):
        return f"the {provider} model is not available to this key - check {model} in .env"
    if "rate limit" in low or "rate_limit" in low or "429" in low or "resource_exhausted" in low:
        return f"{provider} is rate-limiting this key - wait a minute and try again"
    if "timed out" in low or "timeout" in low or "connection" in low:
        return f"{provider} could not be reached (network or timeout) - try again"
    return f"the {provider} call failed ({text[:200]})"


def unavailable(what: str) -> LLMUnavailable:
    """The error to raise from a change request when Gemini produced nothing usable."""
    from rex_ai import config
    if not config.USE_GEMINI:
        why = ("no LLM is configured for change requests - set GEMINI_API_KEY in .env (and REX_USE_LLM=1), or edit "
               "the fields directly" if not config.GEMINI_API_KEY else "the LLM is switched off (REX_USE_LLM=0)")
    else:
        why = reason() or "the LLM's answer could not be used (it did not follow the required format)"
    return LLMUnavailable(f"The change was not applied to the {what}: {why}. Nothing was changed.")
