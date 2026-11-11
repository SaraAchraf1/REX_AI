"""REX AI tool server (MCP): the ONE place that talks to external services (owner: Member 3).

Every agent reaches OpenAI, Gemini, OpenAI embeddings, SMTP e-mail, webhooks and the public web through these MCP
tools (rex_ai/mcp/client.py), never directly - so keys, timeouts and error messages live in one place, and any MCP
client (Claude Desktop, an IDE, another service) can use the same tools.

Run it standalone and point the API at it with REX_MCP_URL=http://127.0.0.1:8765/mcp:
    python -m rex_ai.mcp.server --http [--host 127.0.0.1] [--port 8765]
    python -m rex_ai.mcp.server             # stdio, e.g. for Claude Desktop
Without REX_MCP_URL the API runs this same server inside its own process (still spoken to over MCP).

A tool that fails raises ToolError with the real reason ("insufficient_quota", "login refused"...), which the
client turns back into an exception the agents already handle.
"""
from __future__ import annotations

import argparse
import base64

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rex_ai import config
from rex_ai.core import mailer

server = MCPServer(
    "rex-ai-tools",
    instructions="External services for the REX AI agents: LLM chat (OpenAI or Gemini), embeddings, e-mail, "
                 "webhooks and web pages. Every tool returns JSON; a failure comes back as an error with the reason.",
    log_level="WARNING",
)


def _fail(exc: Exception) -> ToolError:
    return exc if isinstance(exc, ToolError) else ToolError(str(exc) or type(exc).__name__)


@server.tool()
def llm_chat(messages: list[dict], provider: str = "openai", temperature: float = 0.3, json_mode: bool = False,
             model: str = "") -> dict:
    """Chat completion. provider: openai (generation, assistant, RFP) or gemini (change requests).
    json_mode asks for a JSON object. Returns {content, provider, model}."""
    try:
        from openai import OpenAI
        if provider == "gemini":
            if not config.GEMINI_API_KEY:
                raise ToolError("Gemini is not configured - set GEMINI_API_KEY in .env")
            client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
            model = model or config.GEMINI_MODEL
        elif provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ToolError("OpenAI is not configured - set OPENAI_API_KEY in .env")
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            model = model or config.OPENAI_MODEL
        else:
            raise ToolError(f"Unknown LLM provider '{provider}' (openai | gemini)")
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
        return {"content": resp.choices[0].message.content or "", "provider": provider, "model": model}
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def embed_texts(texts: list[str], model: str = "") -> dict:
    """OpenAI embeddings for the RAG pipeline (REX_RAG_EMBEDDINGS=openai). Returns {vectors, model}."""
    try:
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise ToolError("OpenAI is not configured - set OPENAI_API_KEY in .env")
        client, model = OpenAI(api_key=config.OPENAI_API_KEY), model or config.OPENAI_EMBED_MODEL
        vectors = []
        for i in range(0, len(texts), 256):
            vectors.extend(d.embedding for d in client.embeddings.create(model=model, input=texts[i:i + 256]).data)
        return {"vectors": vectors, "model": model}
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def send_email(to: list[str], subject: str, body: str, attachments: list[dict] | None = None, html: str = "") -> dict:
    """Send a real e-mail over SMTP (REX_SMTP_*). attachments: [{filename, content_base64}]; html: optional HTML
    version (body stays the plain-text alternative).
    Returns {sent, channel, to}; fails with the reason if the message is not sent."""
    try:
        files = [(a["filename"], base64.b64decode(a["content_base64"])) for a in attachments or []]
        return mailer.deliver(to, subject, body, files, html)
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def post_webhook(url: str, payload: dict) -> dict:
    """POST JSON to a webhook (Slack / Teams / n8n). Returns {status_code, ok}."""
    try:
        import requests
        r = requests.post(url, json=payload, timeout=10)
        return {"status_code": r.status_code, "ok": r.status_code < 400}
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def http_get(url: str, timeout: float = 20, headers: dict | None = None) -> dict:
    """Fetch a public web page (tender portals). Returns {status_code, text, url, content_type}."""
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers=headers or {})
        return {"status_code": r.status_code, "text": r.text, "url": r.url,
                "content_type": r.headers.get("content-type", "")}
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def smtp_check() -> dict:
    """Log in to the mail server without sending anything. Returns {ok, host, port}."""
    try:
        return mailer.check_login()
    except Exception as exc:
        raise _fail(exc) from exc


@server.tool()
def service_status() -> dict:
    """Which external services are configured (never the secrets)."""
    return {
        "openai": {"configured": bool(config.OPENAI_API_KEY), "model": config.OPENAI_MODEL,
                   "enabled": config.USE_LLM, "used_for": "generation, REX assistant, RFP summaries"},
        "gemini": {"configured": bool(config.GEMINI_API_KEY), "model": config.GEMINI_MODEL,
                   "enabled": config.USE_GEMINI, "used_for": "change requests (curriculum, documents, marketing)"},
        "embeddings": {"backend": config.RAG_EMBEDDINGS, "openai_model": config.OPENAI_EMBED_MODEL},
        "smtp": {"configured": mailer.configured(), "host": config.SMTP_HOST, "port": config.SMTP_PORT,
                 "login_set": bool(config.SMTP_USER and config.SMTP_PASSWORD)},
        "webhook": {"configured": bool(config.RFP_WEBHOOK_URL)},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="REX AI MCP tool server")
    p.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless the network is trusted: the tools "
                                                       "use the API keys and the mail account")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    if a.http:
        server.run("streamable-http", host=a.host, port=a.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
