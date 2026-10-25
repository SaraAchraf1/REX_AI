"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
try:    # .env next to requirements.txt; variables already set in the shell (or by tests) win
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env", override=False)
except ImportError:
    pass
DATA_DIR = Path(os.getenv("REX_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = Path(os.getenv("REX_DB_PATH", ROOT_DIR / "rex_ai.db"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Set REX_USE_LLM=0 to force the deterministic (no-LLM) mode, e.g. in tests and demos without a key.
USE_LLM = os.getenv("REX_USE_LLM", "1") == "1" and bool(OPENAI_API_KEY)

# "localhost" makes Windows try IPv6 first; uvicorn listens on IPv4 only, so every call waited ~2 s
API_URL = os.getenv("REX_API_URL", "http://127.0.0.1:8000").replace("//localhost", "//127.0.0.1")

# ---- Shared RAG pipeline (rex_ai/core/rag.py) --------------------------------
CHROMA_DIR = Path(os.getenv("REX_CHROMA_DIR", ROOT_DIR / "data" / "chroma"))
# auto = local sentence-transformers model, falling back to the offline hashing embedder if it can't load
# sentence-transformers | openai | hash
RAG_EMBEDDINGS = os.getenv("REX_RAG_EMBEDDINGS", "auto")
RAG_EMBED_MODEL = os.getenv("REX_RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# rag = shared RAG pipeline (Chroma) for the REX Assistant; tfidf = Sprint 1 in-memory retriever
RETRIEVER = os.getenv("REX_RETRIEVER", "rag").lower()
# OCR for PDF pages whose fonts decode to garbage: auto (use rapidocr_onnxruntime if installed) | off
RAG_OCR = os.getenv("REX_RAG_OCR", "auto").lower()
# Merge new/changed catalogue files from data/catalog/ into the course data when the API starts (1/0)
AUTO_IMPORT = os.getenv("REX_AUTO_IMPORT", "1") == "1"
# ---- Tender / RFP Research Agent (rex_ai/agents/rfp) -------------------------
RFP_DIR = Path(os.getenv("REX_RFP_DIR", DATA_DIR / "rfp"))
# Pin "today" for demos on fixed sample data (YYYY-MM-DD); empty = the real date
RFP_TODAY = os.getenv("REX_RFP_TODAY", "")
RFP_HTTP_TIMEOUT = float(os.getenv("REX_RFP_HTTP_TIMEOUT", "20"))
# Alerts: always stored for the dashboard; also POSTed to a webhook (Slack/Teams/n8n) and/or emailed if set
RFP_WEBHOOK_URL = os.getenv("REX_RFP_WEBHOOK_URL", "")
RFP_ALERT_EMAILS = [e.strip() for e in os.getenv("REX_RFP_ALERT_EMAILS", "").split(",") if e.strip()]
SMTP_HOST = os.getenv("REX_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("REX_SMTP_PORT", "587"))
SMTP_USER = os.getenv("REX_SMTP_USER", "")
SMTP_PASSWORD = os.getenv("REX_SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("REX_SMTP_FROM", SMTP_USER)
# ---- Curriculum & Accreditation Alignment Agent (rex_ai/agents/curriculum) ----
CURRICULUM_DIR = Path(os.getenv("REX_CURRICULUM_DIR", DATA_DIR / "curriculum"))
CURRICULUM_OUTPUT_DIR = Path(os.getenv("REX_CURRICULUM_OUTPUT_DIR", ROOT_DIR / "outputs" / "curricula"))
# Semantic match needed for a framework competency to count as "partial" coverage; empty = per-embedder default
CURRICULUM_SEMANTIC_MIN = os.getenv("REX_CURRICULUM_SEMANTIC_MIN", "")
# TTF font with Arabic glyphs for PDFs (and optional bold variant); empty = first one found on the system
PDF_FONT = os.getenv("REX_PDF_FONT", "")
PDF_FONT_BOLD = os.getenv("REX_PDF_FONT_BOLD", "")
