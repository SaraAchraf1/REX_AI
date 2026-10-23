"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("REX_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = Path(os.getenv("REX_DB_PATH", ROOT_DIR / "rex_ai.db"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Set REX_USE_LLM=0 to force the deterministic (no-LLM) mode, e.g. in tests and demos without a key.
USE_LLM = os.getenv("REX_USE_LLM", "1") == "1" and bool(OPENAI_API_KEY)

API_URL = os.getenv("REX_API_URL", "http://localhost:8000")

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