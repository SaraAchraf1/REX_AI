import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Isolated DB and deterministic mode (no LLM) for every test run - set before rex_ai is imported.
os.environ["REX_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["REX_USE_LLM"] = "0"
# Isolated Chroma store + offline embedder so tests are fast, deterministic and need no download.
os.environ["REX_CHROMA_DIR"] = str(Path(tempfile.mkdtemp()) / "chroma")
os.environ.setdefault("REX_RAG_EMBEDDINGS", "hash")
# A copy of data/: catalogue auto-import, uploads and merges in tests never rewrite the real data files.
_data_copy = Path(tempfile.mkdtemp()) / "data"
shutil.copytree(Path(__file__).resolve().parent.parent / "data", _data_copy,
                ignore=shutil.ignore_patterns("chroma", "raw", "kb", "backup"))
os.environ["REX_DATA_DIR"] = str(_data_copy)
os.environ["REX_RFP_DIR"] = str(_data_copy / "rfp")
os.environ["REX_CURRICULUM_OUTPUT_DIR"] = str(Path(tempfile.mkdtemp()) / "curricula")
os.environ["REX_DOCUMENT_OUTPUT_DIR"] = str(Path(tempfile.mkdtemp()) / "documents")
os.environ["REX_MARKETING_OUTPUT_DIR"] = str(Path(tempfile.mkdtemp()) / "campaigns")
# Fixed "today" so the sample tenders' deadlines give the same results on any date.
os.environ["REX_RFP_TODAY"] = "2026-10-24"
os.environ["REX_RFP_WEBHOOK_URL"] = ""           # empty, not removed: .env must not fill them in
os.environ["REX_SMTP_HOST"] = ""                 # no real e-mail from tests
os.environ["REX_MCP_URL"] = ""                   # the MCP tool server runs in-process
os.environ["REX_ORCH_INTERVAL"] = "0"            # no background flow checks; tests call them
for _k, _v in (("REX_SMTP_PORT", "587"), ("REX_SMTP_USER", ""), ("REX_SMTP_PASSWORD", ""), ("REX_SMTP_FROM", "")):
    os.environ[_k] = _v                          # tests' fake SMTP server, whatever .env says
# Never the shared database from .env: SQLite by default; REX_TEST_DATABASE_URL runs the whole suite on PostgreSQL in
# a throw-away schema (dropped at the end). Set before rex_ai is imported (.env does not override set variables).
_TEST_PG = os.environ.get("REX_TEST_DATABASE_URL", "")
os.environ["REX_DATABASE_URL"] = _TEST_PG
for _name in ("DATABASE_URL", "DATABASE_URL_POOLED", "DATABASE_URL_UNPOOLED", "REX_DATABASE_URL_DIRECT"):
    os.environ[_name] = ""                     # e.g. Neon's URLs in .env: never touched by the tests
os.environ["REX_SHARED_FILES"] = "0"
# Agent tests call the API without logging in; tests/test_auth.py switches the login check on.
os.environ["REX_API_AUTH"] = "0"
os.environ["REX_API_CACHE"] = "0"             # tests change the database directly; test_api_cache.py switches it on
os.environ["REX_ADMIN_SIGNUP_CODE"] = "test-admin-code"
os.environ["REX_DB_SCHEMA"] = f"rex_test_{os.getpid()}_{int(time.time())}" if _TEST_PG else ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from rex_ai.core import db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db():
    db.init_db()
    yield
    if _TEST_PG:                                   # drop the throw-away PostgreSQL schema
        with db.connect() as conn:
            conn.raw.execute(f'DROP SCHEMA IF EXISTS "{os.environ["REX_DB_SCHEMA"]}" CASCADE')
        db.close_pools()


@pytest.fixture(scope="session")
def assistant():
    from rex_ai.agents.rex.assistant import RexAssistant
    return RexAssistant()


@pytest.fixture
def chat(assistant):
    """chat('msg', sid) -> ChatResponse; each test gets a fresh session id by default."""
    import uuid
    from rex_ai.core.schemas import ChatRequest
    sid = uuid.uuid4().hex[:8]

    def _send(message, session_id=None, language=None):
        return assistant.handle(ChatRequest(session_id=session_id or sid, message=message, language=language))
    return _send