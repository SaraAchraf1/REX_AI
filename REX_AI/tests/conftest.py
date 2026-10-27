import os
import shutil
import sys
import tempfile
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
# Fixed "today" so the sample tenders' deadlines give the same results on any date.
os.environ["REX_RFP_TODAY"] = "2026-10-24"
os.environ.pop("REX_RFP_WEBHOOK_URL", None)
os.environ.pop("REX_SMTP_HOST", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from rex_ai.core import db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db():
    db.init_db()


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