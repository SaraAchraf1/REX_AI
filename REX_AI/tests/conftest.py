import os
import sys
import tempfile
from pathlib import Path

# Isolated DB and deterministic mode (no LLM) for every test run - set before rex_ai is imported.
os.environ["REX_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["REX_USE_LLM"] = "0"
# Isolated Chroma store + offline embedder so tests are fast, deterministic and need no download.
os.environ["REX_CHROMA_DIR"] = str(Path(tempfile.mkdtemp()) / "chroma")
os.environ.setdefault("REX_RAG_EMBEDDINGS", "hash")
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