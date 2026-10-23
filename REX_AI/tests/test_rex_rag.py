"""REX Assistant on top of the shared RAG pipeline (REX_RETRIEVER=rag, the default)."""
import pytest
from fastapi.testclient import TestClient

from rex_ai import config

from rex_ai.agents.rex.knowledge_base import get_kb
from rex_ai.agents.rex.retriever import RAGRetriever, Retriever, make_retriever

pytestmark = pytest.mark.skipif(config.RETRIEVER == "tfidf", reason="REX_RETRIEVER=tfidf")


def test_assistant_uses_rag_retriever(assistant):
    assert isinstance(assistant.retriever, RAGRetriever)
    assert assistant.retriever.rag.collection.name == "academy_kb"


def test_academy_chunks_indexed_once(assistant):
    col = assistant.retriever.rag.collection
    n = len(get_kb().chunks)
    assert col.count() >= n
    RAGRetriever(get_kb().chunks, rag=assistant.retriever.rag)        # second start: no duplicates
    assert len(col.get(where={"origin": "academy_kb"})["ids"]) == n


def test_faq_answered_from_rag_with_source(chat):
    resp = chat("Do I get a certificate?")
    assert resp.intent.startswith("faq:")
    assert any(s.source_id == "SRC-ACAD-FAQ" for s in resp.sources)


def test_arabic_faq_from_rag(chat):
    resp = chat("هل الجلسات مسجلة؟")
    assert resp.intent == "faq:format" and resp.language.value == "ar"


def test_course_search_returns_course_chunk(assistant):
    hits = assistant.retriever.search("power bi dashboards", k=1, filters={"type": "course"})
    assert hits and hits[0].chunk.metadata["course_id"] == "DS-101"


def test_data_change_rebuilds_index(assistant):
    chunks = list(get_kb().chunks)
    chunks[0] = type(chunks[0])(chunks[0].chunk_id, chunks[0].text + " updated", chunks[0].metadata)
    r = RAGRetriever(chunks, rag=assistant.retriever.rag)
    stored = r.rag.collection.get(ids=[chunks[0].chunk_id])["documents"][0]
    assert stored.endswith("updated")
    RAGRetriever(get_kb().chunks, rag=assistant.retriever.rag)        # restore


def test_tfidf_switch(monkeypatch):
    from rex_ai import config
    monkeypatch.setattr(config, "RETRIEVER", "tfidf")
    assert isinstance(make_retriever(get_kb().chunks), Retriever)


def test_rag_search_endpoint(tmp_path):
    from rex_ai.api.main import app
    from rex_ai.core.rag import get_rag
    doc = tmp_path / "notes.md"
    doc.write_text("# Transformers\nSelf-attention compares every token with every other token.\n"
                   "# Graphs\nGraph neural networks pass messages between nodes.", encoding="utf-8")
    get_rag("api_test").ingest(doc, metadata={"source_id": "SRC-TEST"})
    client = TestClient(app)
    r = client.post("/rag/search", json={"query": "message passing on graphs", "collection": "api_test", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["metadata"]["section"] == "Graphs"
    assert body["context"].startswith("[1] Source:") and body["sources"][0]["source_id"] == "SRC-TEST"
    assert client.get("/rag/collections/api_test").json()["chunks"] >= 1
    assert client.get("/health").json()["retriever"] == "RAGRetriever"