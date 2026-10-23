"""Lightweight bilingual retriever over knowledge-base chunks.

Uses TF-IDF on character n-grams, which works for Arabic and English without
downloading an embedding model. The interface (search(query, k, filters)) matches
what a Chroma/FAISS retriever would expose. Sprint 2: RAGRetriever (below) implements the same
interface on top of the shared RAG pipeline (rex_ai/core/rag.py); make_retriever() picks one.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rex_ai import config
from rex_ai.agents.rex.knowledge_base import Chunk
from rex_ai.agents.rex.text_utils import normalize


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                          preprocessor=normalize)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 4, filters: Optional[dict] = None,
               min_score: float = 0.0) -> list[Hit]:
        """filters: exact-match metadata, e.g. {"type": "faq"} or {"department": "AI"}."""
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        hits = []
        for idx in scores.argsort()[::-1]:
            chunk, score = self.chunks[idx], float(scores[idx])
            if score < min_score:
                break
            if filters and any(chunk.metadata.get(key) != val for key, val in filters.items()):
                continue
            hits.append(Hit(chunk, score))
            if len(hits) == k:
                break
        return hits


# =============================================================================
# Sprint 2: the same interface backed by the shared RAG pipeline (Chroma + embeddings)
# =============================================================================
log = logging.getLogger("rex.retriever")

# Score thresholds the assistant uses (course fallback, FAQ match, FAQ lead over runner-up).
# Cosine scores depend on the embedder, so each backend has its own values. Tune with
# `python scripts/eval_retriever.py`; override with REX_COURSE_MIN_SCORE / REX_FAQ_MIN_SCORE / REX_FAQ_MARGIN.
THRESHOLDS = {
    "tfidf": {"course_min": 0.30, "faq_min": 0.20, "faq_margin": 0.10, "doc_min": 1.0},
    "hash": {"course_min": 0.28, "faq_min": 0.28, "faq_margin": 0.03, "doc_min": 0.30},
    "st": {"course_min": 0.80, "faq_min": 0.84, "faq_margin": 0.015, "doc_min": 0.82},      # multilingual-e5-small
    "openai": {"course_min": 0.35, "faq_min": 0.45, "faq_margin": 0.03, "doc_min": 0.40},
}


def _thresholds(backend: str) -> dict:
    t = dict(THRESHOLDS.get(backend, THRESHOLDS["st"]))
    for key, env in (("course_min", "REX_COURSE_MIN_SCORE"), ("faq_min", "REX_FAQ_MIN_SCORE"),
                     ("faq_margin", "REX_FAQ_MARGIN"), ("doc_min", "REX_DOC_MIN_SCORE")):
        if os.getenv(env):
            t[key] = float(os.getenv(env))
    return t


Retriever.backend = "tfidf"
Retriever.thresholds = property(lambda self: _thresholds("tfidf"))
Retriever.search_docs = lambda self, query, k=3, min_score=0.0: []    # documents need the RAG pipeline


class RAGRetriever:
    """Drop-in replacement for Retriever: course + FAQ chunks are indexed in the Chroma collection
    `academy_kb` and searched semantically. The index is rebuilt automatically when data/*.json change.
    Uploaded documents (data/catalog/, data/documents/) live in the same collection under DOCS_ORIGIN and
    are kept when the course/FAQ chunks are rebuilt."""

    COLLECTION = "academy_kb"
    ORIGIN = "academy_kb"
    DOCS_ORIGIN = "academy_docs"

    def __init__(self, chunks: list[Chunk], collection: str = COLLECTION, rag=None):
        from rex_ai.core.rag import get_rag
        self.chunks = {c.chunk_id: c for c in chunks}
        self.rag = rag or get_rag(collection)
        self.backend = self.rag.embedder.name.split(":", 1)[0]
        self.thresholds = _thresholds(self.backend)
        self._sync(chunks)

    def _sync(self, chunks: list[Chunk]):
        fingerprint = hashlib.sha1("".join(c.chunk_id + c.text for c in chunks).encode("utf-8")).hexdigest()[:16]
        col = self.rag.collection
        if col.get(where={"$and": [{"origin": {"$eq": self.ORIGIN}}, {"kb_fingerprint": {"$eq": fingerprint}}]},
                   limit=1)["ids"]:
            return
        self.rag.delete(where={"origin": self.ORIGIN})               # data changed -> rebuild
        self.rag.add_texts([c.text for c in chunks],
                           [{**c.metadata, "origin": self.ORIGIN, "kb_fingerprint": fingerprint,
                             "source": c.metadata.get("source_id", "")} for c in chunks],
                           ids=[c.chunk_id for c in chunks])
        log.info("Indexed %d academy chunks into '%s'", len(chunks), col.name)

    def search(self, query: str, k: int = 4, filters: Optional[dict] = None,
               min_score: float = 0.0) -> list[Hit]:
        where = {"origin": self.ORIGIN, **(filters or {})}
        hits = self.rag.search(query, k=k, where=where, min_score=min_score)
        return [Hit(self.chunks[h.chunk_id], h.score) for h in hits if h.chunk_id in self.chunks]

    def search_docs(self, query: str, k: int = 3, min_score: float = 0.0):
        """Passages from uploaded documents (rex_ai.core.rag.Hit objects, with citations)."""
        return self.rag.search(query, k=k, where={"origin": self.DOCS_ORIGIN}, min_score=min_score, mmr=True)


def make_retriever(chunks: list[Chunk]):
    """REX_RETRIEVER=rag (default) uses the shared RAG pipeline; tfidf keeps the Sprint 1 retriever.
    If the RAG stack can't start (missing package, embedder mismatch...), REX falls back to TF-IDF."""
    if config.RETRIEVER == "tfidf":
        return Retriever(chunks)
    try:
        return RAGRetriever(chunks)
    except Exception as exc:
        log.warning("RAG retriever unavailable (%s) - using TF-IDF.", exc)
        return Retriever(chunks)