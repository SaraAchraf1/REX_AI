"""Shared RAG pipeline for every REX AI agent (Sprint 2).

One module, used by all five agents:

    from rex_ai.core.rag import get_rag

    rag = get_rag("curriculum_refs")                       # one Chroma collection per knowledge area
    rag.ingest("data/raw/mit_6_7960/", metadata={"source_id": "SRC-MIT-OCW-6.7960", "department": "AI"})
    hits = rag.search("How does self-attention work?", k=5, where={"department": "AI"})
    prompt_context = rag.build_context(hits)              # numbered, cited context for the LLM
    sources = rag.to_source_refs(hits)                    # SourceRef list -> traceability rule

Pipeline
    load      PDF (pdfplumber, falls back to pypdf) | Word .docx | Excel .xlsx/.xlsm/.xls | CSV | TXT/MD
              Each loader keeps the document's structure: headings, pages, tables, sheets and rows.
    chunk     1. document-aware : split on the document's own structure (heading sections, Excel sheets);
                                  never mix two sections in one chunk; every chunk carries its heading path.
              2. semantic       : long sections are cut where the topic shifts (embedding distance between
                                  neighbouring sentences crosses a percentile threshold).
              3. recursive      : any piece still above chunk_size is split on paragraph -> line ->
                                  sentence -> word boundaries, with overlap.
    embed     sentence-transformers (multilingual, Arabic + English) | OpenAI | offline hashing fallback
    store     ChromaDB (persistent, cosine), idempotent: unchanged files are skipped, changed files replaced.
    search    semantic search + metadata filters + optional MMR re-ranking for diverse results.

CLI
    python -m rex_ai.core.rag ingest data/raw/mit_6_7960 --collection curriculum_refs --source-id SRC-MIT-OCW-6.7960
    python -m rex_ai.core.rag query "what is a graph neural network" --collection curriculum_refs -k 5
    python -m rex_ai.core.rag stats --collection curriculum_refs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from rex_ai import config

log = logging.getLogger("rex.rag")

PathLike = Union[str, Path]
# Bump when loaders/chunking change: files ingested by an older version are re-ingested automatically.
PIPELINE_VERSION = "6"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".txt", ".md"}


# =============================================================================
# Data model
# =============================================================================
@dataclass
class Block:
    """A structural unit produced by a loader (a paragraph, a table, a spreadsheet row...)."""
    text: str
    heading_path: list[str] = field(default_factory=list)
    kind: str = "text"                    # text | table | row
    page: Optional[int] = None            # 1-based page for PDFs
    extra: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    doc_id: str                           # sha256 of the file bytes -> unchanged files are skipped
    source_path: str
    title: str
    file_type: str
    blocks: list[Block]
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str                             # what is embedded and returned (includes the context header)
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float                          # cosine similarity, 1 = identical

    @property
    def citation(self) -> str:
        m = self.metadata
        parts = [m.get("title") or m.get("source", "")]
        if m.get("section"):
            parts.append(m["section"])
        if m.get("page_start"):
            pages = m["page_start"] if m.get("page_end") in (None, m["page_start"]) else f"{m['page_start']}-{m['page_end']}"
            parts.append(f"p. {pages}")
        if m.get("sheet"):
            parts.append(f"sheet {m['sheet']}, rows {m.get('row_start')}-{m.get('row_end')}")
        return " — ".join(str(p) for p in parts if p)


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000                # max characters per chunk (~250 tokens)
    chunk_overlap: int = 150              # characters repeated between recursive splits
    min_chunk_size: int = 200             # smaller pieces are merged into a neighbour from the same section
    document_aware: bool = True
    semantic: bool = True
    semantic_percentile: float = 85.0     # higher = fewer, larger semantic groups
    semantic_window: int = 1              # neighbouring sentences combined when measuring topic shift
    context_header: bool = True           # prefix "[Title > Section]" so each chunk is self-describing


# =============================================================================
# Embeddings
# =============================================================================
class Embedder:
    name = "base"
    dim: Optional[int] = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Local model. Default multilingual-e5-small: Arabic + English, 512-token input, ~470 MB."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        try:    # use the cached copy without contacting the Hub (no retries on slow/blocked networks)
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:   # first run: download it
            self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self._e5 = "e5" in model_name.lower()   # e5 models expect "query: " / "passage: " prefixes
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts] if self._e5 else list(texts)
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, text):
        q = f"query: {text}" if self._e5 else text
        return self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model_name = model_name
        self.name = f"openai:{model_name}"

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(model=self.model_name, input=list(texts[i:i + 256]))
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)


class HashingEmbedder(Embedder):
    """Offline, deterministic fallback (character + word n-gram hashing). No download, works for Arabic.
    Good enough for tests and demos without internet; use a real model for the final system."""
    name = "hash:v1"
    dim = 1024

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=768,
                                       alternate_sign=False, norm=None, lowercase=True)
        self._word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=256,
                                       alternate_sign=False, norm=None, lowercase=True)

    def embed_documents(self, texts):
        c = self._char.transform(texts).toarray()
        w = self._word.transform(texts).toarray() * 2.0
        arr = np.sqrt(np.hstack([c, w])).astype(np.float32)       # sqrt damps very frequent n-grams
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)


@lru_cache(maxsize=4)
def get_embedder(backend: Optional[str] = None, model: Optional[str] = None) -> Embedder:
    """backend: auto | sentence-transformers | openai | hash   (env REX_RAG_EMBEDDINGS)."""
    backend = (backend or config.RAG_EMBEDDINGS).lower()
    if backend == "hash":
        return HashingEmbedder()
    if backend == "openai":
        return OpenAIEmbedder(model or config.OPENAI_EMBED_MODEL)
    try:
        return SentenceTransformerEmbedder(model or config.RAG_EMBED_MODEL)
    except Exception as exc:
        if backend != "auto":
            raise
        log.warning("Embedding model unavailable (%s) - using the offline hashing embedder.", exc)
        return HashingEmbedder()


# =============================================================================
# Loaders (document-aware: keep headings, pages, tables, sheets)
# =============================================================================
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+){0,3}\.?|[IVX]+\.|Chapter\s+\d+|Lecture\s+\d+|Section\s+\d+)\s+\S")


# Slide decks made with Keynote/PowerPoint + LaTeXiT embed the LaTeX source of every equation as invisible
# text: <latexit sha1_base64="...">AAAC...base64...</latexit>. PDF extraction splits it into many short
# fragments (sometimes inside "table" cells), so it is removed both as whole blocks and token by token.
_LATEXIT_BLOCK = re.compile(r"<latexit\b[^>]*>.*?</latexit>", re.S | re.I)
_LATEXIT_TAG = re.compile(r"</?latexit\b[^>]*>?|sha1_base64=\"?[^\s\">]*\"?>?", re.I)
_CITATION = re.compile(r"^\(.*\)$|\bet al\b|\b(19|20)\d{2}[a-z]?\)?$", re.I)
_SLIDE_NUMBER = re.compile(r"^\d{1,3}$")
# Fonts without a Unicode map come out of pdfminer as "(cid:55)" placeholders instead of letters.
_CID = re.compile(r"\(cid:\d+\)")


def _char_class(c: str) -> int:
    return 0 if c.islower() else 1 if c.isupper() else 2 if c.isdigit() else 3


def _is_base64_token(token: str) -> bool:
    """Random-looking [A-Za-z0-9+/] runs (base64 data), not real words such as PyTorch, ResNet50, input/output."""
    core = token.strip("\"'<>=,.;:()[]{}")
    if len(core) < 12 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", core):
        return False
    core = core.rstrip("=")
    if not any(c.isdigit() or c in "+/" for c in core):
        return False
    upper, lower = sum(c.isupper() for c in core), sum(c.islower() for c in core)
    switches = sum(_char_class(a) != _char_class(b) for a, b in zip(core, core[1:]))
    return upper >= 2 and lower >= 2 and switches / len(core) > 0.3


def _is_noise_word(word: str) -> bool:
    """A PDF 'word' that is embedded LaTeX/base64 data rather than visible text."""
    low = word.lower()
    return "latexit" in low or "base64" in low or "sha1_" in low or _is_base64_token(word)


def _cid_ratio(text: str) -> float:
    return len(_CID.findall(text)) / max(len(text.split()), 1)


def _strip_noise(text: str) -> str:
    text = _CID.sub("", text)
    text = _LATEXIT_BLOCK.sub(" ", text)
    text = _LATEXIT_TAG.sub(" ", text)
    return re.sub(r"\S+", lambda m: " " if _is_noise_word(m.group()) else m.group(), text)


def _is_junk(text: str) -> bool:
    """True when a piece has (almost) no readable words, e.g. only leftover symbols or encoded data."""
    words = re.findall(r"[^\W\d_]{2,}", text)
    return len("".join(words)) < max(15, 0.35 * len(text.replace(" ", "")))


def _clean(text: str) -> str:
    text = _strip_noise(text)
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:24]


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if _CITATION.search(s):
        return False
    if not (2 < len(s) <= 90) or s.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADING.match(s):
        return False
    if _NUMBERED_HEADING.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and len(s.split()) <= 10 and sum(c.isupper() for c in letters) / len(letters) > 0.8


# Characters that appear when a PDF font maps glyphs to the wrong letters (",oadma«" = "Roadmap").
_GARBLE_CHARS = set("«»ƂƁƀ¬¦¤¨¸¯þÞðÐ¹²³¼½¾") | {chr(c) for c in range(0x0180, 0x0250)}


def _is_figure_noise(line: str) -> bool:
    """OCR of diagrams: 'oFαTo■oTαTo', '{ 0o 0', 'UC', 'C' - fragments, not sentences."""
    t = line.strip()
    if len(t) <= 3 or "■" in t or "□" in t:
        return True
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 0.5 * len(t.replace(" ", "")):
        return True
    for tok in t.split():                                   # Latin and Greek letters mixed inside one word
        latin = any("a" <= c.lower() <= "z" for c in tok)
        greek = any("\u0370" <= c <= "\u03ff" for c in tok)
        if latin and greek and len(tok) > 2:
            return True
    return False


def _is_garbled(text: str) -> bool:
    """True when page text looks wrongly decoded: (cid:N) codes, or words with stray symbols
    such as ',oadma«' / '*ower'. Arabic and math symbols are fine."""
    if _cid_ratio(text) > 0.05:
        return True
    tokens = text.split()
    if len(tokens) < 3:
        return False
    bad = sum(1 for t in tokens if any(c in _GARBLE_CHARS for c in t) or re.match(r"^[,*][A-Za-z]", t))
    return bad >= 2 and bad / len(tokens) > 0.03


@lru_cache(maxsize=1)
def _ocr_engine():
    """RapidOCR (pip install rapidocr_onnxruntime): offline, models included, no system install.
    Disabled with REX_RAG_OCR=off."""
    if config.RAG_OCR == "off":
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        log.warning("Some PDF pages have badly encoded fonts. Install OCR to read them correctly: "
                    "pip install rapidocr_onnxruntime")
        return None


def _ocr_page(path: Path, index: int) -> Optional[list[dict]]:
    """Render one page and OCR it. Returns lines [{text, size, top}] in PDF points, or None."""
    engine = _ocr_engine()
    if engine is None:
        return None
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        try:
            page = doc[index]
            scale = 1280 / max(page.get_width(), page.get_height())     # ~1280 px long side works best
            result, _ = engine(page.render(scale=scale).to_numpy())
        finally:
            doc.close()
    except Exception as exc:
        log.warning("OCR failed on %s p.%d: %s", path.name, index + 1, exc)
        return None
    lines = []
    for box, text, conf in result or []:
        if float(conf) < 0.75 or not text.strip() or _is_figure_noise(text):
            continue
        top = min(pt[1] for pt in box) / scale
        height = (max(pt[1] for pt in box) - min(pt[1] for pt in box)) / scale
        lines.append({"text": text.strip(), "size": round(height * 0.75, 1), "top": top})   # box height ~ 1.33x font
    return sorted(lines, key=lambda ln: ln["top"])


def _words_to_lines(words: list[dict], min_size: float = 0.0) -> list[dict]:
    lines, cur = [], []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if w["size"] < min_size:
            continue
        if cur and abs(w["top"] - cur[-1]["top"]) > 3:
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [{"text": " ".join(w["text"] for w in sorted(ln, key=lambda w: w["x0"])),
             "size": max(round(w["size"], 1) for w in ln), "top": min(w["top"] for w in ln)} for ln in lines]


def load_pdf(path: Path) -> list[Block]:
    try:
        import pdfplumber
    except ImportError:
        return _load_pdf_pypdf(path)

    pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages):
            words = [w for w in page.extract_words(extra_attrs=["size"], use_text_flow=False)
                     if not _is_noise_word(w["text"])]
            tables = []
            for t in page.find_tables():
                rendered = _render_table(t.extract())
                if rendered:                                    # only tables that look like real tables
                    tables.append((t.bbox, rendered))
            if tables:                                          # table text is kept once, as the table block
                words = [w for w in words if not any(_inside(w, bbox) for bbox, _ in tables)]
            lines, source = _words_to_lines(words), "text"

            # badly encoded fonts: try pdfium's decoder, then OCR the rendered page
            if lines and _is_garbled(" ".join(ln["text"] for ln in lines)):
                alt = _page_text_pdfium(path, index)
                if alt and not _is_garbled(alt):
                    lines = [{"text": x, "size": None, "top": None} for x in alt.splitlines() if x.strip()]
                    source = "pdfium"
                else:
                    ocr = _ocr_page(path, index)
                    if ocr:
                        lines, source = ocr, "ocr"
            pages.append({"lines": lines, "tables": [r for _, r in tables], "source": source,
                          "landscape": page.width > page.height, "height": float(page.height)})

    all_lines = [ln for pg in pages for ln in pg["lines"]]
    if not all_lines:
        log.warning("%s has no extractable text (scanned PDF?) - install OCR: pip install rapidocr_onnxruntime",
                    path.name)
        return []
    if sum(pg["source"] != "text" for pg in pages):
        log.info("%s: %d page(s) re-read with pdfium/OCR because of badly encoded fonts", path.name,
                 sum(pg["source"] != "text" for pg in pages))
    slides = sum(pg["landscape"] for pg in pages) >= 0.6 * len(pages)
    return _pdf_blocks_slides(pages) if slides else _pdf_blocks_document(pages)


def _pdf_blocks_slides(pages: list[dict]) -> list[Block]:
    """Slide decks: each slide's title = the largest line(s) at the top of the slide.
    A slide without a title continues the previous slide's section."""
    blocks: list[Block] = []
    title = ""
    deck_sizes = [ln["size"] for pg in pages for ln in pg["lines"] if ln["size"]]
    deck_median = statistics.median(deck_sizes) if deck_sizes else 0
    for page_no, pg in enumerate(pages, start=1):
        lines = [dict(ln, text=_clean(ln["text"])) for ln in pg["lines"]]
        lines = [ln for ln in lines if ln["text"] and not _SLIDE_NUMBER.match(ln["text"])]
        if not lines and not pg["tables"]:
            continue
        sized = [ln for ln in lines if ln["size"]]
        new_title, body = None, lines
        if sized:
            biggest = max(ln["size"] for ln in sized)
            median = statistics.median(ln["size"] for ln in sized)
            top_zone = pg["height"] * 0.3
            head = []
            for ln in lines:                                   # leading run of title-sized lines at the top
                if (ln["size"] and ln["size"] >= biggest * 0.95 and biggest >= median * 1.1 * (len(sized) > 1)
                        and biggest >= deck_median * 1.1
                        and ln["top"] is not None and ln["top"] <= top_zone):
                    head.append(ln)
                else:
                    break
            candidate = " ".join(ln["text"] for ln in head)
            if head and len(candidate.split()) <= 12 and not _CITATION.search(candidate):
                new_title, body = candidate, lines[len(head):]
        elif lines and len(lines[0]["text"].split()) <= 10 and not _CITATION.search(lines[0]["text"]):
            new_title, body = lines[0]["text"], lines[1:]      # no font sizes (pdfium text): first line
        if new_title:
            title = new_title
        text = _clean("\n".join(ln["text"] for ln in body))
        if text:
            blocks.append(Block(text, [title] if title else [], "text", page_no, {"text_source": pg["source"]}))
        for rendered in pg["tables"]:
            blocks.append(Block(rendered, [title] if title else [], "table", page_no))
    return blocks


def _pdf_blocks_document(pages: list[dict]) -> list[Block]:
    """Portrait documents (papers, notes, reports): headings = large, rarely used font sizes."""
    sizes = [ln["size"] for pg in pages for ln in pg["lines"] for _ in ln["text"] if ln["size"]]
    body = statistics.median(sizes) if sizes else 0
    share = {sz: sizes.count(sz) / len(sizes) for sz in set(sizes)} if sizes else {}
    heading_sizes = sorted((sz for sz in share if sz >= body * 1.15 and share[sz] <= 0.2), reverse=True)[:2]

    blocks: list[Block] = []
    path_stack: list[tuple[int, str]] = []
    for page_no, pg in enumerate(pages, start=1):
        buf: list[str] = []
        last_level: Optional[int] = None

        def flush():
            text = _clean("\n".join(buf))
            if text:
                blocks.append(Block(text, [h for _, h in path_stack], "text", page_no, {"text_source": pg["source"]}))
            buf.clear()

        for ln in pg["lines"]:
            text, size = _clean(ln["text"]), ln["size"] or body
            if not text or _SLIDE_NUMBER.match(text):
                continue
            if _is_pdf_heading(text, size, body, len(pg["lines"]), heading_sizes):
                level = heading_sizes.index(size) if size in heading_sizes else len(heading_sizes)
                if last_level == level and not buf and path_stack and \
                        len(f"{path_stack[-1][1]} {text}".split()) <= 12:
                    path_stack[-1] = (level, f"{path_stack[-1][1]} {text}")    # title wrapped onto 2 lines
                    continue
                flush()
                while path_stack and path_stack[-1][0] >= level:
                    path_stack.pop()
                path_stack.append((level, text))
                last_level = level
            else:
                buf.append(text)
                last_level = None
        flush()
        for rendered in pg["tables"]:
            blocks.append(Block(rendered, [h for _, h in path_stack], "table", page_no))
    return blocks


def _page_text_pdfium(path: Path, index: int) -> Optional[str]:
    """Re-read one page with pdfium (ships with pdfplumber); used when pdfminer's text is garbled.
    Returns None if pdfium can't read the page."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        try:
            text = doc[index].get_textpage().get_text_range()
        finally:
            doc.close()
    except Exception as exc:
        log.debug("pdfium fallback failed on %s p.%d: %s", path.name, index + 1, exc)
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufffd", "")
    return text if text.strip() else None


def _inside(word: dict, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    cx, cy = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _is_pdf_heading(text: str, size: float, body: float, n_lines: int, heading_sizes: Sequence[float] = ()) -> bool:
    n_words = len(text.split())
    if _CITATION.search(text) or n_words > 12:
        return False
    if text.endswith((".", ",", ";")) and n_words > 6:      # a sentence, not a title
        return False
    big = size in heading_sizes if heading_sizes else size >= body * 1.15
    if big:
        return True
    if heading_sizes:            # the PDF has real title fonts: numbered list items are bullets, not headings
        return False
    # no distinct title font: short numbered ("2.1 Results") or ALL-CAPS lines count as headings
    return size >= body and n_lines > 1 and n_words <= 8 and _looks_like_heading(text)


def _load_pdf_pypdf(path: Path) -> list[Block]:
    from pypdf import PdfReader
    blocks, heading_path = [], []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        buf: list[str] = []
        for line in (page.extract_text() or "").splitlines():
            if _looks_like_heading(line):
                if buf:
                    blocks.append(Block(_clean("\n".join(buf)), list(heading_path), "text", page_no))
                    buf = []
                heading_path = [line.strip()]
            else:
                buf.append(line)
        if buf and _clean("\n".join(buf)):
            blocks.append(Block(_clean("\n".join(buf)), list(heading_path), "text", page_no))
    return blocks


def _render_table(rows: list[list[Any]]) -> str:
    """Rows as "header: value" lines. Returns "" for layout artefacts that are not real tables
    (slides often produce 'tables' with one column, mostly empty cells or noise)."""
    rows = [[_clean("" if c is None else str(c).replace("\n", " ")) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if len(rows) < 3 or max(len(r) for r in rows) < 2:
        return ""
    cells = [c for r in rows for c in r]
    if sum(bool(c) for c in cells) / len(cells) < 0.6 or sum(bool(h) for h in rows[0]) / len(rows[0]) < 0.5:
        return ""
    header = [h or f"col{i + 1}" for i, h in enumerate(rows[0])]
    return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for r in rows[1:])


def load_docx(path: Path) -> list[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[Block] = []
    path_stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def heading_path():
        return [t for _, t in path_stack]

    def flush():
        if buf:
            blocks.append(Block(_clean("\n".join(buf)), heading_path()))
            buf.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            m = re.match(r"(Heading|عنوان)\s*(\d)", style)
            level = 0 if style == "Title" else int(m.group(2)) if m else None
            if level is not None:
                flush()
                while path_stack and path_stack[-1][0] >= level:
                    path_stack.pop()
                path_stack.append((level, text))
            elif style.startswith("List"):
                buf.append(f"- {text}")
            else:
                buf.append(text)
        elif tag == "tbl":
            flush()
            table = Table(child, document)
            rendered = _render_table([[cell.text for cell in row.cells] for row in table.rows])
            if rendered:
                blocks.append(Block(rendered, heading_path(), "table"))
    flush()
    return blocks


def load_spreadsheet(path: Path) -> list[Block]:
    import pandas as pd
    if path.suffix.lower() == ".csv":
        sheets = {path.stem: pd.read_csv(path, dtype=str, keep_default_na=False)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    blocks = []
    for sheet, df in sheets.items():
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed") or df[c].str.strip().any()]]
        columns = [str(c).strip() for c in df.columns]
        for i, row in enumerate(df.itertuples(index=False), start=2):   # row 1 is the header
            cells = [(col, str(v).strip()) for col, v in zip(columns, row) if str(v).strip()]
            if cells:
                blocks.append(Block("; ".join(f"{c}: {v}" for c, v in cells), [str(sheet)], "row",
                                    extra={"sheet": str(sheet), "row": i, "columns": ", ".join(columns)}))
    return blocks


def load_text(path: Path) -> list[Block]:
    blocks, path_stack, buf = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                blocks.append(Block(_clean("\n".join(buf)), [t for _, t in path_stack]))
                buf = []
            level = len(m.group(1))
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    if buf and _clean("\n".join(buf)):
        blocks.append(Block(_clean("\n".join(buf)), [t for _, t in path_stack]))
    return blocks


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".xlsx": load_spreadsheet, ".xlsm": load_spreadsheet,
           ".xls": load_spreadsheet, ".csv": load_spreadsheet, ".txt": load_text, ".md": load_text}


def load_document(path: PathLike, metadata: Optional[dict] = None) -> LoadedDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".doc":
        raise ValueError(f"{path.name}: old .doc format - save it as .docx first.")
    if ext not in LOADERS:
        raise ValueError(f"{path.name}: unsupported type {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    blocks = [b for b in LOADERS[ext](path) if b.text.strip()]
    metadata = dict(metadata or {})
    title = metadata.pop("title", None) or path.stem.replace("_", " ")
    return LoadedDocument(_file_hash(path), str(path), title, ext.lstrip("."), blocks, metadata)


# =============================================================================
# Chunking: document-aware -> semantic -> recursive
# =============================================================================
_SENTENCE_END = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "? ", "! ", "; ", "، ", ", ", " ", ""]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def recursive_split(text: str, chunk_size: int, overlap: int,
                    separators: Sequence[str] = RECURSIVE_SEPARATORS) -> list[str]:
    """Split on the coarsest separator present (paragraph -> line -> sentence -> word), recurse into
    pieces that are still too big, then pack pieces back up to chunk_size with `overlap` chars carried over."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    idx = next(i for i, s in enumerate(separators) if s == "" or s in text)
    sep, rest = separators[idx], separators[idx + 1:]
    if sep == "":                                              # no boundary left: hard cut
        step = max(chunk_size - overlap, 1)
        return [text[i:i + chunk_size].strip() for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]

    parts = text.split(sep)
    units = [p + sep for p in parts[:-1]] + [parts[-1]]        # keep the separator with its piece
    small: list[str] = []
    for u in units:
        if len(u) > chunk_size:
            small.extend(s + " " for s in recursive_split(u, chunk_size, overlap, rest))
        elif u.strip():
            small.append(u)

    chunks, cur = [], ""
    for u in small:
        if cur and len(cur) + len(u) > chunk_size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail   # start overlap on a word boundary
            cur = tail + u if len(tail) + len(u) <= chunk_size else u
        else:
            cur += u
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def semantic_split(text: str, embedder: Embedder, percentile: float = 85.0, window: int = 1,
                   min_size: int = 200) -> list[str]:
    """Group consecutive sentences; cut where the embedding distance to the next sentence window is in
    the top (100 - percentile)% of distances for this section, i.e. where the topic changes."""
    sentences = split_sentences(text)
    if len(sentences) < 4:
        return [text]
    windows = [" ".join(sentences[max(0, i - window): i + window + 1]) for i in range(len(sentences))]
    emb = embedder.embed_documents(windows)
    distances = [1.0 - float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    threshold = float(np.percentile(distances, percentile))

    groups, cur = [], [sentences[0]]
    for i, d in enumerate(distances):
        if d > threshold and len(" ".join(cur)) >= min_size:
            groups.append(" ".join(cur))
            cur = []
        cur.append(sentences[i + 1])
    groups.append(" ".join(cur))
    return groups


class Chunker:
    def __init__(self, cfg: Optional[ChunkingConfig] = None, embedder: Optional[Embedder] = None):
        self.cfg = cfg or ChunkingConfig()
        self.embedder = embedder

    # -- step 1: document-aware sections -------------------------------------
    def _sections(self, doc: LoadedDocument) -> list[dict]:
        if not self.cfg.document_aware:
            text = "\n\n".join(b.text for b in doc.blocks)
            pages = [b.page for b in doc.blocks if b.page]
            return [{"path": [], "kind": "text", "texts": [text], "pages": pages, "extras": []}]
        sections: list[dict] = []
        for b in doc.blocks:
            last = sections[-1] if sections else None
            if last and last["path"] == b.heading_path and last["kind"] == b.kind:
                last["texts"].append(b.text)
                last["pages"].append(b.page)
                last["extras"].append(b.extra)
            else:
                sections.append({"path": b.heading_path, "kind": b.kind, "texts": [b.text],
                                 "pages": [b.page], "extras": [b.extra]})
        return sections

    # -- steps 2 + 3 ----------------------------------------------------------
    def _split_section(self, text: str) -> list[tuple[str, str]]:
        cfg = self.cfg
        if len(text) <= cfg.chunk_size:
            return [(text, "document" if cfg.document_aware else "recursive")]
        groups = [text]
        method = "recursive"
        if cfg.semantic and self.embedder is not None:
            groups = semantic_split(text, self.embedder, cfg.semantic_percentile, cfg.semantic_window,
                                    cfg.min_chunk_size)
            method = "semantic"
        out = []
        for g in groups:
            if len(g) <= cfg.chunk_size:
                out.append((g, method))
            else:
                out.extend((p, f"{method}+recursive" if method == "semantic" else "recursive")
                           for p in recursive_split(g, cfg.chunk_size, cfg.chunk_overlap))
        return out

    def _pack_rows(self, section: dict) -> list[dict]:
        """Spreadsheet rows: pack whole rows (never split a row), keep sheet + row range."""
        out, cur, start = [], [], None
        for text, extra in zip(section["texts"], section["extras"]):
            if cur and len("\n".join(cur)) + len(text) + 1 > self.cfg.chunk_size:
                out.append({"text": "\n".join(cur), "row_start": start, "row_end": prev_row, "method": "rows"})
                cur, start = [], None
            cur.append(text)
            start = extra.get("row") if start is None else start
            prev_row = extra.get("row")
        if cur:
            out.append({"text": "\n".join(cur), "row_start": start, "row_end": prev_row, "method": "rows"})
        for o in out:
            o["sheet"] = section["extras"][0].get("sheet")
            o["columns"] = section["extras"][0].get("columns")
        return out

    def chunk(self, doc: LoadedDocument) -> list[Chunk]:
        cfg = self.cfg
        pieces: list[dict] = []
        for sec in self._sections(doc):
            pages = [p for p in sec["pages"] if p]
            base = {"path": sec["path"], "kind": sec["kind"],
                    "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None}
            if sec["kind"] == "row":
                pieces.extend({**base, **r} for r in self._pack_rows(sec))
                continue
            for text, method in self._split_section("\n".join(sec["texts"])):
                pieces.append({**base, "text": text, "method": method})

        for p in pieces:                                        # final safety net before storing
            if p["kind"] != "row":
                p["text"] = _clean(p["text"])
        pieces = [p for p in pieces if p["text"] and (p["kind"] == "row" or not _is_junk(p["text"]))]
        pieces = self._merge_small(pieces)

        chunks = []
        for i, p in enumerate(pieces):
            # drop heading entries that just repeat the document title (common on slide decks)
            path_parts = [h for n, h in enumerate(p["path"])
                          if h.lower() not in doc.title.lower() and (n == 0 or h != p["path"][n - 1])]
            section = " > ".join(path_parts)
            header = ""
            if cfg.context_header:
                header = f"[{doc.title}{' > ' + section if section else ''}]"
                if p.get("columns"):
                    header += f"\nColumns: {p['columns']}"
                header += "\n"
            meta = {
                **doc.metadata,
                "doc_id": doc.doc_id, "source": Path(doc.source_path).name, "source_path": doc.source_path,
                "file_type": doc.file_type, "title": doc.title, "section": section,
                "heading": path_parts[-1] if path_parts else "", "pipeline_version": PIPELINE_VERSION,
                "also_covers": " | ".join(dict.fromkeys(x for x in p.get("extra_sections", [])
                                                         if x and x != section))[:500], "block_type": p["kind"],
                "page_start": p.get("page_start"), "page_end": p.get("page_end"),
                "sheet": p.get("sheet"), "row_start": p.get("row_start"), "row_end": p.get("row_end"),
                "chunk_index": i, "chunk_method": p["method"], "char_count": len(p["text"]),
            }
            chunks.append(Chunk(f"{doc.doc_id}:{i:05d}", header + p["text"], meta))
        return chunks

    def _merge_small(self, pieces: list[dict]) -> list[dict]:
        """Merge pieces below min_chunk_size into the previous piece of the same section."""
        merged: list[dict] = []
        for p in pieces:
            prev = merged[-1] if merged else None
            small = prev and (len(p["text"]) < self.cfg.min_chunk_size or len(prev["text"]) < self.cfg.min_chunk_size)
            fits = prev and len(prev["text"]) + len(p["text"]) + 1 <= self.cfg.chunk_size * 1.25
            if small and fits and prev["kind"] == p["kind"] != "row":
                if prev["path"] == p["path"]:
                    prev["text"] = f"{prev['text']}\n{p['text']}"
                elif self._adjacent_pages(prev, p):
                    # slides: a few lines per page -> join neighbouring slides, keeping each slide's title
                    title = p["path"][-1] if p["path"] else ""
                    prev["text"] = f"{prev['text']}\n\n{title + ':' + chr(10) if title else ''}{p['text']}"
                    prev.setdefault("extra_sections", []).append(" > ".join(p["path"]))
                else:
                    merged.append(dict(p))
                    continue
                prev["page_end"] = p.get("page_end") or prev.get("page_end")
                if p["method"] != prev["method"]:
                    prev["method"] = "merged"
            else:
                merged.append(dict(p))
        return merged

    @staticmethod
    def _adjacent_pages(a: dict, b: dict) -> bool:
        if not (a.get("page_end") and b.get("page_start")):
            return False
        same_parent = a["path"][:1] == b["path"][:1] or len(a["path"]) < 2 or len(b["path"]) < 2
        return 0 <= b["page_start"] - a["page_end"] <= 1 and same_parent


# =============================================================================
# Vector store (Chroma)
# =============================================================================
def _clean_metadata(meta: dict) -> dict:
    """Chroma accepts str/int/float/bool only: drop None, flatten lists, stringify dates."""
    out = {}
    for k, v in meta.items():
        if v is None or v == "":
            continue
        if isinstance(v, (list, tuple, set)):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, (date, datetime)):
            v = v.isoformat()
        elif hasattr(v, "value"):          # enums
            v = v.value
        elif isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False, default=str)
        elif not isinstance(v, (str, int, float, bool)):
            v = str(v)
        out[k] = v
    return out


def _to_where(filters: Optional[dict]) -> Optional[dict]:
    """{"department": "AI", "skill_level": ["Beginner", "Intermediate"]} -> Chroma where clause.
    Dicts already using operators ($and, $or, $eq, $in...) are passed through unchanged."""
    if not filters:
        return None
    if any(k.startswith("$") for k in filters):
        return filters
    clauses = []
    for k, v in filters.items():
        if isinstance(v, dict):
            clauses.append({k: v})
        elif isinstance(v, (list, tuple, set)):
            clauses.append({k: {"$in": list(v)}})
        else:
            clauses.append({k: {"$eq": v}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class RAGPipeline:
    def __init__(self, collection: str = "rex_kb", persist_dir: Optional[PathLike] = None,
                 embedder: Optional[Embedder] = None, chunking: Optional[ChunkingConfig] = None):
        import chromadb
        self.embedder = embedder or get_embedder()
        self.chunker = Chunker(chunking, self.embedder)
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder.name},
        )
        stored = (self.collection.metadata or {}).get("embedder")
        if stored and stored != self.embedder.name and self.collection.count():
            raise RuntimeError(
                f"Collection '{collection}' was built with embedder '{stored}' but the current one is "
                f"'{self.embedder.name}'. Vectors are not comparable: rebuild it with "
                f"`python -m rex_ai.core.rag reset --collection {collection}` or switch REX_RAG_EMBEDDINGS back.")

    # ---- ingestion ----------------------------------------------------------
    def ingest(self, paths: Union[PathLike, Iterable[PathLike]], metadata: Optional[dict] = None,
               recursive: bool = True, force: bool = False) -> dict:
        """Ingest files and/or folders. Returns a report {files, skipped, chunks, errors}."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files: list[Path] = []
        for p in map(Path, paths):
            if p.is_dir():
                pattern = "**/*" if recursive else "*"
                files.extend(f for f in sorted(p.glob(pattern)) if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            elif p.exists():
                files.append(p)
            else:
                raise FileNotFoundError(p)

        report = {"files": 0, "skipped": 0, "chunks": 0, "errors": []}
        for f in files:
            try:
                n = self.ingest_file(f, metadata=metadata, force=force)
                report["files" if n else "skipped"] += 1
                report["chunks"] += n
            except Exception as exc:
                log.error("Failed to ingest %s: %s", f, exc)
                report["errors"].append({"file": str(f), "error": str(exc)})
        return report

    def ingest_file(self, path: PathLike, metadata: Optional[dict] = None, force: bool = False) -> int:
        """Load -> chunk -> embed -> store one file. Returns the number of chunks added (0 = unchanged)."""
        path = Path(path)
        doc_id = _file_hash(path)
        current = {"$and": [{"doc_id": {"$eq": doc_id}}, {"pipeline_version": {"$eq": PIPELINE_VERSION}}]}
        if not force and self.collection.get(where=current, limit=1)["ids"]:
            log.info("skip %s (already ingested)", path.name)
            return 0
        self.delete(where={"source_path": str(path)})           # file changed -> replace old chunks
        self.delete(where={"doc_id": doc_id})

        meta = {"ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(metadata or {})}
        doc = load_document(path, meta)
        if not doc.blocks:
            log.warning("%s: no text extracted", path.name)
            return 0
        chunks = self.chunker.chunk(doc)
        self._add(chunks)
        log.info("ingested %s -> %d chunks", path.name, len(chunks))
        return len(chunks)

    def add_texts(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None,
                  ids: Optional[Sequence[str]] = None, chunk: bool = False) -> int:
        """Add already-prepared records (e.g. course/FAQ chunks from knowledge_base.py, scraped RFPs).
        chunk=True runs long texts through the semantic + recursive splitters first."""
        metadatas = list(metadatas or [{} for _ in texts])
        ids = list(ids or [hashlib.sha1(t.encode()).hexdigest()[:24] for t in texts])
        chunks: list[Chunk] = []
        for text, meta, cid in zip(texts, metadatas, ids):
            parts = [(text, "record")] if not chunk else self.chunker._split_section(text)
            for j, (part, method) in enumerate(parts):
                chunks.append(Chunk(cid if len(parts) == 1 else f"{cid}:{j:03d}", part,
                                    {"doc_id": cid, "chunk_method": method, **meta}))
        self._add(chunks)
        return len(chunks)

    def _add(self, chunks: list[Chunk], batch_size: int = 256):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            self.collection.upsert(ids=[c.chunk_id for c in batch], documents=[c.text for c in batch],
                                   embeddings=[v.tolist() for v in vectors],
                                   metadatas=[_clean_metadata(c.metadata) for c in batch])

    # ---- retrieval ----------------------------------------------------------
    def search(self, query: str, k: int = 5, where: Optional[dict] = None, contains: Optional[str] = None,
               min_score: float = 0.0, mmr: bool = False, fetch_k: int = 25, mmr_lambda: float = 0.6) -> list[Hit]:
        """Semantic search.
        where     metadata filter, e.g. {"department": "AI", "file_type": "pdf"} or a raw Chroma clause
        contains  keep only chunks whose text contains this string (exact keyword filter)
        mmr       re-rank with Maximal Marginal Relevance so the k hits are relevant AND not duplicates
        """
        if self.collection.count() == 0:
            return []
        q = self.embedder.embed_query(query)
        n = min(max(k, fetch_k if mmr else k), self.collection.count())
        res = self.collection.query(
            query_embeddings=[q.tolist()], n_results=n, where=_to_where(where),
            where_document={"$contains": contains} if contains else None,
            include=["documents", "metadatas", "distances"] + (["embeddings"] if mmr else []),
        )
        ids, docs, metas, dists = res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        hits = [Hit(i, d, m or {}, 1.0 - float(dist)) for i, d, m, dist in zip(ids, docs, metas, dists)]
        if mmr and hits:
            order = self._mmr(q, np.asarray(res["embeddings"][0], dtype=np.float32), k, mmr_lambda)
            hits = [hits[i] for i in order]
        return [h for h in hits if h.score >= min_score][:k]

    @staticmethod
    def _mmr(query: np.ndarray, emb: np.ndarray, k: int, lam: float) -> list[int]:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
        rel = emb @ (query / max(np.linalg.norm(query), 1e-12))
        chosen: list[int] = []
        candidates = list(range(len(emb)))
        while candidates and len(chosen) < k:
            if not chosen:
                best = int(np.argmax(rel))
            else:
                sim_to_chosen = (emb[candidates] @ emb[chosen].T).max(axis=1)
                scores = lam * rel[candidates] - (1 - lam) * sim_to_chosen
                best = candidates[int(np.argmax(scores))]
            chosen.append(best)
            candidates.remove(best)
        return chosen

    @staticmethod
    def build_context(hits: Sequence[Hit], max_chars: int = 6000) -> str:
        """Numbered context block for an LLM prompt. Ask the model to cite [n]."""
        parts, used = [], 0
        for n, h in enumerate(hits, start=1):
            block = f"[{n}] Source: {h.citation}\n{h.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    @staticmethod
    def to_source_refs(hits: Sequence[Hit]):
        """Hits -> SourceRef list (brief rule: every output traceable to its source data)."""
        from rex_ai.core.schemas import SourceRef
        now = datetime.now(timezone.utc)
        refs, seen = [], set()
        for h in hits:
            m = h.metadata
            record = m.get("source", h.chunk_id)
            if m.get("page_start"):
                record += f"#page={m['page_start']}"
            elif m.get("sheet"):
                record += f"#{m['sheet']}!{m.get('row_start')}"
            if record in seen:
                continue
            seen.add(record)
            refs.append(SourceRef(source_id=m.get("source_id", "SRC-RAG"), title=h.citation,
                                  url=m.get("url"), record_id=record, retrieved_at=now))
        return refs

    # ---- maintenance --------------------------------------------------------
    def delete(self, where: Optional[dict] = None, ids: Optional[list[str]] = None):
        if ids:
            self.collection.delete(ids=ids)
        elif where:
            existing = self.collection.get(where=_to_where(where), include=[])["ids"]
            if existing:
                self.collection.delete(ids=existing)

    def stats(self) -> dict:
        data = self.collection.get(include=["metadatas"])
        metas = data["metadatas"] or []
        by_doc: dict[str, int] = {}
        methods: dict[str, int] = {}
        for m in metas:
            by_doc[m.get("source", "?")] = by_doc.get(m.get("source", "?"), 0) + 1
            methods[m.get("chunk_method", "?")] = methods.get(m.get("chunk_method", "?"), 0) + 1
        return {"collection": self.collection.name, "embedder": self.embedder.name, "chunks": len(metas),
                "documents": len(by_doc), "by_document": by_doc, "chunk_methods": methods}

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=None, metadata={"hnsw:space": "cosine", "embedder": self.embedder.name})


@lru_cache(maxsize=None)
def get_rag(collection: str = "rex_kb") -> RAGPipeline:
    """Shared, cached pipeline per collection - call this from any agent.
    Suggested collections: academy_kb (courses/FAQs), curriculum_refs (syllabi, lecture notes,
    standards), rfp_docs (tender documents), templates (document-generation templates)."""
    return RAGPipeline(collection)


# =============================================================================
# CLI
# =============================================================================
def _main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(prog="python -m rex_ai.core.rag", description="REX AI shared RAG pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="ingest files or folders (pdf, docx, xlsx, csv, txt, md)")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--source-id", help="source_id from data/data_sources.yaml")
    pi.add_argument("--meta", action="append", default=[], help="extra metadata key=value (repeatable)")
    pi.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    pq = sub.add_parser("query", help="semantic search")
    pq.add_argument("text")
    pq.add_argument("-k", type=int, default=5)
    pq.add_argument("--where", help='JSON metadata filter, e.g. {"department": "AI"}')
    pq.add_argument("--mmr", action="store_true")
    sub.add_parser("stats")
    sub.add_parser("reset")
    for p in sub.choices.values():
        p.add_argument("--collection", default="rex_kb")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rag = RAGPipeline(args.collection)
    if args.cmd == "ingest":
        meta = dict(kv.split("=", 1) for kv in args.meta)
        if args.source_id:
            meta["source_id"] = args.source_id
        print(json.dumps(rag.ingest(args.paths, metadata=meta, force=args.force), indent=2))
    elif args.cmd == "query":
        for n, h in enumerate(rag.search(args.text, k=args.k, where=json.loads(args.where) if args.where else None,
                                         mmr=args.mmr), start=1):
            print(f"\n[{n}] score={h.score:.3f}  {h.citation}\n{h.text[:500]}")
    elif args.cmd == "stats":
        print(json.dumps(rag.stats(), indent=2, ensure_ascii=False))
    elif args.cmd == "reset":
        rag.reset()
        print(f"Collection '{args.collection}' cleared.")


if __name__ == "__main__":
    _main()