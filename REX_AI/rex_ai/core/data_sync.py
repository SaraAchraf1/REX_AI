"""Keeps the REX Assistant on the latest academy data - shared by the API and the Streamlit app.

New data can arrive four ways, and all of it is ADDED to what REX already knows (nothing is deleted):
    data/catalog/     course catalogues (PDF, Word, Excel, CSV, TXT) -> merged into data/*.json
    data/documents/   any other PDF / Word / Excel / CSV / TXT (policies, brochures, handbooks...)
                      -> indexed so REX can answer questions from them
    POST /data/upload a file through the API (saved to one of the two folders above, never over an existing file)
    POST /data/courses course records as JSON (merged like a catalogue file)

get_assistant() checks the data folders on every call; when a file was added or changed it imports it and
rebuilds the assistant, so a running API or UI picks up new data without a restart.
"""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from rex_ai import config
from rex_ai.core import catalog_import as ci

log = logging.getLogger("rex.data")
_lock = threading.RLock()
_state: dict = {"signature": None, "assistant": None}


def catalog_dir() -> Path:
    return config.DATA_DIR / "catalog"


def documents_dir() -> Path:
    return config.DATA_DIR / "documents"


def _data_files() -> list[Path]:
    files = sorted(config.DATA_DIR.glob("*.json"))
    for folder in (catalog_dir(), documents_dir()):
        if folder.is_dir():
            files += ci.catalog_files(folder)
    return files


def signature() -> tuple:
    """Changes whenever a data file or an uploaded file is added, removed or modified."""
    out = [str(config.DATA_DIR)]
    for f in _data_files():
        st = f.stat()
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


# ------------------------------------------------------------------ importing
def ingest_documents() -> Optional[dict]:
    """Index every file in data/catalog/ and data/documents/ for REX's document answers.
    Unchanged files are skipped; a changed file replaces only its own passages. Never raises."""
    if config.RETRIEVER == "tfidf":
        return None
    files = _data_files()
    files = [f for f in files if f.suffix.lower() != ".json"]
    if not files:
        return None
    try:
        from rex_ai.agents.rex.retriever import RAGRetriever
        from rex_ai.core.rag import get_rag
        rag = get_rag(RAGRetriever.COLLECTION)
        report = rag.ingest(files, metadata={"origin": RAGRetriever.DOCS_ORIGIN, "source_id": "SRC-ACAD-DOCS"})
    except Exception as exc:
        log.error("Document indexing failed: %s", exc)
        return None
    if report["files"]:
        log.info("Indexed %d new/changed document(s) -> %d passages", report["files"], report["chunks"])
    return report


def sync(always: bool = False) -> dict:
    """Merge new catalogue files and index new documents. always=True ignores REX_AUTO_IMPORT=0."""
    return {"catalog": ci.auto_import(always=always),
            "documents": ingest_documents() if (always or config.AUTO_IMPORT) else None}


def get_assistant(always: bool = False):
    """The current RexAssistant, rebuilt when the data changed since it was created."""
    from rex_ai.agents.rex.assistant import RexAssistant
    from rex_ai.agents.rex.knowledge_base import get_kb
    from rex_ai.core import db
    with _lock:
        if _state["assistant"] is not None and not always and signature() == _state["signature"]:
            return _state["assistant"]
        _state["last_sync"] = sync(always)
        get_kb.cache_clear()
        db.init_db()                               # upserts the merged courses into the database
        _state["assistant"] = RexAssistant()
        _state["signature"] = signature()
        return _state["assistant"]


def reload() -> dict:
    """Import everything new now and rebuild the assistant (POST /data/reload)."""
    with _lock:
        get_assistant(always=True)
        return _state["last_sync"]


# ------------------------------------------------------------------- uploads
def _safe_name(filename: str) -> str:
    name = Path(filename or "upload").name
    stem = re.sub(r"[^\w\-. ]+", "_", Path(name).stem, flags=re.UNICODE).strip(" .") or "upload"
    return stem + Path(name).suffix.lower()


def _unique_path(folder: Path, name: str) -> Path:
    """data/catalog/x.pdf exists -> x-2.pdf, x-3.pdf ... (an upload never replaces an existing file)."""
    path, n = folder / name, 1
    while path.exists():
        n += 1
        path = folder / f"{Path(name).stem}-{n}{Path(name).suffix}"
    return path


def save_upload(filename: str, content: bytes) -> dict:
    """Save an uploaded file where it belongs and load it into REX.
    Course catalogues go to data/catalog/, everything else to data/documents/."""
    name = _safe_name(filename)
    ext = Path(name).suffix.lower()
    if ext not in ci.CATALOG_EXTENSIONS:
        raise ValueError(f"'{ext or name}' is not supported. Use one of: {', '.join(sorted(ci.CATALOG_EXTENSIONS))}")
    if not content:
        raise ValueError("the file is empty")

    digest = hashlib.sha256(content).hexdigest()
    for existing in _data_files():
        if existing.suffix.lower() != ".json" and hashlib.sha256(existing.read_bytes()).hexdigest() == digest:
            return {"status": "duplicate", "file": existing.relative_to(config.DATA_DIR).as_posix(),
                    "kind": "catalog" if existing.is_relative_to(catalog_dir()) else "document",
                    "courses": [], "errors": [], "warnings": []}

    with tempfile.TemporaryDirectory() as tmp:     # peek: does it contain course records?
        probe = Path(tmp) / name
        probe.write_bytes(content)
        parsed = ci.parse_file(probe)
    kind = "catalog" if parsed.courses else "document"
    folder = catalog_dir() if parsed.courses else documents_dir()
    folder.mkdir(parents=True, exist_ok=True)

    with _lock:
        path = _unique_path(folder, name)
        path.write_bytes(content)
        report = reload()
    merged = (report.get("catalog") or {}).get("merge") or {}
    return {"status": "saved", "file": path.relative_to(config.DATA_DIR).as_posix(), "kind": kind,
            "courses": [c["course_id"] for c in parsed.courses],
            "added": merged.get("added", []), "updated": merged.get("updated", []),
            "errors": parsed.errors if parsed.courses else [], "warnings": parsed.warnings}


def add_courses(records: list[dict], status: Optional[str] = None) -> dict:
    """Merge course records sent through the API, then rebuild the assistant."""
    with _lock:
        report = ci.import_records(records, "api", status)
        get_assistant(always=True)
    res, merged = report["files"][0], report["merge"] or {}
    return {"courses": [c["course_id"] for c in res.courses], "added": merged.get("added", []),
            "updated": merged.get("updated", []), "total_courses": merged.get("courses"),
            "errors": res.errors, "warnings": res.warnings}


def status() -> dict:
    from rex_ai.agents.rex.knowledge_base import get_kb
    files = [f for f in _data_files() if f.suffix.lower() != ".json"]
    kb = get_kb()
    return {"courses": len(kb.courses), "last_updated": kb.meta.get("last_updated"),
            "imported_from": kb.meta.get("imported_from", []),
            "catalog_files": [f.relative_to(config.DATA_DIR).as_posix() for f in files if f.is_relative_to(catalog_dir())],
            "documents": [f.relative_to(config.DATA_DIR).as_posix() for f in files if f.is_relative_to(documents_dir())]}
