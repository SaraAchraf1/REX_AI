"""Approved templates for the Document Agent (brief, Agent 4: "generate structured content based on templates").

  data/doc_templates/templates.yaml     section order + headings (EN/AR) per document type, locked fact sections
  data/doc_templates/references/        reference example documents (brief §11), indexed in the shared RAG
                                        collection "doc_templates" with metadata doc_type, cited in every document
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import yaml

from rex_ai import config
from rex_ai.core.schemas import DocumentType, DocumentTypeInfo, SourceRef

log = logging.getLogger("rex.documents.templates")

TEMPLATES_COLLECTION = "doc_templates"
TEMPLATES_SOURCE_ID = "SRC-DOC-TEMPLATES"


@dataclass
class Section:
    key: str
    heading: dict[str, str]
    locked: bool = False

    def title(self, lang: str, **fmt) -> str:
        text = self.heading.get(lang) or self.heading["en"]
        return text.format(**fmt) if fmt else text


@dataclass
class Template:
    template_id: str
    doc_type: str
    title: dict[str, str]
    reader: dict[str, str]
    description: str
    sections: list[Section]
    default: bool = False

    def info(self, all_ids: list[str]) -> DocumentTypeInfo:
        return DocumentTypeInfo(doc_type=self.doc_type, template_id=self.template_id, title=self.title["en"],
                                title_ar=self.title.get("ar", self.title["en"]), reader=self.reader["en"],
                                description=self.description, sections=[s.key for s in self.sections],
                                templates=all_ids)


@lru_cache(maxsize=1)
def _load() -> tuple[dict, list[Template]]:
    path = config.DOC_TEMPLATES_DIR / "templates.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    templates = [Template(template_id=t["template_id"], doc_type=t["doc_type"], title=t["title"],
                          reader=t.get("reader") or {"en": ""}, description=t.get("description", ""),
                          sections=[Section(s["key"], s["heading"], bool(s.get("locked"))) for s in t["sections"]],
                          default=bool(t.get("default")))
                 for t in data["templates"]]
    unknown = {t.doc_type for t in templates} - {d.value for d in DocumentType}
    if unknown:
        raise ValueError(f"templates.yaml: unknown doc_type {sorted(unknown)}")
    return data.get("_meta", {}), templates


def meta() -> dict:
    return _load()[0]


def templates_for(doc_type: str) -> list[Template]:
    return [t for t in _load()[1] if t.doc_type == doc_type]


def get_template(doc_type: str, template_id: Optional[str] = None) -> Template:
    """The requested template, else the type's default. LookupError when the id does not exist for this type."""
    options = templates_for(doc_type)
    if not options:
        raise LookupError(f"No template for document type {doc_type}")
    if template_id:
        for t in options:
            if t.template_id.lower() == template_id.lower():
                return t
        raise LookupError(f"Template '{template_id}' not found for {doc_type}. Available: "
                          + ", ".join(t.template_id for t in options))
    return next((t for t in options if t.default), options[0])


def document_types() -> list[DocumentTypeInfo]:
    out = []
    for d in DocumentType:
        options = templates_for(d.value)
        if options:
            out.append(get_template(d.value).info([t.template_id for t in options]))
    return out


def template_source(t: Template) -> SourceRef:
    return SourceRef(source_id=meta().get("source_id", TEMPLATES_SOURCE_ID), title=f"Template {t.template_id}",
                     record_id=t.template_id)


# ---------------------------------------------------------------------------- reference examples (RAG)
def templates_rag():
    """The doc_templates collection (rebuilt if it was made with another embedder); None if RAG is unavailable."""
    try:
        from rex_ai.core.rag import get_rag
        try:
            return get_rag(TEMPLATES_COLLECTION)
        except RuntimeError as exc:
            if "built with embedder" not in str(exc):
                raise
            import chromadb
            log.warning("%s - rebuilding '%s' with the current embedder", exc, TEMPLATES_COLLECTION)
            chromadb.PersistentClient(path=str(config.CHROMA_DIR)).delete_collection(TEMPLATES_COLLECTION)
            _indexed.clear()
            return get_rag(TEMPLATES_COLLECTION)
    except Exception as exc:
        log.warning("RAG collection '%s' unavailable: %s", TEMPLATES_COLLECTION, exc)
        return None


_indexed: dict[str, tuple] = {}                  # folder -> signature of the files last indexed


def ensure_templates_indexed(rag=None, force: bool = False) -> dict:
    """Ingest references/ with the metadata from its manifest.yaml. Runs again only when a file was added or
    changed, so new academy examples are picked up without a restart."""
    folder = config.DOC_TEMPLATES_DIR / "references"
    report = {"files": 0, "skipped": 0, "chunks": 0, "errors": []}
    rag = rag or templates_rag()
    if rag is None or not folder.exists():
        return report
    from rex_ai.core.rag import SUPPORTED_EXTENSIONS
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS | {".yaml"})
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime) for p in files)
    if _indexed.get(str(folder)) == signature and not force:
        return report
    manifest_path = folder / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    defaults = manifest.get("defaults", {})
    for path in (p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS):
        info = (manifest.get("files") or {}).get(path.name) or {}
        meta_ = {**defaults, **info, "origin": "document_template"}
        meta_.setdefault("source_id", TEMPLATES_SOURCE_ID)
        meta_.setdefault("doc_type", "general")
        try:
            n = rag.ingest_file(path, metadata=meta_, force=force)
            report["files" if n else "skipped"] += 1
            report["chunks"] += n
        except Exception as exc:
            log.warning("Could not ingest template example %s: %s", path.name, exc)
            report["errors"].append({"file": path.name, "error": str(exc)})
    _indexed[str(folder)] = signature
    return report


def reference_examples(doc_type: str, query: str, k: int = 4):
    """(context for the LLM, source refs) from the reference examples of this document type."""
    rag = templates_rag()
    if rag is None:
        return "", []
    try:
        ensure_templates_indexed(rag)
        hits = rag.search(query, k=k, where={"doc_type": doc_type})
        return rag.build_context(hits, max_chars=2500), rag.to_source_refs(hits)
    except Exception as exc:
        log.warning("Template example search failed: %s", exc)
        return "", []
