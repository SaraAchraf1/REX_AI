"""Channel rules, objectives and audience personas for the Marketing Agent, plus its reference material in the shared
RAG (brief, Agent 5 items 37-38).

  data/marketing/channels.yaml        length limits, hashtags, CTA buttons, budget weights; objective -> CTA, KPIs
  data/marketing/audiences.yaml       personas: pains, motivations, preferred channels, tone, targeting
  data/marketing/references/          brand voice guide, channel playbook, approved example campaigns, indexed in
                                      the RAG collection "marketing_refs" (metadata kind, language) and cited
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import yaml

from rex_ai import config
from rex_ai.core.schemas import Channel, ChannelInfo, SourceRef

log = logging.getLogger("rex.marketing.rules")

REFS_COLLECTION = "marketing_refs"
GUIDE_SOURCE_ID = "SRC-MKT-GUIDE"
# Heading of each channel's section in the channel playbook (used to fetch that section from the RAG)
PLAYBOOK_HEADINGS = {"social_caption": "Social media caption", "facebook_instagram_ad": "Facebook / Instagram ad",
                     "linkedin_post": "LinkedIn post", "email": "Marketing e-mail",
                     "whatsapp": "WhatsApp promotional message"}


@lru_cache(maxsize=1)
def _channels_file() -> dict:
    return yaml.safe_load((config.MARKETING_DIR / "channels.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _audiences_file() -> dict:
    return yaml.safe_load((config.MARKETING_DIR / "audiences.yaml").read_text(encoding="utf-8"))


def channel(ch: str) -> dict:
    return _channels_file()["channels"][ch]


def channel_infos() -> list[ChannelInfo]:
    out = []
    for ch in Channel:
        c = channel(ch.value)
        out.append(ChannelInfo(channel=ch, name=c["name"]["en"], name_ar=c["name"]["ar"], role=c["role"]["en"],
                               headline_max=c["headline_max"], body_max=c["body_max"], hashtags=c["hashtags"],
                               paid=bool(c["paid"])))
    return out


def objective(name: Optional[str]) -> tuple[str, dict]:
    """(objective key, its settings). Unknown objectives use 'registrations' settings but keep their own name."""
    objectives = _channels_file()["objectives"]
    key = (name or "registrations").strip().lower()
    if key in objectives:
        return key, objectives[key]
    for k in objectives:                              # "webinar", "lead generation", "brand awareness"...
        if k.split()[0].rstrip("s") in key:
            return k, objectives[k]
    return key, objectives["registrations"]


def personas() -> list[dict]:
    return _audiences_file()["personas"]


def default_persona() -> dict:
    pid = _audiences_file().get("default_persona")
    return next((p for p in personas() if p["persona_id"] == pid), personas()[0])


def rule_sources() -> list[SourceRef]:
    return [SourceRef(source_id=_channels_file()["_meta"]["source_id"], title="Channel rules and objective defaults",
                      record_id="channels.yaml"),
            SourceRef(source_id=_audiences_file()["_meta"]["source_id"], title="Audience personas",
                      record_id="audiences.yaml")]


# ---------------------------------------------------------------------------- reference material (RAG)
def refs_rag():
    """The marketing_refs collection (rebuilt if it was made with another embedder); None if RAG is unavailable."""
    try:
        from rex_ai.core.rag import get_rag
        try:
            return get_rag(REFS_COLLECTION)
        except RuntimeError as exc:
            if "built with embedder" not in str(exc):
                raise
            import chromadb
            log.warning("%s - rebuilding '%s' with the current embedder", exc, REFS_COLLECTION)
            chromadb.PersistentClient(path=str(config.CHROMA_DIR)).delete_collection(REFS_COLLECTION)
            _indexed.clear()
            return get_rag(REFS_COLLECTION)
    except Exception as exc:
        log.warning("RAG collection '%s' unavailable: %s", REFS_COLLECTION, exc)
        return None


_indexed: dict[str, tuple] = {}                  # folder -> signature of the files last indexed


def ensure_refs_indexed(rag=None, force: bool = False) -> dict:
    """Ingest references/ with the metadata from its manifest.yaml; again only when a file was added or changed."""
    folder = config.MARKETING_DIR / "references"
    report = {"files": 0, "skipped": 0, "chunks": 0, "errors": []}
    rag = rag or refs_rag()
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
        meta = {**defaults, **info, "origin": "marketing_reference"}
        meta.setdefault("source_id", GUIDE_SOURCE_ID)
        meta.setdefault("kind", "general")
        try:
            n = rag.ingest_file(path, metadata=meta, force=force)
            report["files" if n else "skipped"] += 1
            report["chunks"] += n
        except Exception as exc:
            log.warning("Could not ingest marketing reference %s: %s", path.name, exc)
            report["errors"].append({"file": path.name, "error": str(exc)})
    _indexed[str(folder)] = signature
    return report


def guidance(channels: list[str], query: str, language: str, k: int = 3) -> tuple[str, list[SourceRef], dict]:
    """(LLM context, source refs, playbook text per channel). Brand rules + each channel's playbook section +
    example campaigns in the campaign's language. Empty when the RAG is unavailable."""
    rag = refs_rag()
    if rag is None:
        return "", [], {}
    try:
        ensure_refs_indexed(rag)
        hits = rag.search(f"brand voice claims facts {query}", k=2, where={"kind": "brand_guide"})
        tips: dict[str, str] = {}
        for ch in channels:
            heading = PLAYBOOK_HEADINGS.get(ch, ch)
            found = rag.search(f"{heading} best practices {query}", k=6, where={"kind": "channel_playbook"})
            # small sections may be merged into one chunk: the heading is then in also_covers
            best = next((h for h in found if heading.lower() in
                         f"{h.citation} {h.metadata.get('also_covers', '')} {h.text}".lower()), None)
            if best:
                hits.append(best)
                tips[ch] = best.text
        hits += rag.search(f"example campaign {query}", k=k, where={"$and": [{"kind": "example"},
                                                                             {"language": language}]})
        unique = list({h.chunk_id: h for h in hits}.values())
        return rag.build_context(unique, max_chars=5000), rag.to_source_refs(unique), tips
    except Exception as exc:
        log.warning("Marketing reference search failed: %s", exc)
        return "", [], {}
