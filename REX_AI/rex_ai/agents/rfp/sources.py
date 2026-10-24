"""Where tender notices come from (owner: Member 2).

    FeedSource      JSON / CSV / Excel listings in data/rfp/feeds/ (sample data, or exports a team member
                    downloads from Etimad, the Egyptian e-tenders portal, HRDF...). Always on.
    DocumentSource  RFP / RFQ / terms-of-reference files in data/rfp/documents/. Each file is ingested into
                    the shared RAG collection "rfp_docs" (every passage searchable and citable), then its fields
                    are extracted - by label rules, and by the LLM over RAG-retrieved passages when a key is set.
    WebPageSource   the public portals listed under `tenders:` in data/data_sources.yaml. Off unless the run asks
                    for live sources; robots.txt is checked first and only training-related links are kept.

Every source returns RawOpportunity objects; validation happens in the agent, not here.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import yaml

from rex_ai import config
from rex_ai.agents.rex.text_utils import normalize
from rex_ai.agents.rfp import llm
from rex_ai.agents.rfp.extract import RawOpportunity, classify, extract_fields

log = logging.getLogger("rex.rfp.sources")

FEED_SOURCE_ID = "SRC-RFP-SAMPLE"
DOCS_SOURCE_ID = "SRC-RFP-DOCS"
RAG_COLLECTION = "rfp_docs"
USER_AGENT = "REX-AI-RFP-Agent/0.2 (academy graduation project; contact: see repository README)"
FEED_EXTENSIONS = {".json", ".csv", ".xlsx", ".xls"}
DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def feeds_dir() -> Path:
    return config.RFP_DIR / "feeds"


def documents_dir() -> Path:
    return config.RFP_DIR / "documents"


def load_registry() -> dict[str, dict]:
    """source_id -> entry (plus its registry "category"), for every source in data_sources.yaml."""
    reg = yaml.safe_load((config.DATA_DIR / "data_sources.yaml").read_text(encoding="utf-8"))
    return {s["source_id"]: {**s, "category": cat} for cat, items in reg.items() for s in items}


class Source:
    source_id: str = ""

    def fetch(self) -> list[RawOpportunity]:
        raise NotImplementedError


# ====================================================================== feeds
# Column names a portal export may use -> RawOpportunity field
COLUMN_ALIASES = {
    "title": ["title", "tender title", "tender_title", "name", "subject", "اسم المنافسة", "عنوان المناقصة",
              "اسم المناقصة", "الموضوع"],
    "organization": ["organization", "organisation", "issuing organization", "issuing_organization", "issuer",
                     "buyer", "agency", "entity", "الجهة", "الجهة الحكومية", "الجهة المالكة"],
    "country": ["country", "الدولة"],
    "deadline": ["deadline", "closing date", "closing_date", "submission deadline", "due date", "آخر موعد",
                 "آخر موعد لتقديم العروض", "تاريخ الإغلاق"],
    "publication_date": ["publication date", "publication_date", "published", "published date", "تاريخ النشر",
                         "تاريخ الطرح"],
    "eligibility": ["eligibility", "eligibility requirements", "eligibility_requirements", "requirements",
                    "شروط التأهيل", "الشروط"],
    "category": ["category", "training category", "training_category", "activity", "التصنيف", "النشاط"],
    "submission_link": ["submission link", "submission_link", "link", "url", "رابط التقديم", "الرابط"],
    "documents": ["documents", "available documents", "available_documents", "attachments", "المستندات"],
    "description": ["description", "details", "scope", "summary", "الوصف", "التفاصيل", "نطاق العمل"],
    "source_id": ["source_id", "source"],
    "source_url": ["source_url", "notice url", "notice_url"],
}
_COLUMN_INDEX = {normalize(a): f for f, aliases in COLUMN_ALIASES.items() for a in aliases}


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v.strip() if isinstance(v, str) else v


def row_to_raw(row: dict, default_source: str, registry: dict, ref: str) -> Optional[RawOpportunity]:
    data: dict = {}
    for key, value in row.items():
        f = _COLUMN_INDEX.get(normalize(str(key)))
        if f and _clean(value) not in (None, ""):
            data[f] = _clean(value)
    if not data.get("title"):
        return None
    docs = data.pop("documents", [])
    if isinstance(docs, str):
        docs = [d.strip() for d in docs.replace("؛", ";").split(";") if d.strip()]
    source_id = data.pop("source_id", None)
    if source_id not in registry:                 # unknown ids fall back to the feed's own registered source
        source_id = default_source
    dates = {k: data.pop(k) for k in ("deadline", "publication_date") if k in data}   # parsed later, any format
    return RawOpportunity(source_id=source_id, documents=list(docs), origin="feed", raw_ref=ref,
                          **dates, **{k: str(v) for k, v in data.items()})


class FeedSource(Source):
    """Every JSON / CSV / Excel file in a folder. JSON: a list of objects, or {"_meta": {...}, "opportunities": [...]};
    _meta.source_id sets the default source for the file's rows."""

    def __init__(self, folder: Optional[Path] = None, registry: Optional[dict] = None):
        self.folder = Path(folder or feeds_dir())
        self.registry = registry if registry is not None else load_registry()
        self.source_id = FEED_SOURCE_ID

    def files(self) -> list[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(f for f in self.folder.rglob("*") if f.suffix.lower() in FEED_EXTENSIONS)

    def fetch(self) -> list[RawOpportunity]:
        out = []
        for path in self.files():
            default = FEED_SOURCE_ID
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    default = data.get("_meta", {}).get("source_id", FEED_SOURCE_ID)
                    data = data.get("opportunities", [])
                rows = data
            else:
                import pandas as pd
                df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
                rows = df.to_dict(orient="records")
            for i, row in enumerate(rows, start=1):
                if raw := row_to_raw(row, default, self.registry, f"{path.name}#{i}"):
                    out.append(raw)
        return out


# ====================================================================== documents (RAG)
class DocumentSource(Source):
    """One tender per document. The file is indexed in the RAG collection first, so the summary, the
    qualification check and POST /rfp/ask can cite its passages."""

    def __init__(self, folder: Optional[Path] = None):
        self.folder = Path(folder or documents_dir())
        self.source_id = DOCS_SOURCE_ID

    def files(self) -> list[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(f for f in self.folder.rglob("*") if f.suffix.lower() in DOC_EXTENSIONS)

    def fetch(self) -> list[RawOpportunity]:
        from rex_ai.core.rag import load_document
        rag = rag_or_none()
        out = []
        for path in self.files():
            if rag is not None:
                try:
                    rag.ingest_file(path, metadata={"origin": "rfp_document", "source_id": DOCS_SOURCE_ID})
                except Exception as exc:
                    log.warning("Could not index %s in RAG: %s", path.name, exc)
            fields = extract_fields(document_text(load_document(path).blocks))
            if rag is not None and llm.enabled():
                fields = {**llm.extract_tender_fields(rag, path, fields), **fields}
            if not fields.get("title"):
                fields["title"] = path.stem.replace("_", " ")
            docs = fields.pop("documents", [])
            out.append(RawOpportunity(source_id=DOCS_SOURCE_ID, origin="document", raw_ref=path.name,
                                      documents=[path.name, *docs],
                                      **{k: v for k, v in fields.items() if k in RawOpportunity.__dataclass_fields__}))
        return out


def document_text(blocks) -> str:
    """Loader blocks back to text, with each new heading as a '# ' line (loaders keep headings apart)."""
    lines, current = [], None
    for b in blocks:
        if b.heading_path and b.heading_path != current:
            lines.append("# " + b.heading_path[-1])
            current = b.heading_path
        lines.append(b.text)
    return "\n".join(lines)


def rag_or_none():
    """The rfp_docs collection, or None if the RAG stack can't start (the agent still works without it)."""
    try:
        from rex_ai.core.rag import get_rag
        return get_rag(RAG_COLLECTION)
    except Exception as exc:
        log.warning("RAG collection '%s' unavailable: %s", RAG_COLLECTION, exc)
        return None


# ====================================================================== live portals
class WebPageSource(Source):
    """Generic listing-page reader: keeps links whose text is training-related, optionally opens a few of them
    to read 'Deadline: ...' style fields. Portal layouts change, so nothing here depends on CSS classes."""

    def __init__(self, entry: dict, max_items: int = 30, follow_details: int = 5, session=None):
        import requests
        self.entry = entry
        self.source_id = entry["source_id"]
        self.url = entry.get("url")
        self.max_items = max_items
        self.follow_details = follow_details
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})
        country = entry.get("country")
        self.country = country if isinstance(country, str) and country in ("EG", "SA") else None

    def allowed(self, url: str) -> bool:
        root = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        try:
            r = self.http.get(root + "/robots.txt", timeout=config.RFP_HTTP_TIMEOUT)
        except Exception:
            return True                                      # unreachable robots.txt = no restrictions stated
        if r.status_code >= 400:
            return True
        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(USER_AGENT, url)

    def _get(self, url: str) -> str:
        r = self.http.get(url, timeout=config.RFP_HTTP_TIMEOUT)
        r.raise_for_status()
        return r.text

    def fetch(self) -> list[RawOpportunity]:
        from bs4 import BeautifulSoup
        if not self.url:
            return []
        if not self.allowed(self.url):
            raise PermissionError(f"robots.txt disallows {self.url}")
        soup = BeautifulSoup(self._get(self.url), "html.parser")
        out, seen = [], set()
        for a in soup.find_all("a", href=True):
            title = " ".join(a.get_text(" ").split())
            link = urljoin(self.url, a["href"])
            if len(title) < 15 or link in seen:
                continue
            context = " ".join((a.find_parent(["tr", "li", "article", "div"]) or a).get_text(" ").split())[:1500]
            raw = RawOpportunity(title=title, source_id=self.source_id, country=self.country, description=context,
                                 submission_link=link, source_url=self.url, origin="web", raw_ref=link)
            if not classify(raw).relevant:
                continue
            seen.add(link)
            out.append(raw)
            if len(out) >= self.max_items:
                break
        for raw in out[: self.follow_details]:
            try:
                if self.allowed(raw.submission_link):
                    page = BeautifulSoup(self._get(raw.submission_link), "html.parser").get_text("\n")
                    for k, v in extract_fields(page).items():
                        if k in ("deadline", "publication_date", "organization", "eligibility", "country") and v:
                            setattr(raw, k, v)
            except Exception as exc:
                log.info("detail page %s skipped: %s", raw.submission_link, exc)
        return out


def build_sources(source_ids: Optional[Iterable[str]] = None, live: bool = False,
                  registry: Optional[dict] = None) -> list[Source]:
    registry = registry if registry is not None else load_registry()
    wanted = set(source_ids) if source_ids else None
    sources: list[Source] = [FeedSource(registry=registry), DocumentSource()]
    if live:
        for sid, entry in registry.items():
            if (entry["category"] == "tenders" and entry.get("access") == "portal_scrape" and entry.get("url")
                    and (not wanted or sid in wanted)):
                sources.append(WebPageSource(entry))
    return sources
