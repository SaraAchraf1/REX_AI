"""Curriculum research (brief, Agent 3 FR 18-19): what should this course teach, and according to whom?

Four kinds of evidence, all cited in the curriculum's sources:
  1. academy catalogue    related courses (same topic / department): modules, outcomes, prerequisites to reuse
  2. reference syllabi    shared RAG collection "curriculum_refs" (data/curriculum/references/ + MIT OCW notes),
                          filtered by the metadata in references/manifest.yaml (department, level, framework...)
  3. industry skills      data/curriculum/industry_skills.yaml: roles and skills employers ask for
  4. framework            the selected (or suggested) accreditation / competency framework

    bundle = research(filled_request, kb)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import yaml

from rex_ai import config
from rex_ai.agents.curriculum import frameworks as fw
from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.core.schemas import Course, CurriculumResearch, SourceRef

log = logging.getLogger("rex.curriculum.research")

REFS_COLLECTION = "curriculum_refs"
REFS_SOURCE_ID = "SRC-CUR-REFS"
INDUSTRY_SOURCE_ID = "SRC-CUR-INDUSTRY"
CATALOGUE_SOURCE_ID = "SRC-ACAD-COURSES"
MODULE_PREFIX = re.compile(r"^(module|unit|الوحدة|وحدة)\s*\d*\s*[:\-–]\s*", re.I)
OUTCOME_PREFIX = re.compile(r"^(outcome|النتيجة)\s*[:\-–]\s*", re.I)
STOPWORDS = {"and", "for", "with", "the", "to", "of", "in", "on", "a", "an", "course", "training", "program",
             "programme", "introduction", "fundamentals", "essentials", "basics", "professional", "advanced",
             "beginner", "intermediate", "from", "using", "your", "دوره", "مقدمه", "في", "الي", "من", "و"}


@dataclass
class ReferenceSection:
    title: str
    topics: list[str]
    outcome: Optional[str]
    citation: str
    source_id: str
    score: float
    doc: str = ""                                # reference document title
    order: int = 0                               # position in that document


@dataclass
class ResearchBundle:
    """Everything the generator needs; .summary is the public part (API / UI / stored metadata)."""
    summary: CurriculumResearch
    department: Optional[str]
    framework: Optional[fw.Framework]
    courses: list[Course] = field(default_factory=list)
    sections: list[ReferenceSection] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    context: str = ""                            # numbered RAG passages for the LLM prompt


# ---------------------------------------------------------------------------- industry skills
@lru_cache(maxsize=1)
def load_industry() -> dict:
    path = config.CURRICULUM_DIR / "industry_skills.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {"departments": {}}


def infer_department(title: str, text: str = "", kb=None) -> Optional[str]:
    """Department from the course title (keywords, English + Arabic), else from the closest catalogue course."""
    best, best_score = None, 0
    for dept, info in load_industry().get("departments", {}).items():
        kws = info.get("keywords", [])
        score = sum(3 for k in kws if contains_any(title, [k])) + sum(1 for k in kws if contains_any(text, [k]))
        if score > best_score:
            best, best_score = dept, score
    if best:
        return best
    if kb is not None and (courses := related_courses(kb, title, None)):
        return courses[0].department
    return None


# ---------------------------------------------------------------------------- catalogue
def _words(text: str) -> set[str]:
    return {w for w in normalize(text).split() if len(w) > 2 and w not in STOPWORDS}


def related_courses(kb, title: str, department: Optional[str], course_id: Optional[str] = None,
                    limit: int = 3) -> list[Course]:
    """Catalogue courses on the same topic: the requested course first, then title/tag overlap."""
    words = _words(title)
    scored: list[tuple[float, Course]] = []
    for c in kb.courses.values():
        if c.course_id == course_id:
            scored.append((99.0, c))
            continue
        cw = _words(f"{c.title_en} {c.title_ar}") | _words(" ".join(c.tags))
        overlap = len(words & cw) / max(len(words), 1)
        if overlap >= 0.34 or (overlap > 0 and c.department == department):
            scored.append((overlap + (0.2 if c.department == department else 0), c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


def prerequisite_course(kb, department: Optional[str], level: str, title: str = "") -> Optional[Course]:
    """The closest catalogue course one level below, in the same department (for 'complete X or equivalent')."""
    below = {"Intermediate": "Beginner", "Advanced": "Intermediate"}.get(level)
    if not (below and department):
        return None
    words = _words(title)
    scored = [(len(words & (_words(f"{c.title_en} {c.title_ar}") | _words(" ".join(c.tags)))), c)
              for c in kb.courses.values() if c.department == department and c.skill_level.value == below]
    best = max(scored, key=lambda x: x[0], default=(0, None))
    return best[1] if best[0] else None          # an unrelated course is no prerequisite


# ---------------------------------------------------------------------------- reference syllabi (RAG)
def refs_rag():
    """The curriculum_refs collection. If it was built with another embedder (e.g. the offline hash one before the
    real model was installed), its vectors are useless: it is rebuilt with the current embedder."""
    try:
        from rex_ai.core.rag import get_rag
        try:
            return get_rag(REFS_COLLECTION)
        except RuntimeError as exc:
            if "built with embedder" not in str(exc):
                raise
            import chromadb
            log.warning("%s - rebuilding '%s' with the current embedder. Re-run scripts/ingest_mit_ocw.py to add the "
                        "MIT lecture notes again.", exc, REFS_COLLECTION)
            chromadb.PersistentClient(path=str(config.CHROMA_DIR)).delete_collection(REFS_COLLECTION)
            _indexed.clear()
            return get_rag(REFS_COLLECTION)
    except Exception as exc:
        log.warning("RAG collection '%s' unavailable: %s", REFS_COLLECTION, exc)
        return None


_indexed: dict[str, tuple] = {}                  # folder -> (file name, size, mtime) signature last indexed


def ensure_references_indexed(rag=None, force: bool = False) -> dict:
    """Ingest data/curriculum/references/ with the metadata from manifest.yaml. Runs again only when a file in the
    folder was added or changed (new references are picked up without restarting); unchanged files are also
    skipped by the RAG pipeline itself."""
    folder = config.CURRICULUM_DIR / "references"
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
        meta = {**defaults, **(manifest.get("files", {}).get(path.name) or {}), "origin": "curriculum_reference"}
        meta.setdefault("source_id", REFS_SOURCE_ID)
        try:
            n = rag.ingest_file(path, metadata=meta, force=force)
            report["files" if n else "skipped"] += 1
            report["chunks"] += n
        except Exception as exc:
            log.warning("Could not ingest reference %s: %s", path.name, exc)
            report["errors"].append({"file": path.name, "error": str(exc)})
    _indexed[str(folder)] = signature
    return report


def parse_section(hit) -> Optional[ReferenceSection]:
    """A reference chunk shaped like '## Module: X' + '- topic' bullets + 'Outcome: ...' -> module candidate."""
    lines = [ln.strip() for ln in hit.text.splitlines() if ln.strip() and not ln.startswith("[")]
    topics = [ln[2:].strip() for ln in lines if ln.startswith(("- ", "* ", "• "))]
    if len(topics) < 2:
        return None
    heading = hit.metadata.get("heading") or ""
    title = MODULE_PREFIX.sub("", heading).strip()
    if not title or normalize(title) in {"practical project ideas", "project ideas"}:
        return None
    outcome = next((OUTCOME_PREFIX.sub("", ln) for ln in lines if OUTCOME_PREFIX.match(ln)), None)
    return ReferenceSection(title, topics, outcome, hit.citation, hit.metadata.get("source_id", REFS_SOURCE_ID),
                            round(hit.score, 3), doc=hit.metadata.get("title", ""),
                            order=int(hit.metadata.get("chunk_index") or 0))


def search_references(rag, query: str, department: Optional[str], title: str, k: int = 8):
    """Department-filtered search (metadata filtering) plus an unfiltered search, so a cross-domain course
    ("Data Analysis for Healthcare Quality") finds references from both departments. Unfiltered hits must share a
    word with the course title: nearest neighbours always exist, relevant ones do not."""
    hits = rag.search(query, k=k, where={"department": department}, mmr=True) if department else []
    seen = {h.chunk_id for h in hits}
    words = _words(title)
    hits += [h for h in rag.search(query, k=k, mmr=True, min_score=0.25)
             if h.chunk_id not in seen and words & _words(f"{h.metadata.get('title', '')} {h.text}")]
    return hits[: k + 6]


# ---------------------------------------------------------------------------- research
def research(req: dict, kb, use_rag: bool = True) -> ResearchBundle:
    """req: the request with the catalogue fields already filled in (agent._fill_request)."""
    title = req["course_title"]
    text = " ".join([req.get("target_audience") or "", req.get("industry") or "", *req.get("expected_outcomes", [])])
    notes: list[str] = []
    sources: list[SourceRef] = []

    department = req.get("department") or infer_department(title, text, kb)
    if not req.get("department"):
        notes.append(f"Department inferred from the title: {department}" if department
                     else "Department could not be inferred; give one for better results")

    courses = related_courses(kb, title, department, req.get("course_id"))
    for c in courses:
        sources.append(SourceRef(source_id=c.source_id or CATALOGUE_SOURCE_ID, title=f"Academy course {c.title_en}",
                                 record_id=c.course_id))

    if req.get("accreditation_framework"):
        framework = fw.find_framework(req["accreditation_framework"])
        reason = "selected by the user"
        if framework is None:
            notes.append(f"Framework '{req['accreditation_framework']}' is not in data/curriculum/frameworks.yaml; "
                         "no alignment possible until it is added")
    else:
        framework, reason = fw.suggest_framework(title, department, req["country"], text)
    if framework:
        sources.append(framework.source_ref())

    industry = load_industry()
    info = industry.get("departments", {}).get(department or "", {})
    if info:
        sources.append(SourceRef(source_id=industry.get("_meta", {}).get("source_id", INDUSTRY_SOURCE_ID),
                                 title=f"Industry skill requirements - {department}", record_id=department))

    sections, context = [], ""
    rag = refs_rag() if use_rag else None
    if rag is not None:
        try:
            ensure_references_indexed(rag)
            query = f"{title}. {req.get('target_audience') or ''}. {' '.join(req.get('expected_outcomes', []))}"
            hits = search_references(rag, query, department, title)
            sections = [s for h in hits if (s := parse_section(h))]
            context = rag.build_context(hits, max_chars=5000)
            sources += rag.to_source_refs(hits)
            if not hits:
                notes.append("No reference syllabus found for this topic - add one to data/curriculum/references/")
        except Exception as exc:
            log.warning("Reference search failed: %s", exc)
            notes.append("Reference search unavailable; curriculum built from catalogue, skills and framework only")

    summary = CurriculumResearch(
        course_title=title, department=department, framework_id=framework.framework_id if framework else None,
        framework_reason=reason if framework else None, related_courses=[c.course_id for c in courses],
        industry_roles=info.get("roles", []), industry_skills=info.get("core_skills", []),
        reference_modules=[{"title": s.title, "topics": s.topics, "citation": s.citation} for s in sections],
        notes=notes, sources=_dedupe_sources(sources))
    return ResearchBundle(summary=summary, department=department, framework=framework, courses=courses,
                          sections=sections, skills=info.get("core_skills", []),
                          soft_skills=info.get("soft_skills", []), roles=info.get("roles", []),
                          industries=info.get("industry", []), context=context)


def _dedupe_sources(refs: list[SourceRef]) -> list[SourceRef]:
    seen, out = set(), []
    for r in refs:
        key = (r.source_id, r.record_id)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
