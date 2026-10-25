"""Accreditation / competency alignment and gap analysis (brief, Agent 3 FR 22-23).

Each competency of the selected framework is compared with the curriculum:
    topic      covered  = a module's title, topics or exercises mention one of its keywords (exact evidence)
               partial  = no keyword, but a module is semantically close (shared RAG embedder) -> reviewer confirms
               missing  = neither
    outcome    the same, against the learning outcomes
    structure  a design rule (outcomes measurable, weights total 100 %, every outcome domain present...)

The report lists covered / partial / missing items with evidence, missing topics and outcomes, a coverage %
and concrete recommendations. It is an ALIGNMENT report: GapReport.disclaimer always says it is not evidence of
accreditation, and nothing here ever states that a course is accredited.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from rex_ai import config
from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.core.schemas import GapItem, GapReport, SourceRef

log = logging.getLogger("rex.curriculum.alignment")

DOMAIN_VERBS = {
    "knowledge": ["explain", "describe", "define", "identify", "list", "recognise", "recognize", "outline",
                  "summarise", "summarize", "state", "recall", "يشرح", "يصف", "يحدد", "يعرف", "يذكر", "شرح", "وصف",
                  "تحديد"],
    "intellectual": ["analyse", "analyze", "compare", "evaluate", "interpret", "assess", "critique", "diagnose",
                     "distinguish", "justify", "solve", "select", "choose", "يحلل", "يقارن", "يقيم", "يفسر", "يختار",
                     "تحليل", "مقارنة", "تقييم"],
    "practical": ["apply", "use", "build", "implement", "configure", "develop", "deploy", "create", "write",
                  "perform", "operate", "automate", "run", "prepare", "produce", "design", "plan", "triage",
                  "clean", "train", "package", "monitor", "conduct", "test", "يطبق", "يستخدم", "يبني", "ينفذ",
                  "يطور", "يصمم", "يعد", "ينشئ", "تطبيق", "إعداد", "اعداد"],
    "transferable": ["communicate", "present", "collaborate", "lead", "document", "reflect", "negotiate",
                     "facilitate", "يعرض", "يتواصل", "يعمل", "يقود", "يتعاون"],
}
# Transferable / values cues anywhere in the outcome (e.g. "Work responsibly and ethically ...")
TRANSFERABLE_CUES = ["ethic", "responsib", "team", "stakeholder", "communicat", "present", "professional conduct",
                     "أخلاق", "بمسؤولية", "مسؤولية", "فريق", "يعرض"]
NON_MEASURABLE = ["understand", "know", "learn", "be familiar", "be aware", "appreciate", "يفهم", "يعرف عن", "يتعرف"]
_LEAD_IN = re.compile(r"^(?:(?:learners|participants|students|trainees)\s+)?(?:(?:will|should)\s+)?(?:be able to\s+)?"
                      r"|^أن\s+", re.I)
DOMAIN_LABELS = {"knowledge": "knowledge and understanding", "intellectual": "intellectual / cognitive skills",
                 "practical": "professional and practical skills", "transferable": "general, transferable and values"}
DOMAIN_EXAMPLES = {"knowledge": "Explain the key concepts of ...", "intellectual": "Analyse and compare ...",
                   "practical": "Apply ... to a realistic task", "transferable": "Work responsibly and present results ..."}
ASSESSES = {"knowledge": {"quiz", "exam"}, "intellectual": {"assignment", "exam", "project"},
            "practical": {"assignment", "project", "lab"}, "transferable": {"project", "presentation"}}


# ============================================================================ learning outcomes
def _lead_words(outcome: str, n: int = 3) -> list[str]:
    text = re.sub(r"^أن\s+", "", outcome.strip())
    return normalize(_LEAD_IN.sub("", text, count=1)).split()[:n]


def outcome_domain(outcome: str) -> set[str]:
    words = _lead_words(outcome)
    norm_verbs = {d: {normalize(v) for v in vs} for d, vs in DOMAIN_VERBS.items()}
    found = {d for d, vs in norm_verbs.items() if words and words[0] in vs}
    if not found:                                   # "Clean and transform ...", "يحلل ويقارن ..."
        found = {d for d, vs in norm_verbs.items() if any(w in vs for w in words[1:])}
    if any(c in normalize(outcome) for c in map(normalize, TRANSFERABLE_CUES)):
        found.add("transferable")
    return found


def outcome_domains(outcomes: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {d: [] for d in DOMAIN_VERBS}
    for o in outcomes:
        for d in outcome_domain(o):
            out[d].append(o)
    return out


def is_measurable(outcome: str) -> bool:
    """Starts with an observable action verb, not 'understand' / 'know' / 'learn about'."""
    lead = " ".join(_lead_words(outcome))
    if not lead or any(lead.startswith(normalize(x)) for x in NON_MEASURABLE):
        return False
    return bool(outcome_domain(outcome))


# ============================================================================ structure checks
def _modules(cur: dict) -> list[dict]:
    return cur.get("modules") or []


def _hours_consistent(cur, req):
    total = float(cur.get("training_hours") or 0)
    s = sum(float(m.get("hours") or 0) for m in _modules(cur))
    ok = total > 0 and abs(s - total) < 0.01 and all(float(m.get("hours") or 0) > 0 for m in _modules(cur))
    return ok, f"modules total {s:g} h, training hours {total:g} h"


def _assessments_complete(cur, req):
    a = cur.get("assessments") or []
    s = sum(float(x.get("weight_pct") or 0) for x in a)
    return bool(a) and abs(s - 100) < 0.01, f"{len(a)} assessments, weights total {s:g}%"


def _outcomes_measurable(cur, req):
    outs = cur.get("learning_outcomes") or []
    weak = [o for o in outs if not is_measurable(o)]
    return bool(outs) and len(weak) <= len(outs) * 0.2, \
        ("all outcomes start with an action verb" if not weak else "not measurable: " + "; ".join(weak[:3]))


def _requested_outcomes(cur, req):
    from rex_ai.agents.curriculum.generator import _same
    missing = [o for o in (req or {}).get("expected_outcomes", [])
               if not any(_same(o, x) for x in cur.get("learning_outcomes") or [])]
    return not missing, ("all requested outcomes included" if not missing else "missing: " + "; ".join(missing))


def _outcomes_assessed(cur, req):
    types = {normalize(a.get("type", "")) for a in cur.get("assessments") or []}
    if cur.get("capstone_project"):
        types.add("project")
    unassessed = [o for o in cur.get("learning_outcomes") or []
                  if (doms := outcome_domain(o)) and not any(ASSESSES[d] & types for d in doms)]
    return bool(types) and not unassessed, \
        (f"assessment types: {', '.join(sorted(types))}" if not unassessed else "not assessed: " + "; ".join(unassessed[:3]))


def _every_module_practical(cur, req):
    teaching = [m for m in _modules(cur) if m.get("exercises") is not None]
    lacking = [m["title"] for m in teaching if not m.get("exercises")]
    return bool(teaching) and not lacking, ("every module has an exercise" if not lacking
                                            else "no exercise in: " + ", ".join(lacking))


def _outcome_domain_check(domain: str) -> Callable:
    def check(cur, req):
        hits = outcome_domains(cur.get("learning_outcomes") or [])[domain]
        return bool(hits), (f"e.g. '{hits[0]}'" if hits else f"no {DOMAIN_LABELS[domain]} outcome")
    return check


STRUCTURE_CHECKS: dict[str, Callable[[dict, dict], tuple[bool, str]]] = {
    "has_objectives": lambda c, r: (len(c.get("objectives") or []) > 0, f"{len(c.get('objectives') or [])} objectives"),
    "has_audience": lambda c, r: (bool((c.get("target_audience") or "").strip()), c.get("target_audience") or "not stated"),
    "has_prerequisites": lambda c, r: (bool(c.get("prerequisites")), "; ".join(c.get("prerequisites") or []) or "none stated"),
    "has_capstone": lambda c, r: (len((c.get("capstone_project") or "").strip()) > 20, (c.get("capstone_project") or "none")[:120]),
    "has_practical_work": lambda c, r: (bool(c.get("practical_exercises")) or any(m.get("exercises") for m in _modules(c)),
                                        f"{len(c.get('practical_exercises') or [])} practical exercises"),
    "hours_consistent": _hours_consistent,
    "assessments_complete": _assessments_complete,
    "outcomes_measurable": _outcomes_measurable,
    "requested_outcomes_covered": _requested_outcomes,
    "outcomes_assessed": _outcomes_assessed,
    "every_module_practical": _every_module_practical,
    **{f"outcome_domain:{d}": _outcome_domain_check(d) for d in DOMAIN_VERBS},
}

STRUCTURE_ADVICE = {
    "has_objectives": "State 3-5 course objectives.",
    "has_audience": "Describe the target audience.",
    "has_prerequisites": "List the prerequisites (knowledge, experience or a previous course).",
    "has_capstone": "Define a capstone project with a deliverable and assessment criteria.",
    "has_practical_work": "Add practical exercises (labs, case studies, role plays).",
    "hours_consistent": "Make the module hours add up to the training hours.",
    "assessments_complete": "Add assessments whose weights total 100%.",
    "outcomes_measurable": "Rewrite outcomes with measurable action verbs (explain, analyse, build...) instead of "
                           "'understand' or 'know'.",
    "requested_outcomes_covered": "Add the outcomes the course owner asked for.",
    "outcomes_assessed": "Make sure every outcome is assessed (quizzes for knowledge, labs/projects for skills, "
                         "presentation for transferable skills).",
    "every_module_practical": "Give every module at least one practical exercise.",
}


# ============================================================================ topic / outcome matching
def keyword_hit(comp, text: str) -> Optional[str]:
    for k in [comp.name, *comp.keywords]:
        if contains_any(text, [k]):
            return k
    return None


def semantic_threshold(embedder_name: str) -> tuple[float, float]:
    """(minimum similarity, minimum margin over the competency's average similarity) for a semantic-only match.
    Measured on multilingual-e5-small: every pair scores 0.79-0.87, related and unrelated modules overlap below
    ~0.85, so only a clearly outstanding score counts; lexical evidence (shared_terms) does most of the work."""
    if config.CURRICULUM_SEMANTIC_MIN:
        return float(config.CURRICULUM_SEMANTIC_MIN), 0.0
    if embedder_name.startswith("hash"):
        return 0.45, 0.0
    if "e5" in embedder_name:
        return 0.85, 0.045
    return 0.55, 0.05


_BASIC_STOP = {"and", "or", "the", "of", "to", "in", "on", "for", "with", "a", "an", "by", "from", "its", "into",
               "و", "في", "من", "على", "الى", "عن", "مع"}


def _stem(word: str) -> str:
    if word.isascii() and len(word) > 5:
        for suffix in ("ing", "es", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    return word


def terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"\w+", normalize(text)) if len(w) > 2 and w not in _BASIC_STOP}


def generic_terms(comps) -> set[str]:
    """Words shared by much of the framework ('test' in ISTQB): they say nothing about which competency is meant."""
    counts: dict[str, int] = {}
    for c in comps:
        for t in terms(" ".join([c.name, *c.keywords])):
            counts[t] = counts.get(t, 0) + 1
    return {t for t, n in counts.items() if n >= max(2, len(comps) / 2)}


class Aligner:
    def __init__(self, embedder=None, use_embeddings: bool = True):
        self._embedder = embedder
        self.use_embeddings = use_embeddings

    @property
    def embedder(self):
        if self._embedder is None and self.use_embeddings:
            try:
                from rex_ai.core.rag import get_embedder
                self._embedder = get_embedder()
            except Exception as exc:
                log.warning("No embedder for semantic alignment, keywords only: %s", exc)
                self.use_embeddings = False
        return self._embedder

    def _similarity(self, queries: list[str], units: list[str]) -> Optional[np.ndarray]:
        emb = self.embedder
        if emb is None or not queries or not units:
            return None
        try:
            q = np.asarray([emb.embed_query(x) for x in queries], dtype=np.float32)
            u = np.asarray(emb.embed_documents(units), dtype=np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
            u /= np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-12)
            return q @ u.T
        except Exception as exc:
            log.warning("Semantic alignment failed, keywords only: %s", exc)
            return None

    def compare(self, cur: dict, framework, request: Optional[dict] = None,
                department: Optional[str] = None) -> GapReport:
        request = request or {}
        comps = framework.competencies_for(department)
        modules = [m for m in _modules(cur)]
        module_units = [f"{m['title']}: {'; '.join(m.get('topics') or [])}. {' '.join(m.get('exercises') or [])}"
                        for m in modules]
        outcome_units = list(cur.get("learning_outcomes") or []) + ([cur["capstone_project"]]
                                                                     if cur.get("capstone_project") else [])
        matchable = [c for c in comps if c.kind in ("topic", "outcome")]
        sims_topic = self._similarity([f"{c.name}. {c.description}" for c in matchable], module_units)
        sims_outcome = self._similarity([f"{c.name}. {c.description}" for c in matchable], outcome_units)
        threshold, margin = semantic_threshold(getattr(self.embedder, "name", "none"))
        generic = generic_terms(matchable)

        items: list[GapItem] = []
        recs: list[str] = []
        for c in comps:
            if c.kind == "structure":
                fn = STRUCTURE_CHECKS.get(c.check or "")
                if fn is None:
                    log.warning("Unknown structure check %s in %s", c.check, framework.framework_id)
                    continue
                ok, evidence = fn(cur, request)
                items.append(GapItem(competency_id=c.id, competency=c.name, kind="structure",
                                     status="covered" if ok else "missing", evidence=evidence, score=1.0 if ok else 0.0))
                if not ok:
                    domain = (c.check or "").split(":", 1)[1] if (c.check or "").startswith("outcome_domain:") else None
                    recs.append(f"{c.name}: add an outcome such as '{DOMAIN_EXAMPLES[domain]}'" if domain
                                else f"{c.name}: {STRUCTURE_ADVICE.get(c.check or '', 'fix this design item.')}")
                continue
            units, sims = (module_units, sims_topic) if c.kind == "topic" else (outcome_units, sims_outcome)
            names = [m["title"] for m in modules] if c.kind == "topic" else outcome_units
            row = matchable.index(c)
            hit_at = next(((i, k) for i, u in enumerate(units) if (k := keyword_hit(c, u))), None)
            if hit_at:
                i, k = hit_at
                where = f"module '{names[i]}'" if c.kind == "topic" else f"outcome '{names[i][:80]}'"
                items.append(GapItem(competency_id=c.id, competency=c.name, kind=c.kind, status="covered",
                                     evidence=f"{where} mentions '{k}'", score=1.0))
                continue
            scores = sims[row] if sims is not None and units else None
            best_i, best = (int(np.argmax(scores)), float(np.max(scores))) if scores is not None else (None, 0.0)
            # partial = a distinctive word in common (closest such unit by embedding), or an outstanding embedding score
            name_terms = terms(c.name) - generic
            comp_terms = terms(" ".join([c.name, *c.keywords])) - generic
            # a word of the competency's name counts; words only in its keywords need two in common
            sharing = [i for i, u in enumerate(units)
                       if name_terms & terms(u) or len(comp_terms & terms(u)) >= 2]
            match, why = None, ""
            if sharing:
                match = max(sharing, key=lambda i: scores[i] if scores is not None else 0)
                shared = comp_terms & terms(units[match])
                words = dict.fromkeys(w for w in re.findall(r"\w+", normalize(units[match])) if _stem(w) in shared)
                why = "shares '" + "', '".join(list(words)[:3]) + "'"
            elif best_i is not None and best >= threshold and best - float(np.mean(scores)) >= margin:
                match, why = best_i, f"semantically close ({best:.2f})"
            if match is not None:
                items.append(GapItem(competency_id=c.id, competency=c.name, kind=c.kind, status="partial",
                                     evidence=f"'{names[match][:80]}' {why}",
                                     score=round(min(float(scores[match]) if scores is not None else 0.5, 1.0), 3)))
                best_i = match
                recs.append(f"Confirm that '{names[best_i][:60]}' covers '{c.name}', or name it explicitly "
                            f"({', '.join(c.keywords[:3])}).")
            else:
                items.append(GapItem(competency_id=c.id, competency=c.name, kind=c.kind, status="missing",
                                     evidence="no module mentions it" if c.kind == "topic" else "no outcome addresses it",
                                     score=round(max(best, 0.0), 3)))
                if c.kind == "topic":
                    target = f"module '{names[best_i]}'" if best_i is not None and best >= threshold else "a new module"
                    recs.append(f"Add '{c.name}' ({', '.join(c.keywords[:3])}) to {target}.")
                else:
                    recs.append(f"Add a learning outcome for '{c.name}' ({c.description or ', '.join(c.keywords[:3])}).")

        covered = [i.competency for i in items if i.status == "covered"]
        partial = [i.competency for i in items if i.status == "partial"]
        missing = [i for i in items if i.status == "missing"]
        domain_checks = {c.id for c in comps if (c.check or "").startswith("outcome_domain:")}
        missing_outcomes = [i.competency for i in missing if i.kind == "outcome" or i.competency_id in domain_checks]
        req_check = _requested_outcomes(cur, request)
        if not req_check[0]:
            missing_outcomes.append(req_check[1])
        total = len(items) or 1
        coverage = round(100 * (len(covered) + 0.5 * len(partial)) / total, 1)
        if not missing and not partial:
            recs.append(f"No gaps against {framework.short_name}. This is alignment only: confirm with "
                        f"{framework.source_id} before making any accreditation or certification statement.")
        return GapReport(
            report_id="GAP-" + uuid.uuid4().hex[:8].upper(), curriculum_id=cur.get("curriculum_id", ""),
            framework=framework.name, framework_id=framework.framework_id, country=framework.country,
            covered=covered, partial=partial,
            missing_topics=[i.competency for i in missing if i.kind == "topic"],
            missing_outcomes=missing_outcomes, coverage_pct=coverage, items=items,
            recommendations=list(dict.fromkeys(recs)), curriculum_version=cur.get("version"),
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
            sources=[framework.source_ref(),
                     SourceRef(source_id="SRC-ACAD-QUALITY", title="Curriculum frameworks (team summaries)",
                               record_id="data/curriculum/frameworks.yaml")])
