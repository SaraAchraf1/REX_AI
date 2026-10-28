"""Accreditation and competency frameworks for the Curriculum Agent (owner: Member 2).

data/curriculum/frameworks.yaml holds team summaries of public frameworks (ISTQB, AWS, PMP, SFIA) and
country design checklists (Egypt NAQAAE, Saudi NQF-KSA), plus the academy's own quality baseline. They are
used for ALIGNMENT only; nothing here can make a course "accredited".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import yaml

from rex_ai import config
from rex_ai.agents.rex.text_utils import contains_any, normalize
from rex_ai.agents.rfp.sources import load_registry
from rex_ai.core.schemas import FrameworkInfo, SourceRef

BASELINE_ID = "REX-QUALITY"
COUNTRY_FRAMEWORKS = {"EG": "EG-NAQAAE-PS", "SA": "SA-NQF-DOMAINS"}


@dataclass
class Competency:
    id: str
    name: str
    kind: str                                   # topic | outcome | structure
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    check: Optional[str] = None                 # structure rule name (alignment.STRUCTURE_CHECKS)
    departments: list[str] = field(default_factory=list)


@dataclass
class Framework:
    framework_id: str
    name: str
    short_name: str
    country: str
    source_id: str
    competencies: list[Competency]
    aliases: list[str] = field(default_factory=list)
    url: Optional[str] = None
    last_checked: Optional[str] = None
    exam_prep: bool = False
    applies_to: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    filter_by_department: bool = False
    note: Optional[str] = None
    status: str = "to_verify"

    def competencies_for(self, department: Optional[str]) -> list[Competency]:
        """SFIA-style frameworks list skills for many fields: keep the ones for this department."""
        if not self.filter_by_department or not department:
            return self.competencies
        kept = [c for c in self.competencies if not c.departments or department in c.departments]
        return kept or self.competencies

    def source_ref(self) -> SourceRef:
        return SourceRef(source_id=self.source_id, title=self.name, url=self.url, record_id=self.framework_id)

    def info(self) -> FrameworkInfo:
        return FrameworkInfo(framework_id=self.framework_id, name=self.name, short_name=self.short_name,
                             country=self.country, source_id=self.source_id, url=self.url, status=self.status,
                             exam_prep=self.exam_prep, competencies=len(self.competencies), note=self.note)


@lru_cache(maxsize=1)
def load_frameworks() -> dict[str, Framework]:
    path = config.CURRICULUM_DIR / "frameworks.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {"frameworks": []}
    registry = load_registry()
    out: dict[str, Framework] = {}
    for f in data.get("frameworks", []):
        comps = [Competency(id=c["id"], name=c["name"], kind=c["kind"], description=c.get("description", ""),
                            keywords=[str(k) for k in c.get("keywords", [])], check=c.get("check"),
                            departments=c.get("departments", [])) for c in f["competencies"]]
        out[f["framework_id"]] = Framework(
            framework_id=f["framework_id"], name=f["name"], short_name=f.get("short_name", f["name"]),
            country=f.get("country", "INTL"), source_id=f["source_id"], competencies=comps,
            aliases=f.get("aliases", []), url=f.get("url"), last_checked=str(f.get("last_checked") or "") or None,
            exam_prep=bool(f.get("exam_prep")), applies_to=f.get("applies_to", []),
            departments=f.get("departments", []), filter_by_department=bool(f.get("filter_by_department")),
            note=" ".join(str(f.get("note", "")).split()) or None,
            status=registry.get(f["source_id"], {}).get("status", "to_verify"))
    return out


def find_framework(name: Optional[str]) -> Optional[Framework]:
    """Resolve 'ISTQB Foundation', 'istqb', 'SFIA-9', 'NAQAAE'... to a framework (None if unknown)."""
    if not name:
        return None
    fws = load_frameworks()
    if name in fws:
        return fws[name]
    key = normalize(name)
    for f in fws.values():
        if key in {normalize(x) for x in [f.framework_id, f.short_name, f.name, *f.aliases]}:
            return f
    if len(key) < 3:
        return None
    for f in fws.values():                       # partial: "SFIA" -> SFIA 9, "ISTQB CTFL v4" -> ISTQB
        names = [normalize(x) for x in [f.short_name, *f.aliases]]
        if any(n in key or key in n for n in names if len(n) >= 3):
            return f
    return None


def suggest_framework(title: str, department: Optional[str], country: str,
                      text: str = "") -> tuple[Optional[Framework], str]:
    """Best framework to align a new course with, and why. Exam-specific frameworks win when the title names
    them; then SFIA for IT departments; then the country's design checklist."""
    fws = load_frameworks()
    haystack = f"{title} {text}"
    for f in fws.values():
        if f.exam_prep and contains_any(title, f.applies_to + f.aliases):
            return f, f"the course title refers to {f.short_name}"
    for f in fws.values():
        if f.exam_prep and contains_any(haystack, f.aliases):
            return f, f"the request mentions {f.short_name}"
    sfia = fws.get("SFIA-9")
    if sfia and department in sfia.departments:
        return sfia, f"SFIA covers {department} skills"
    if country_fw := country_framework(country):
        return country_fw, f"design checklist for {country}"
    return fws.get(BASELINE_ID), "no specific framework applies; academy quality baseline"


def named_in(title: str, framework: Optional[Framework]) -> bool:
    """True when the title names the framework itself ('PMP Exam Prep', 'ISTQB Foundation'), not just its field."""
    return bool(framework) and contains_any(title, [framework.short_name, *framework.aliases])


def country_framework(country: Optional[str]) -> Optional[Framework]:
    return load_frameworks().get(COUNTRY_FRAMEWORKS.get(country or "", ""))


def baseline() -> Framework:
    return load_frameworks()[BASELINE_ID]


def sync_to_db() -> None:
    """Copy every competency into the accreditation_standards table (traceable, queryable)."""
    from rex_ai.core import db
    rows = [{"standard_id": f"{f.framework_id}:{c.id}", "framework": f.framework_id, "country": f.country,
             "competency": c.name, "description": c.description or c.check, "source_id": f.source_id,
             "last_checked": f.last_checked} for f in load_frameworks().values() for c in f.competencies]
    db.save_standards(rows)
