"""Course catalogue import - shared by the CLI (scripts/import_catalog.py) and the API startup.

Every import MERGES into the existing data: new course IDs are added, an ID that already exists is
updated, nothing is ever deleted. Drop catalogue files (PDF, Word, Excel, CSV, TXT) into data/catalog/
(sub-folders too, except backup/): a running API or UI merges them automatically (REX_AUTO_IMPORT=1, see
rex_ai/core/data_sync.py), POST /data/upload and POST /data/courses add them through the API, or run

    python scripts/import_catalog.py                       # merge every new/changed file in data/catalog/
    python scripts/import_catalog.py path/to/file.pdf      # merge one file
    python scripts/import_catalog.py --dry-run             # check only, write nothing

Files that were already imported and have not changed are skipped (data/catalog/.imported.json).
Before writing, the old data files are backed up to data/backup/<timestamp>/; if the merged data fails
validation the backup is restored.

Written files:
    data/courses.json             course records (validated with the Course model)
    data/pricing_promotions.json  fees + installments per course
    data/schedules.json           cohorts from "Schedule:" lines

Catalogue format
    PDF / Word / TXT: one block per course, each field on its own line as "Label: value". A new course starts
    at every "Course ID:" line. Long values may wrap onto the next lines. Lists use ";" (or one "-"/"•"
    bullet per line). Word tables with two columns (Label | Value) also work.
    Excel / CSV: one row per course (every sheet is read), column headers = the labels below.

    Course ID: AI-301                    Department: AI
    Title: Deep Learning with PyTorch     Arabic Title: (optional, falls back to Title)
    Level: Beginner | Intermediate | Advanced
    Target Audience: ...                  Industry: Technology; Healthcare
    Duration (hours): 48                  Duration (weeks): 8
    Formats: Online live; Hybrid; Onsite  Language: English; Arabic
    Description: ...                      Arabic Description: (optional)
    Learning Outcomes: a; b; c            Modules: a; b; c
    Prerequisites: a; b                   Career Opportunities: a; b
    Tags: deep learning; pytorch          Certificate: ...
    Instructors: INS-01 (or an instructor's name from data/instructors.json)
    Country: Egypt; Saudi Arabia          Fee (EGP): 16,500        Installments: Yes
    Schedule: 2026-12-06 | Online live | Sun, Tue | 19:00-22:00 | 20 seats    (repeatable, optional)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from rex_ai import config
from rex_ai.agents.rex.profile_extractor import TOPIC_TO_TAGS
from rex_ai.core.schemas import Course

log = logging.getLogger("rex.catalog")
CATALOG_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".txt", ".md"}
TABLE_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv"}
DATA_FILES = ("courses.json", "pricing_promotions.json", "schedules.json")
REGISTRY = ".imported.json"

# label (lower case) -> field. Several spellings accepted.
LABELS = {
    "course id": "course_id", "course code": "course_id", "code": "course_id", "id": "course_id",
    "title": "title_en", "course title": "title_en", "title (en)": "title_en", "english title": "title_en",
    "arabic title": "title_ar", "title (ar)": "title_ar",
    "department": "department",
    "industry": "industry", "industries": "industry",
    "level": "skill_level", "skill level": "skill_level",
    "target audience": "target_audience", "audience": "target_audience",
    "duration (hours)": "duration_hours", "hours": "duration_hours", "duration hours": "duration_hours",
    "duration (weeks)": "duration_weeks", "weeks": "duration_weeks", "duration weeks": "duration_weeks",
    "formats": "formats", "format": "formats", "delivery": "formats", "delivery format": "formats",
    "language": "language", "languages": "language",
    "description": "description_en", "description (en)": "description_en",
    "arabic description": "description_ar", "description (ar)": "description_ar",
    "learning outcomes": "learning_outcomes", "outcomes": "learning_outcomes",
    "modules": "modules", "topics": "modules",
    "prerequisites": "prerequisites", "requirements": "prerequisites",
    "career opportunities": "career_opportunities", "careers": "career_opportunities",
    "tags": "tags", "keywords": "tags",
    "certificate": "certificate",
    "accreditation alignment": "accreditation_alignment", "accreditation": "accreditation_alignment",
    "instructors": "instructor_ids", "instructor": "instructor_ids", "instructor ids": "instructor_ids",
    "country": "country", "countries": "country",
    "fee": "fee", "fee (egp)": "fee", "price": "fee", "price (egp)": "fee", "fees": "fee",
    "installments": "installments", "instalments": "installments",
    "schedule": "schedule", "schedules": "schedule", "cohort": "schedule", "next start": "schedule",
}
LABELS.update({f: f for f in set(LABELS.values())})   # JSON records from the API may use the field names
SKIP_DIRS = {"backup"}                                   # sub-folders of data/catalog/ that are never imported
LIST_FIELDS = {"industry", "formats", "language", "learning_outcomes", "modules", "prerequisites",
               "career_opportunities", "tags", "instructor_ids", "country", "schedule"}
REQUIRED = ["course_id", "title_en", "department", "skill_level", "duration_hours", "description_en"]

FORMAT_WORDS = {"online_live": ["online", "live", "virtual", "اونلاين", "أونلاين"],
                "onsite": ["onsite", "on-site", "in person", "in-person", "classroom", "حضوري"],
                "hybrid": ["hybrid", "blended", "هجين"]}
LANG_WORDS = {"en": ["english", "en", "انجليزي", "الإنجليزية"], "ar": ["arabic", "ar", "عربي", "العربية"]}
COUNTRY_WORDS = {"EG": ["egypt", "eg", "مصر"], "SA": ["saudi", "ksa", "sa", "السعودية"]}
LEVEL_WORDS = {"Beginner": ["beginner", "entry", "basic", "مبتدئ"], "Intermediate": ["intermediate", "متوسط"],
               "Advanced": ["advanced", "expert", "متقدم"]}

LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ()/]{1,40}?)\s*[:：]\s*(.*)$")
HEADING_RE = re.compile(r"^[A-Z]{2,4}-?\d{2,4}\b")
BULLET_RE = re.compile(r"^\s*[-•·▪*]\s+(.*)$")


# --------------------------------------------------------------------- reading
def read_lines(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return [ln for page in pdf.pages for ln in (page.extract_text() or "").splitlines()]
    if ext == ".docx":
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        d = docx.Document(str(path))
        lines = []
        for child in d.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                lines.append(Paragraph(child, d).text)
            elif tag == "tbl":
                for row in Table(child, d).rows:
                    cells = [c.text.strip() for c in row.cells]
                    if len(cells) >= 2 and cells[0]:
                        lines.append(f"{cells[0].rstrip(':')}: {' '.join(cells[1:])}")
        return lines
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8").splitlines()
    raise ValueError(f"{path.suffix} is not a text catalogue (use PDF, DOCX, TXT, XLSX or CSV)")


def read_rows(path: Path) -> list[dict]:
    import pandas as pd
    if path.suffix.lower() == ".csv":
        frames = [pd.read_csv(path, dtype=str, keep_default_na=False)]
    else:
        frames = list(pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False).values())
    return rows_to_records([row for df in frames for row in df.to_dict("records")])


def rows_to_records(rows: list[dict]) -> list[dict]:
    """{column label: value} rows (Excel/CSV rows or JSON objects) -> raw course records."""
    records = []
    for row in rows:
        rec: dict = {}
        for col, val in row.items():
            field = LABELS.get(_label(str(col)))
            if not field:
                continue
            for v in (val if isinstance(val, list) else [val]):
                if isinstance(v, dict):                # {"start_date": ..., "format": ...} schedule from JSON
                    v = " | ".join(str(v.get(k, "")) for k in ("start_date", "format", "days", "time", "seats_left"))
                elif isinstance(v, bool):
                    v = "yes" if v else "no"
                if v is not None and str(v).strip():
                    rec.setdefault(field, []).append(str(v).strip())
        if rec.get("course_id"):
            records.append(rec)
    return records


def _label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_lines(lines: list[str]) -> list[dict]:
    """'Label: value' lines -> one dict per course (values are lists of raw strings)."""
    records, rec, field = [], None, None
    for raw in lines:
        line = unicodedata.normalize("NFKC", raw).strip()
        if not line:
            continue
        if HEADING_RE.match(line) and not LABEL_RE.match(line):
            field = None                               # a course heading such as "AI-301 - Deep Learning"
            continue
        m = LABEL_RE.match(line)
        f = LABELS.get(_label(m.group(1))) if m else None
        if f == "course_id":
            rec = {}
            records.append(rec)
        if rec is None:
            continue                                   # text before the first course (cover page...)
        if f:
            field = f
            if m.group(2).strip():
                rec.setdefault(field, []).append(m.group(2).strip())
            else:
                rec.setdefault(field, [])
            continue
        b = BULLET_RE.match(line)
        if field in LIST_FIELDS and b:
            rec.setdefault(field, []).append(b.group(1).strip())
        elif field == "schedule":
            continue                                   # schedule lines never wrap; ignore stray text
        elif field and rec.get(field):
            rec[field][-1] += " " + line                # value wrapped onto the next line
        elif field:
            rec[field] = [line]
    return records


# ----------------------------------------------------------------- normalising
def _split(values: list[str]) -> list[str]:
    out = []
    for v in values:
        out += [p.strip(" .") for p in re.split(r"\s*[;|]\s*", v) if p.strip(" .")]
    return out


def _match(value: str, table: dict[str, list[str]]) -> str | None:
    v = value.lower()
    for key, words in table.items():
        if v == key.lower() or any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", v) for w in words):
            return key
    return None


def _int(value: str) -> int | None:
    m = re.search(r"\d+", value.replace(",", ""))
    return int(m.group()) if m else None


def build_course(raw: dict, instructors: dict[str, str], issues: list[str]) -> tuple[dict, dict, list[dict]]:
    """raw lists -> (course dict for courses.json, price dict, schedules)."""
    one = {k: " ".join(v).strip() for k, v in raw.items() if k not in LIST_FIELDS}
    lists = {k: _split(v) for k, v in raw.items() if k in LIST_FIELDS and k != "schedule"}
    cid = one.get("course_id", "?").upper().replace(" ", "")
    tag = f"{cid}:"

    for f in REQUIRED:
        if not one.get(f):
            issues.append(f"{tag} missing required field '{f}'")

    formats, langs, countries = [], [], []
    for x in lists.get("formats", []):
        key = _match(x, FORMAT_WORDS)
        (formats.append(key) if key else issues.append(f"{tag} unknown format '{x}'"))
    for x in lists.get("language", []):
        key = _match(x, LANG_WORDS)
        (langs.append(key) if key else issues.append(f"{tag} unknown language '{x}'"))
    for x in lists.get("country", []):
        key = _match(x, COUNTRY_WORDS)
        if key:
            countries.append(key)
        else:
            issues.append(f"{tag} country '{x}' ignored (only Egypt and Saudi Arabia are supported)")

    ins_ids = []
    for x in lists.get("instructor_ids", []):
        key = x.upper() if x.upper() in instructors else next(
            (i for i, name in instructors.items() if x.lower() in name.lower()), None)
        (ins_ids.append(key) if key else issues.append(f"{tag} instructor '{x}' not in instructors.json - skipped"))

    hours = _int(one.get("duration_hours", "")) or 0
    weeks = _int(one.get("duration_weeks", "")) or max(1, round(hours / 6))
    level = _match(one.get("skill_level", ""), LEVEL_WORDS) or one.get("skill_level", "")
    missing_ar = [x for x, f in (("title", "title_ar"), ("description", "description_ar")) if not one.get(f)]
    if missing_ar:
        issues.append(f"{tag} no Arabic {' or '.join(missing_ar)} - English text is used in Arabic replies")

    tags = [t.lower() for t in lists.get("tags", [])]
    if not any(set(tags) & topic_tags for topic_tags in TOPIC_TO_TAGS.values()):
        issues.append(f"{tag} no tag matches a REX interest topic, so it is only recommended by name. "
                      f"Add tags such as: {', '.join(sorted({t for s in TOPIC_TO_TAGS.values() for t in s})[:12])}...")

    course = {
        "course_id": cid, "title_en": one.get("title_en", ""), "title_ar": one.get("title_ar") or one.get("title_en", ""),
        "department": one.get("department", ""), "industry": lists.get("industry", []),
        "skill_level": level, "target_audience": one.get("target_audience", ""),
        "duration_hours": hours, "duration_weeks": weeks,
        "formats": formats or ["online_live"], "language": langs or ["en"],
        "description_en": one.get("description_en", ""),
        "description_ar": one.get("description_ar") or one.get("description_en", ""),
        "learning_outcomes": lists.get("learning_outcomes", []), "modules": lists.get("modules", []),
        "prerequisites": lists.get("prerequisites", []), "career_opportunities": lists.get("career_opportunities", []),
        "tags": tags, "certificate": one.get("certificate") or "Academy certificate of completion",
        "accreditation_alignment": one.get("accreditation_alignment") or None,
        "instructor_ids": ins_ids, "country": countries or ["EG"],
    }
    price = {}
    if one.get("fee"):
        fee = re.sub(r"[^\d.]", "", one["fee"].replace(",", ""))
        if fee:
            price = {"course_id": cid, "fee": float(fee) if "." in fee else int(fee),
                     "installments": one.get("installments", "").strip().lower() in ("yes", "true", "y", "1", "نعم")}
    schedules = []
    for line in raw.get("schedule", []):
        parts = [p.strip() for p in line.split("|")]
        try:
            start = datetime.strptime(parts[0], "%Y-%m-%d").date().isoformat()
        except ValueError:
            issues.append(f"{tag} schedule '{line}' skipped (date must be YYYY-MM-DD)")
            continue
        fmt = _match(parts[1], FORMAT_WORDS) if len(parts) > 1 else None
        schedules.append({"course_id": cid, "start_date": start, "format": fmt or course["formats"][0],
                          "days": parts[2] if len(parts) > 2 else "TBA", "time": parts[3] if len(parts) > 3 else "TBA",
                          "location": None if (fmt or course["formats"][0]) == "online_live" else "Academy campus",
                          "seats_left": _int(parts[4]) if len(parts) > 4 and _int(parts[4]) else 20})
    return course, price, schedules


# ---------------------------------------------------------------------- parsing a file
@dataclass
class FileResult:
    file: str
    courses: list[dict] = field(default_factory=list)
    prices: list[dict] = field(default_factory=list)
    schedules: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False                       # unchanged since the last import
    digest: str = ""


def parse_file(path: Path, instructors: Optional[dict[str, str]] = None) -> FileResult:
    """Read one catalogue file into validated course records (nothing is written)."""
    instructors = instructors if instructors is not None else \
        {i["instructor_id"]: i["name"] for i in _load("instructors.json")["instructors"]}
    res = FileResult(path.name)
    try:
        raw_records = read_rows(path) if path.suffix.lower() in TABLE_EXTENSIONS else parse_lines(read_lines(path))
    except Exception as exc:
        res.errors.append(f"could not read the file: {exc}")
        return res
    if not raw_records:
        res.errors.append("no courses found - each course must start with a 'Course ID: ...' line "
                          "(or a 'Course ID' column in Excel/CSV)")
        return res
    return _build(res, raw_records, instructors)


def parse_records(rows: list[dict], name: str = "api", instructors: Optional[dict[str, str]] = None) -> FileResult:
    """Course records sent as JSON (labels or field names as keys) -> validated FileResult."""
    instructors = instructors if instructors is not None else \
        {i["instructor_id"]: i["name"] for i in _load("instructors.json")["instructors"]}
    res = FileResult(name)
    raw_records = rows_to_records(rows)
    if len(raw_records) < len(rows):
        res.errors.append(f"{len(rows) - len(raw_records)} record(s) without a course_id were ignored")
    return _build(res, raw_records, instructors)


def _build(res: FileResult, raw_records: list[dict], instructors: dict[str, str]) -> FileResult:
    for raw in raw_records:
        course, price, sch = build_course(raw, instructors, res.warnings)
        try:
            Course(**course)
        except ValidationError as exc:
            res.errors += [f"{course['course_id']}: {'.'.join(map(str, e['loc']))} - {e['msg']}" for e in exc.errors()]
            continue
        res.courses.append(course)
        if price:
            res.prices.append(price)
        else:
            res.warnings.append(f"{course['course_id']}: no fee - REX will say the price is not available")
        res.schedules += sch
    return res


# ---------------------------------------------------------------------- merging + writing
def _load(name: str) -> dict:
    return json.loads((config.DATA_DIR / name).read_text(encoding="utf-8"))


def _save(name: str, data: dict):
    (config.DATA_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    # CRLF -> LF first: a Windows checkout (git core.autocrlf) must not look like a changed catalogue
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:24]


def merge(results: list[FileResult], status: Optional[str] = None) -> dict:
    """Merge parsed files into data/*.json. Existing courses are kept; same course_id = updated.
    Returns {"added": [...], "updated": [...], "courses": n, "backup": path}."""
    courses = [c for r in results for c in r.courses]
    if not courses:
        return {"added": [], "updated": [], "courses": None, "backup": None}

    backup = config.DATA_DIR / "backup" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        shutil.copy2(config.DATA_DIR / name, backup / name)

    cat, pricing, sched = _load("courses.json"), _load("pricing_promotions.json"), _load("schedules.json")
    by_id = {c["course_id"]: c for c in cat["courses"]}
    existing = set(by_id)
    for c in courses:                              # later files win when two files share a course ID
        by_id[c["course_id"]] = c
    new_ids = {c["course_id"] for c in courses}
    cat["courses"] = list(by_id.values())
    sources = list(dict.fromkeys(cat["_meta"].get("imported_from", []) if isinstance(
        cat["_meta"].get("imported_from"), list) else [x for x in [cat["_meta"].get("imported_from")] if x]))
    sources += [r.file for r in results if r.courses and r.file not in sources]
    cat["_meta"].update({"last_updated": date.today().isoformat(), "imported_from": sources})
    if status:
        cat["_meta"]["status"] = status

    prices = {p["course_id"]: p for p in pricing["prices"]}
    for r in results:
        for p in r.prices:
            prices[p["course_id"]] = p
    pricing["prices"] = list(prices.values())
    pricing["_meta"]["last_updated"] = date.today().isoformat()

    # cohorts: an imported course's cohorts come from its catalogue file; other courses keep theirs
    n = max([int(re.sub(r"\D", "", s["schedule_id"]) or 0) for s in sched["schedules"]] + [0])
    new_sched = [s for r in results for s in r.schedules]
    with_new_cohorts = {s["course_id"] for s in new_sched}
    sched["schedules"] = [s for s in sched["schedules"] if s["course_id"] not in with_new_cohorts]
    for s in new_sched:
        n += 1
        sched["schedules"].append({"schedule_id": f"SCH-{n:03d}", **s})

    _save("courses.json", cat)
    _save("pricing_promotions.json", pricing)
    _save("schedules.json", sched)

    try:                                           # validate everything; roll back on failure
        from rex_ai.agents.rex.knowledge_base import load_knowledge_base
        kb = load_knowledge_base(config.DATA_DIR)
    except Exception as exc:
        for name in DATA_FILES:
            shutil.copy2(backup / name, config.DATA_DIR / name)
        raise RuntimeError(f"merged data failed validation ({exc}); previous files restored from {backup}") from exc
    return {"added": sorted(new_ids - existing), "updated": sorted(new_ids & existing),
            "courses": len(kb.courses), "backup": str(backup)}


def import_paths(paths: list[Path], dry_run: bool = False, force: bool = False,
                 status: Optional[str] = None) -> dict:
    """Parse + merge files and/or folders. Unchanged, already-imported files are skipped unless force."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += catalog_files(p)
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)

    catalog_dir = config.DATA_DIR / "catalog"
    registry_path = catalog_dir / REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    instructors = {i["instructor_id"]: i["name"] for i in _load("instructors.json")["instructors"]}
    results: list[FileResult] = []
    for f in files:
        try:                                       # files in sub-folders are keyed by their relative path
            key = f.resolve().relative_to(catalog_dir.resolve()).as_posix()
        except ValueError:
            key = f.name
        digest = _file_hash(f)
        if not force and registry.get(key) == digest:
            results.append(FileResult(key, skipped=True))
            continue
        res = parse_file(f, instructors)
        res.file, res.digest = key, digest
        results.append(res)

    report = {"files": results, "merge": None}
    to_merge = [r for r in results if not r.skipped and r.courses]
    if dry_run or not to_merge:
        return report
    report["merge"] = merge(to_merge, status)
    for r in to_merge:
        registry[r.file] = r.digest
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return report


def catalog_files(folder: Path) -> list[Path]:
    """Supported files in a folder and its sub-folders (except backup/ and hidden/temp files)."""
    return sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in CATALOG_EXTENSIONS
                  and not f.name.startswith((".", "~$"))
                  and not any(part in SKIP_DIRS or part.startswith(".") for part in f.relative_to(folder).parts[:-1]))


def import_records(rows: list[dict], name: str = "api", status: Optional[str] = None) -> dict:
    """Merge course records sent through the API. Same merge rules as files: add or update, never delete."""
    res = parse_records(rows, name)
    return {"files": [res], "merge": merge([res], status) if res.courses else None}


def auto_import(always: bool = False) -> Optional[dict]:
    """Merge new/changed files from data/catalog/ (on startup and whenever the folder changes). Never raises.
    always=True runs even when REX_AUTO_IMPORT=0 (explicit upload / reload)."""
    folder = config.DATA_DIR / "catalog"
    if not (always or config.AUTO_IMPORT) or not folder.is_dir():
        return None
    try:
        report = import_paths([folder])
    except Exception as exc:
        log.error("Catalogue auto-import failed: %s", exc)
        return None
    for r in report["files"]:
        if r.errors:
            log.warning("Catalogue %s: %d error(s), e.g. %s", r.file, len(r.errors), r.errors[0])
    m = report["merge"]
    if m:
        log.info("Catalogue merged: %d added, %d updated -> %d courses", len(m["added"]), len(m["updated"]),
                 m["courses"])
    return report


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge academy catalogue files into REX's course data")
    ap.add_argument("paths", nargs="*", type=Path, help="files or folders (default: data/catalog/)")
    ap.add_argument("--dry-run", action="store_true", help="parse and validate only, write nothing")
    ap.add_argument("--force", action="store_true", help="re-import files even if unchanged")
    ap.add_argument("--status", default=None, help="set _meta.status (e.g. approved once the academy signs off)")
    args = ap.parse_args(argv)

    paths = args.paths or [config.DATA_DIR / "catalog"]
    try:
        report = import_paths(paths, dry_run=args.dry_run, force=args.force, status=args.status)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    had_errors = False
    for r in report["files"]:
        if r.skipped:
            print(f"= {r.file}: unchanged since last import (use --force to re-import)")
            continue
        print(f"\n{r.file}: {len(r.courses)} valid course(s), {len(r.errors)} error(s)")
        for c in r.courses:
            print(f"    {c['course_id']:<8} {c['title_en'][:45]:<45} {c['skill_level']:<12} {c['duration_hours']}h  "
                  f"{'/'.join(c['formats'])}")
        for w in r.warnings:
            print(f"  - {w}")
        for e in r.errors:
            print(f"  ! {e}")
        had_errors |= bool(r.errors)

    m = report["merge"]
    if args.dry_run:
        print("\nDry run - nothing written.")
    elif m:
        print(f"\nMerged: {len(m['added'])} added {m['added']}, {len(m['updated'])} updated {m['updated']}. "
              f"REX now has {m['courses']} courses (nothing removed).")
        print(f"Backup of the previous data: {m['backup']}")
        print("A running API / UI picks them up on the next message - the RAG index rebuilds automatically.")
    else:
        print("\nNothing new to merge.")
    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())