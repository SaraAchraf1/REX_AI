"""SQLite persistence helpers (owner: Member 3). Agents call these instead of raw SQL."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import yaml

from rex_ai import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None, seed: bool = True) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)
        if seed:
            _seed(conn)


# Columns added after Sprint 1: CREATE TABLE IF NOT EXISTS does not add them to an existing database.
_ADDED_COLUMNS = {
    "rfps": {"source_title": "TEXT", "relevance_score": "REAL NOT NULL DEFAULT 0", "matched_course_ids": "TEXT",
             "priority": "TEXT NOT NULL DEFAULT 'normal'", "first_seen": "TEXT"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _load(name: str) -> dict:
    return json.loads((config.DATA_DIR / name).read_text(encoding="utf-8"))


def _seed(conn: sqlite3.Connection) -> None:
    """Load data/*.json and data/data_sources.yaml into the database (idempotent)."""
    reg = yaml.safe_load((config.DATA_DIR / "data_sources.yaml").read_text(encoding="utf-8"))
    for category, items in reg.items():
        for s in items:
            country = s.get("country")
            conn.execute(
                "INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?,?,?,?,?)",
                (s["source_id"], s["name"], category,
                 json.dumps(country) if isinstance(country, list) else country,
                 s.get("url") or s.get("location"), s.get("access"), s["status"], s.get("owner"), None),
            )

    courses = _load("courses.json")
    meta = courses["_meta"]
    prices = {p["course_id"]: p for p in _load("pricing_promotions.json")["prices"]}
    for c in courses["courses"]:
        p = prices.get(c["course_id"], {})
        conn.execute(
            """INSERT OR REPLACE INTO courses VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (c["course_id"], c["title_en"], c["title_ar"], c["department"], json.dumps(c["industry"]),
             c["skill_level"], c["target_audience"], c["duration_hours"], c["duration_weeks"],
             json.dumps(c["formats"]), json.dumps(c["language"]), c["description_en"], c["description_ar"],
             json.dumps(c["learning_outcomes"]), json.dumps(c["modules"]), json.dumps(c["prerequisites"]),
             json.dumps(c["career_opportunities"]), json.dumps(c["tags"]), c.get("certificate"),
             c.get("accreditation_alignment"), json.dumps(c["country"]), p.get("fee"),
             meta.get("currency", "EGP"), int(bool(p.get("installments"))), meta["source_id"],
             meta["last_updated"], int(meta.get("status") == "sample")),
        )

    for i in _load("instructors.json")["instructors"]:
        conn.execute("INSERT OR REPLACE INTO instructors VALUES (?,?,?,?,?,?,?)",
                     (i["instructor_id"], i["name"], i["title"], i["years_experience"],
                      json.dumps(i["expertise"]), i["bio_en"], i["bio_ar"]))
    for c in courses["courses"]:
        for iid in c["instructor_ids"]:
            conn.execute("INSERT OR IGNORE INTO course_instructors VALUES (?,?)", (c["course_id"], iid))

    for s in _load("schedules.json")["schedules"]:
        conn.execute("INSERT OR REPLACE INTO schedules VALUES (?,?,?,?,?,?,?,?)",
                     (s["schedule_id"], s["course_id"], s["start_date"], s["format"], s["days"],
                      s["time"], s["location"], s["seats_left"]))

    for p in _load("pricing_promotions.json")["promotions"]:
        applies = p["applies_to"] if isinstance(p["applies_to"], str) else json.dumps(p["applies_to"])
        conn.execute("INSERT OR REPLACE INTO promotions VALUES (?,?,?,?,?,?,?,?)",
                     (p["promo_id"], p["name_en"], p["name_ar"], p["discount_pct"], applies,
                      p["condition_en"], p["condition_ar"], p["valid_until"]))

    for f in _load("faqs.json")["faqs"]:
        conn.execute("INSERT OR REPLACE INTO faqs VALUES (?,?,?,?,?,?)",
                     (f["faq_id"], f["category"], f["q_en"], f["a_en"], f["q_ar"], f["a_ar"]))


# --------------------------------------------------------------------------- #
# Chat sessions (REX)
# --------------------------------------------------------------------------- #
def load_session(session_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None


def save_session(session_id: str, language: str, profile: dict, stage: str,
                 db_path: Optional[Path] = None) -> None:
    now = _now()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO chat_sessions (session_id, language, profile, stage, created_at, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET language=excluded.language,
                 profile=excluded.profile, stage=excluded.stage, updated_at=excluded.updated_at""",
            (session_id, language, json.dumps(profile, ensure_ascii=False), stage, now, now),
        )


def add_message(session_id: str, role: str, content: str, sources: Optional[list] = None,
                db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT INTO chat_messages (session_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
                     (session_id, role, content, json.dumps(sources or [], ensure_ascii=False), _now()))


def recent_messages(session_id: str, limit: int = 10, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
            (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# Leads (consent enforced by schema AND by the LeadCreate model)
# --------------------------------------------------------------------------- #
def save_lead(lead: dict, db_path: Optional[Path] = None) -> dict:
    if not lead.get("consent"):
        raise ValueError("consent required")
    lead_id = "LEAD-" + uuid.uuid4().hex[:10]
    now = _now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO leads VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lead_id, lead.get("session_id"), lead["name"], lead.get("email"), lead.get("phone"),
             json.dumps(lead.get("interested_course_ids", [])),
             json.dumps(lead.get("profile") or {}, ensure_ascii=False), 1, now, now),
        )
    return {**lead, "lead_id": lead_id, "created_at": now}


def list_leads(db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM leads ORDER BY created_at DESC")]


def log_event(level: str, event: str, agent: Optional[str] = None, detail: Optional[str] = None,
              db_path: Optional[Path] = None) -> None:
    try:
        with connect(db_path) as conn:
            conn.execute("INSERT INTO app_logs (level, agent, event, detail, created_at) VALUES (?,?,?,?,?)",
                         (level, agent, event, detail, _now()))
    except sqlite3.Error:
        pass  # logging must never break a request


# --------------------------------------------------------------------------- #
# Job runs (every agent run is traceable: queued -> running -> succeeded/failed)
# --------------------------------------------------------------------------- #
def start_job(agent: str, payload: dict, db_path: Optional[Path] = None) -> str:
    job_id = "JOB-" + uuid.uuid4().hex[:10]
    with connect(db_path) as conn:
        conn.execute("INSERT INTO job_runs (job_id, agent, status, input, started_at) VALUES (?,?,?,?,?)",
                     (job_id, agent, "running", json.dumps(payload, ensure_ascii=False, default=str), _now()))
    return job_id


def finish_job(job_id: str, status: str, output_ref: Optional[str] = None, error: Optional[str] = None,
               db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE job_runs SET status=?, output_ref=?, error=?, finished_at=? WHERE job_id=?",
                     (status, output_ref, error, _now(), job_id))


# --------------------------------------------------------------------------- #
# Tender / RFP research (owner: Member 2)
# --------------------------------------------------------------------------- #
_RFP_JSON = ("available_documents", "matched_course_ids")


def _rfp_row(row: sqlite3.Row) -> dict:
    r = dict(row)
    for key in _RFP_JSON:
        r[key] = json.loads(r[key]) if r.get(key) else []
    return r


def save_rfp(rec: dict, db_path: Optional[Path] = None) -> None:
    """Insert a new RFP or refresh an existing one (same rfp_id). Status set by a reviewer is kept."""
    src = rec["source"]
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO rfps (rfp_id, title, issuing_organization, country, deadline, eligibility_requirements,
                 training_category, submission_link, publication_date, available_documents, summary, qualification,
                 qualification_notes, status, dedupe_hash, source_id, source_url, last_checked, source_title,
                 relevance_score, matched_course_ids, priority, first_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(rfp_id) DO UPDATE SET deadline=excluded.deadline,
                 eligibility_requirements=excluded.eligibility_requirements, submission_link=excluded.submission_link,
                 available_documents=excluded.available_documents, summary=excluded.summary,
                 qualification=excluded.qualification, qualification_notes=excluded.qualification_notes,
                 relevance_score=excluded.relevance_score, matched_course_ids=excluded.matched_course_ids,
                 last_checked=excluded.last_checked""",
            (rec["rfp_id"], rec["title"], rec["issuing_organization"], rec["country"], rec.get("deadline"),
             rec.get("eligibility_requirements"), rec["training_category"], rec.get("submission_link"),
             rec.get("publication_date"), json.dumps(rec.get("available_documents", []), ensure_ascii=False),
             rec.get("summary"), rec["qualification"], rec.get("qualification_notes"), rec["status"],
             rec["dedupe_hash"], src["source_id"], src.get("url"), rec["last_checked"], src.get("title"),
             rec.get("relevance_score", 0.0), json.dumps(rec.get("matched_course_ids", [])),
             rec.get("priority", "normal"), rec.get("first_seen") or rec["last_checked"]),
        )


def get_rfp(rfp_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM rfps WHERE rfp_id=?", (rfp_id,)).fetchone()
        return _rfp_row(row) if row else None


def find_rfp_by_hash(dedupe_hash: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM rfps WHERE dedupe_hash=?", (dedupe_hash,)).fetchone()
        return _rfp_row(row) if row else None


def list_rfps(country: Optional[str] = None, status: Optional[str] = None, category: Optional[str] = None,
              qualification: Optional[str] = None, q: Optional[str] = None, open_only: bool = False,
              today: Optional[str] = None, limit: int = 500, db_path: Optional[Path] = None) -> list[dict]:
    """Dashboard search: exact filters + free text over title, organisation, eligibility and summary."""
    sql, args = "SELECT * FROM rfps WHERE 1=1", []
    for col, val in (("country", country), ("status", status), ("training_category", category),
                     ("qualification", qualification)):
        if val:
            sql += f" AND {col}=?"
            args.append(val)
    if q:
        sql += (" AND (title LIKE ? OR issuing_organization LIKE ? OR eligibility_requirements LIKE ?"
                " OR summary LIKE ?)")
        args += [f"%{q}%"] * 4
    if open_only:
        sql += " AND (deadline IS NULL OR deadline >= ?)"
        args.append(today or datetime.now(timezone.utc).date().isoformat())
    sql += " ORDER BY CASE priority WHEN 'high' THEN 0 ELSE 1 END, deadline IS NULL, deadline LIMIT ?"
    args.append(limit)
    with connect(db_path) as conn:
        return [_rfp_row(r) for r in conn.execute(sql, args)]


def update_rfp_status(rfp_id: str, status: str, db_path: Optional[Path] = None) -> bool:
    with connect(db_path) as conn:
        return conn.execute("UPDATE rfps SET status=? WHERE rfp_id=?", (status, rfp_id)).rowcount > 0


def expire_rfps(today: str, db_path: Optional[Path] = None) -> int:
    """Mark stored RFPs whose deadline has passed (reviewer decisions qualified/rejected are kept)."""
    with connect(db_path) as conn:
        return conn.execute(
            "UPDATE rfps SET status='expired' WHERE deadline < ? AND status IN ('new','flagged_review','monitoring')",
            (today,)).rowcount


def save_rfp_alert(alert: dict, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT INTO rfp_alerts VALUES (?,?,?,?,?,?,?)",
                     (alert["alert_id"], alert["rfp_id"], alert["country"], alert["priority"], alert["message"],
                      json.dumps(alert["channels"]), alert["created_at"]))


def list_rfp_alerts(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM rfp_alerts ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return [{**dict(r), "channels": json.loads(r["channels"])} for r in rows]
