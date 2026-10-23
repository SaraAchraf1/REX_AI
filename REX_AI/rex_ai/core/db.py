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
        if seed:
            _seed(conn)


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
