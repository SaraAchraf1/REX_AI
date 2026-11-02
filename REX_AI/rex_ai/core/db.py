"""Persistence helpers (owner: Member 3). Agents call these instead of raw SQL.

Two backends, same functions:
  shared server   REX_DATABASE_URL (or DATABASE_URL) = postgresql://user:password@host:5432/dbname
                  Every action (chats, leads, RFPs, curricula, documents, campaigns, reviews, versions, handoffs,
                  jobs, logs, uploaded data files) is stored on that PostgreSQL server, so the app shows the same
                  history on any machine or server that points to it. Connections are pooled.
  local file      no URL: SQLite file REX_DB_PATH (rex_ai.db) - offline development and the test suite.

The SQL below is written once (SQLite dialect); _pg_sql() translates it for PostgreSQL (placeholders, upserts,
case-insensitive LIKE). The table definitions live in schema.sql for both.

    python -m rex_ai.core.db status                       # which database, row counts
    python -m rex_ai.core.db migrate --from rex_ai.db     # copy a local SQLite history into the shared database
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import yaml

from rex_ai import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Backend selection, connections and SQL dialect
# --------------------------------------------------------------------------- #
def backend() -> str:
    return "postgres" if config.DATABASE_URL else "sqlite"


def describe() -> str:
    """Where the data lives, without the password (health check, UI captions)."""
    if backend() == "sqlite":
        return f"sqlite:{config.DB_PATH}"
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", config.DATABASE_URL) + \
        (f" (schema {config.DB_SCHEMA})" if config.DB_SCHEMA else "")


_pools: dict[str, object] = {}
_pool_lock = threading.Lock()


def _pool():
    """One connection pool per process: a remote server is ~50-200 ms per new connection, a pooled one is free."""
    # A schema is a session setting, which Neon's pooled (PgBouncer) endpoint does not keep: use the direct URL then
    url = config.DATABASE_URL_DIRECT if config.DB_SCHEMA and config.DATABASE_URL_DIRECT else config.DATABASE_URL
    key = f"{url}|{config.DB_SCHEMA}"
    with _pool_lock:
        if key not in _pools:
            from psycopg_pool import ConnectionPool
            if not _pools:                             # close cleanly when a script or the server exits
                import atexit
                atexit.register(close_pools)
            _pools[key] = ConnectionPool(url, min_size=1, max_size=config.DB_POOL_SIZE, kwargs=_pg_kwargs(),
                                         check=ConnectionPool.check_connection, max_idle=300, open=True,
                                         name="rex-db")
        return _pools[key]


def _pg_kwargs() -> dict:
    from psycopg.rows import dict_row
    kwargs = {"row_factory": dict_row, "prepare_threshold": None}   # None: works behind PgBouncer poolers (Neon)
    if config.DB_SCHEMA:
        kwargs["options"] = f"-c search_path={config.DB_SCHEMA}"
    return kwargs


@contextmanager
def _direct() -> Iterator["_PgConn"]:
    """A one-off direct (unpooled) PostgreSQL connection for creating / migrating tables - Neon recommends the
    direct endpoint for schema changes. Falls back to DATABASE_URL when no direct URL is configured."""
    import psycopg
    with psycopg.connect(config.DATABASE_URL_DIRECT or config.DATABASE_URL, **_pg_kwargs()) as raw:
        yield _PgConn(raw)                         # commits when the block ends, rolls back on an error


def close_pools() -> None:
    with _pool_lock:
        for p in _pools.values():
            p.close()
        _pools.clear()


class _PgConn:
    """The few sqlite3.Connection methods db.py uses, on a psycopg connection (SQL translated on the fly)."""

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql: str, args=()):
        return self.raw.execute(_pg_sql(sql), tuple(args) if args else None)

    def executemany(self, sql: str, rows):
        rows = [tuple(r) for r in rows]
        cur = self.raw.cursor()
        if rows:
            cur.executemany(_pg_sql(sql), rows)
        return cur

    def executescript(self, script: str):
        self.raw.execute(_pg_schema(script))


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator:
    """A transaction: committed when the block ends, rolled back on an error. db_path only applies to SQLite."""
    if backend() == "postgres":
        with _pool().connection() as raw:          # commits on success, rolls back on an exception
            yield _PgConn(raw)
        return
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


@lru_cache(maxsize=1)
def tables() -> dict[str, dict]:
    """{table: {"columns": [...], "pk": [...]}} in schema.sql order (parents before children)."""
    out: dict[str, dict] = {}
    text = re.sub(r"--[^\n]*", "", SCHEMA_PATH.read_text(encoding="utf-8"))
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\);", text, re.S):
        cols, pk = [], []
        for part in (p.strip() for p in _split_top(body) if p.strip()):
            word = part.split()[0]
            if word.upper() in ("PRIMARY", "CHECK", "FOREIGN", "UNIQUE", "CONSTRAINT"):
                if word.upper() == "PRIMARY":
                    pk = [c.strip() for c in re.search(r"\((.*?)\)", part).group(1).split(",")]
                continue
            cols.append(word)
            if "PRIMARY KEY" in part.upper():
                pk = [word]
        out[name] = {"columns": cols, "pk": pk}
    return out


@lru_cache(maxsize=512)
def _pg_sql(sql: str) -> str:
    """SQLite statement -> PostgreSQL: ? placeholders, INSERT OR REPLACE / IGNORE, case-insensitive LIKE, rowid."""
    m = re.match(r"\s*INSERT OR (REPLACE|IGNORE) INTO (\w+)", sql, re.I)
    if m:
        table = m.group(2)
        sql = re.sub(r"INSERT OR (?:REPLACE|IGNORE) INTO", "INSERT INTO", sql, count=1, flags=re.I)
        if m.group(1).upper() == "IGNORE":
            sql += " ON CONFLICT DO NOTHING"
        else:
            info = tables()[table]
            update = [c for c in info["columns"] if c not in info["pk"]]
            sql += f" ON CONFLICT ({', '.join(info['pk'])}) DO " + \
                ("UPDATE SET " + ", ".join(f"{c}=EXCLUDED.{c}" for c in update) if update else "NOTHING")
    sql = re.sub(r",\s*rowid(\s+DESC)?\b", "", sql, flags=re.I)        # PostgreSQL has no rowid
    sql = re.sub(r"\bLIKE\b", "ILIKE", sql)                             # SQLite LIKE ignores case
    return sql.replace("?", "%s")


def _pg_schema(script: str) -> str:
    script = re.sub(r"^\s*PRAGMA[^\n]*\n", "", script, flags=re.M)
    script = re.sub(r"INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY", script)
    script = re.sub(r"\bREAL\b", "DOUBLE PRECISION", script)
    return re.sub(r"\bBLOB\b", "BYTEA", script)


def init_db(db_path: Optional[Path] = None, seed: bool = True) -> None:
    with (_direct() if backend() == "postgres" else connect(db_path)) as conn:
        if backend() == "postgres" and config.DB_SCHEMA:
            conn.raw.execute(f'CREATE SCHEMA IF NOT EXISTS "{config.DB_SCHEMA}"')
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)
        if seed:
            _seed(conn)


# Columns added after Sprint 1: CREATE TABLE IF NOT EXISTS does not add them to an existing database.
_ADDED_COLUMNS = {
    "rfps": {"source_title": "TEXT", "relevance_score": "REAL NOT NULL DEFAULT 0", "matched_course_ids": "TEXT",
             "priority": "TEXT NOT NULL DEFAULT 'normal'", "first_seen": "TEXT"},
    "documents": {"content": "TEXT", "request": "TEXT", "version": "INTEGER NOT NULL DEFAULT 1", "updated_at": "TEXT"},
    "campaigns": {"content": "TEXT", "document_id": "TEXT", "language": "TEXT", "version": "INTEGER NOT NULL DEFAULT 1",
                  "updated_at": "TEXT"},
    "chat_sessions": {"user_id": "TEXT"},
    "leads": {"user_id": "TEXT"},
}


def _migrate(conn) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        if backend() == "postgres":
            for name, decl in columns.items():
                conn.raw.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {_pg_schema(decl)}")
            continue
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _load(name: str) -> dict:
    return json.loads((config.DATA_DIR / name).read_text(encoding="utf-8"))


def _seed(conn) -> None:
    """Load data/*.json and data/data_sources.yaml into the database (idempotent; batched, so a remote server
    needs a few round trips, not one per row)."""
    reg = yaml.safe_load((config.DATA_DIR / "data_sources.yaml").read_text(encoding="utf-8"))
    conn.executemany("INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?,?,?,?,?)", [
        (s["source_id"], s["name"], category,
         json.dumps(s.get("country")) if isinstance(s.get("country"), list) else s.get("country"),
         s.get("url") or s.get("location"), s.get("access"), s["status"], s.get("owner"), None)
        for category, items in reg.items() for s in items])

    courses = _load("courses.json")
    meta = courses["_meta"]
    prices = {p["course_id"]: p for p in _load("pricing_promotions.json")["prices"]}
    rows = []
    for c in courses["courses"]:
        p = prices.get(c["course_id"], {})
        rows.append((c["course_id"], c["title_en"], c["title_ar"], c["department"], json.dumps(c["industry"]),
                     c["skill_level"], c["target_audience"], c["duration_hours"], c["duration_weeks"],
                     json.dumps(c["formats"]), json.dumps(c["language"]), c["description_en"], c["description_ar"],
                     json.dumps(c["learning_outcomes"]), json.dumps(c["modules"]), json.dumps(c["prerequisites"]),
                     json.dumps(c["career_opportunities"]), json.dumps(c["tags"]), c.get("certificate"),
                     c.get("accreditation_alignment"), json.dumps(c["country"]), p.get("fee"),
                     meta.get("currency", "EGP"), int(bool(p.get("installments"))), meta["source_id"],
                     meta["last_updated"], int(meta.get("status") == "sample")))
    conn.executemany("INSERT OR REPLACE INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     rows)

    conn.executemany("INSERT OR REPLACE INTO instructors VALUES (?,?,?,?,?,?,?)", [
        (i["instructor_id"], i["name"], i["title"], i["years_experience"], json.dumps(i["expertise"]), i["bio_en"],
         i["bio_ar"]) for i in _load("instructors.json")["instructors"]])
    conn.executemany("INSERT OR IGNORE INTO course_instructors VALUES (?,?)",
                     [(c["course_id"], iid) for c in courses["courses"] for iid in c["instructor_ids"]])
    conn.executemany("INSERT OR REPLACE INTO schedules VALUES (?,?,?,?,?,?,?,?)", [
        (s["schedule_id"], s["course_id"], s["start_date"], s["format"], s["days"], s["time"], s["location"],
         s["seats_left"]) for s in _load("schedules.json")["schedules"]])
    conn.executemany("INSERT OR REPLACE INTO promotions VALUES (?,?,?,?,?,?,?,?)", [
        (p["promo_id"], p["name_en"], p["name_ar"], p["discount_pct"],
         p["applies_to"] if isinstance(p["applies_to"], str) else json.dumps(p["applies_to"]),
         p["condition_en"], p["condition_ar"], p["valid_until"]) for p in _load("pricing_promotions.json")["promotions"]])
    conn.executemany("INSERT OR REPLACE INTO faqs VALUES (?,?,?,?,?,?)", [
        (f["faq_id"], f["category"], f["q_en"], f["a_en"], f["q_ar"], f["a_ar"]) for f in _load("faqs.json")["faqs"]])


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
            "INSERT INTO leads (lead_id, session_id, name, email, phone, interested_course_ids, profile, consent, "
            "consent_at, created_at, user_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (lead_id, lead.get("session_id"), lead["name"], lead.get("email"), lead.get("phone"),
             json.dumps(lead.get("interested_course_ids", [])),
             json.dumps(lead.get("profile") or {}, ensure_ascii=False), 1, now, now, lead.get("user_id")),
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
    except Exception:
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


def _rfp_row(row) -> dict:
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


# --------------------------------------------------------------------------- #
# Curriculum and accreditation alignment (owner: Member 2)
# --------------------------------------------------------------------------- #
def save_standards(rows: list[dict], db_path: Optional[Path] = None) -> None:
    """Framework competencies -> accreditation_standards (idempotent)."""
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO accreditation_standards VALUES (?,?,?,?,?,?,?)",
                         [(r["standard_id"], r["framework"], r.get("country"), r["competency"], r.get("description"),
                           r["source_id"], r.get("last_checked")) for r in rows])


def list_standards(framework: Optional[str] = None, db_path: Optional[Path] = None) -> list[dict]:
    sql, args = "SELECT * FROM accreditation_standards", []
    if framework:
        sql, args = sql + " WHERE framework=?", [framework]
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY standard_id", args)]


def save_curriculum(cur: dict, request: dict, change_note: Optional[str] = None, changed_by: Optional[str] = None,
                    db_path: Optional[Path] = None) -> None:
    """Insert or update a curriculum and keep a copy of this version in curriculum_versions."""
    meta = cur.get("metadata") or {}
    now = _now()
    content = json.dumps(cur, ensure_ascii=False, default=str)
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO curricula (curriculum_id, course_id, course_title, request, content, department,
                 target_audience, skill_level, industry, duration_hours, accreditation_framework, country, status,
                 version, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(curriculum_id) DO UPDATE SET course_title=excluded.course_title, content=excluded.content,
                 department=excluded.department, target_audience=excluded.target_audience,
                 skill_level=excluded.skill_level, industry=excluded.industry, duration_hours=excluded.duration_hours,
                 accreditation_framework=excluded.accreditation_framework, country=excluded.country,
                 status=excluded.status, version=excluded.version, updated_at=excluded.updated_at""",
            (cur["curriculum_id"], cur.get("course_id"), cur["course_title"],
             json.dumps(request, ensure_ascii=False, default=str), content, meta.get("department"),
             cur.get("target_audience"), meta.get("skill_level"), meta.get("industry"), cur.get("training_hours"),
             meta.get("accreditation_framework"), meta.get("country"), cur["status"], cur["version"], now, now))
        conn.execute("INSERT OR REPLACE INTO curriculum_versions VALUES (?,?,?,?,?,?)",
                     (cur["curriculum_id"], cur["version"], content, change_note, changed_by, now))


def _curriculum_row(row) -> dict:
    r = dict(row)
    r["content"] = json.loads(r["content"])
    r["request"] = json.loads(r["request"])
    return r


def get_curriculum(curriculum_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM curricula WHERE curriculum_id=?", (curriculum_id,)).fetchone()
        return _curriculum_row(row) if row else None


def list_curricula(department: Optional[str] = None, skill_level: Optional[str] = None,
                   country: Optional[str] = None, status: Optional[str] = None, framework: Optional[str] = None,
                   industry: Optional[str] = None, audience: Optional[str] = None, q: Optional[str] = None,
                   min_hours: Optional[float] = None, max_hours: Optional[float] = None, limit: int = 200,
                   db_path: Optional[Path] = None) -> list[dict]:
    """Metadata filtering (brief: department, audience, level, industry, duration, framework, country) + free text."""
    sql, args = "SELECT * FROM curricula WHERE 1=1", []
    for col, val in (("department", department), ("skill_level", skill_level), ("country", country),
                     ("status", status)):
        if val:
            sql += f" AND lower({col})=lower(?)"
            args.append(val)
    for col, val in (("accreditation_framework", framework), ("industry", industry), ("target_audience", audience)):
        if val:
            sql += f" AND {col} LIKE ?"
            args.append(f"%{val}%")
    if min_hours is not None:
        sql += " AND duration_hours >= ?"
        args.append(min_hours)
    if max_hours is not None:
        sql += " AND duration_hours <= ?"
        args.append(max_hours)
    if q:
        sql += " AND (course_title LIKE ? OR content LIKE ?)"
        args += [f"%{q}%"] * 2
    sql += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with connect(db_path) as conn:
        return [_curriculum_row(r) for r in conn.execute(sql, args)]


def list_curriculum_versions(curriculum_id: str, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM curriculum_versions WHERE curriculum_id=? ORDER BY version",
                            (curriculum_id,))
        return [{**dict(r), "content": json.loads(r["content"])} for r in rows]


def save_gap_report(report: dict, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO gap_reports VALUES (?,?,?,?,?)",
                     (report["report_id"], report["curriculum_id"], report["framework"],
                      json.dumps(report, ensure_ascii=False, default=str), _now()))


def list_gap_reports(curriculum_id: str, db_path: Optional[Path] = None) -> list[dict]:
    """Newest first."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT content FROM gap_reports WHERE curriculum_id=? "
                            "ORDER BY created_at DESC, rowid DESC", (curriculum_id,))
        return [json.loads(r["content"]) for r in rows]


def save_review(object_type: str, object_id: str, action: str, reviewer: str, comment: Optional[str] = None,
                db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT INTO reviews (object_type, object_id, action, reviewer, comment, created_at) "
                     "VALUES (?,?,?,?,?,?)", (object_type, object_id, action, reviewer, comment, _now()))


def list_reviews(object_type: str, object_id: str, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM reviews WHERE object_type=? AND object_id=? ORDER BY review_id",
                            (object_type, object_id))
        return [dict(r) for r in rows]


def save_handoff(from_agent: str, to_agent: str, payload_type: str, payload_id: str,
                 db_path: Optional[Path] = None) -> dict:
    """Agent-to-agent handoff (docs/interfaces.md §4), e.g. approved curriculum -> Document Agent."""
    event = {"event_id": "EVT-" + uuid.uuid4().hex[:10], "from_agent": from_agent, "to_agent": to_agent,
             "payload_type": payload_type, "payload_id": payload_id, "created_at": _now()}
    with connect(db_path) as conn:
        conn.execute("INSERT INTO handoff_events VALUES (?,?,?,?,?,?)", tuple(event.values()))
    return event


def list_handoffs(payload_id: Optional[str] = None, db_path: Optional[Path] = None) -> list[dict]:
    sql, args = "SELECT * FROM handoff_events", []
    if payload_id:
        sql, args = sql + " WHERE payload_id=?", [payload_id]
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY created_at, rowid", args)]


# --------------------------------------------------------------------------- #
# Generated documents (owner: Member 3)
# --------------------------------------------------------------------------- #
def save_document(doc: dict, request: dict, change_note: Optional[str] = None, changed_by: Optional[str] = None,
                  db_path: Optional[Path] = None) -> None:
    """Insert or update a document and keep a copy of this version in document_versions."""
    now = _now()
    content = json.dumps(doc, ensure_ascii=False, default=str)
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO documents (document_id, doc_type, title, course_id, curriculum_id, language, file_path,
                 status, sources, emailed_to, created_at, content, request, version, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(document_id) DO UPDATE SET title=excluded.title, file_path=excluded.file_path,
                 status=excluded.status, sources=excluded.sources, emailed_to=excluded.emailed_to,
                 content=excluded.content, version=excluded.version, updated_at=excluded.updated_at""",
            (doc["document_id"], doc["doc_type"], doc["title"], doc.get("course_id"), doc.get("curriculum_id"),
             doc["language"], doc["file_path"], doc["status"], json.dumps(doc.get("sources") or [], default=str),
             json.dumps(doc.get("emailed_to") or []), doc.get("created_at") or now, content,
             json.dumps(request, ensure_ascii=False, default=str), doc.get("version", 1), now))
        conn.execute("INSERT OR REPLACE INTO document_versions VALUES (?,?,?,?,?,?)",
                     (doc["document_id"], doc.get("version", 1), content, change_note, changed_by, now))


def _document_row(row) -> dict:
    r = dict(row)
    r["content"] = json.loads(r["content"]) if r.get("content") else None
    r["request"] = json.loads(r["request"]) if r.get("request") else {}
    return r


def get_document(document_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        return _document_row(row) if row else None


def list_documents(doc_type: Optional[str] = None, status: Optional[str] = None, course_id: Optional[str] = None,
                   curriculum_id: Optional[str] = None, language: Optional[str] = None, q: Optional[str] = None,
                   limit: int = 200, db_path: Optional[Path] = None) -> list[dict]:
    """Newest first. Rows from Sprint 1 without content are skipped (nothing to show or edit)."""
    sql, args = "SELECT * FROM documents WHERE content IS NOT NULL", []
    for col, val in (("doc_type", doc_type), ("status", status), ("course_id", course_id),
                     ("curriculum_id", curriculum_id), ("language", language)):
        if val:
            sql += f" AND {col}=?"
            args.append(val)
    if q:
        sql += " AND (title LIKE ? OR content LIKE ?)"
        args += [f"%{q}%"] * 2
    sql += " ORDER BY coalesce(updated_at, created_at) DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with connect(db_path) as conn:
        return [_document_row(r) for r in conn.execute(sql, args)]


def list_document_versions(document_id: str, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM document_versions WHERE document_id=? ORDER BY version", (document_id,))
        return [{**dict(r), "content": json.loads(r["content"])} for r in rows]


def lead_interest(course_id: str, db_path: Optional[Path] = None) -> list[dict]:
    """Profiles of consented leads interested in a course - no names or contact details (Marketing Agent)."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT interested_course_ids, profile, created_at FROM leads "
                            "WHERE interested_course_ids LIKE ?", (f'%"{course_id}"%',))
        return [{"profile": json.loads(r["profile"] or "{}"), "created_at": r["created_at"]} for r in rows]


# --------------------------------------------------------------------------- #
# Marketing campaigns (owner: Member 1)
# --------------------------------------------------------------------------- #
def save_campaign(camp: dict, request: dict, change_note: Optional[str] = None, changed_by: Optional[str] = None,
                  db_path: Optional[Path] = None) -> None:
    """Insert or update a campaign, rewrite its current variants and keep a copy of this version."""
    now = _now()
    content = json.dumps(camp, ensure_ascii=False, default=str)
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO campaigns (campaign_id, course_id, curriculum_id, request, strategy, course_facts_used,
                 status, created_at, content, document_id, language, version, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(campaign_id) DO UPDATE SET strategy=excluded.strategy,
                 course_facts_used=excluded.course_facts_used, status=excluded.status, content=excluded.content,
                 version=excluded.version, updated_at=excluded.updated_at""",
            (camp["campaign_id"], camp["course_id"], camp.get("curriculum_id"),
             json.dumps(request, ensure_ascii=False, default=str), camp["strategy"],
             json.dumps(camp["course_facts_used"], ensure_ascii=False, default=str), camp["status"],
             camp.get("created_at") or now, content, camp.get("document_id"), camp.get("language"),
             camp.get("version", 1), now))
        conn.execute("DELETE FROM campaign_variants WHERE campaign_id=?", (camp["campaign_id"],))
        conn.executemany("INSERT INTO campaign_variants (campaign_id, channel, variant_no, headline, body, cta) "
                         "VALUES (?,?,?,?,?,?)",
                         [(camp["campaign_id"], v["channel"], v["variant_no"], v.get("headline"), v["body"], v["cta"])
                          for v in camp["variants"]])
        conn.execute("INSERT OR REPLACE INTO campaign_versions VALUES (?,?,?,?,?,?)",
                     (camp["campaign_id"], camp.get("version", 1), content, change_note, changed_by, now))


def _campaign_row(row) -> dict:
    r = dict(row)
    r["content"] = json.loads(r["content"]) if r.get("content") else None
    r["request"] = json.loads(r["request"]) if r.get("request") else {}
    return r


def get_campaign(campaign_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
        return _campaign_row(row) if row else None


def list_campaigns(course_id: Optional[str] = None, status: Optional[str] = None, curriculum_id: Optional[str] = None,
                   language: Optional[str] = None, q: Optional[str] = None, limit: int = 200,
                   db_path: Optional[Path] = None) -> list[dict]:
    """Newest first. Rows without content (written before Sprint 2) are skipped."""
    sql, args = "SELECT * FROM campaigns WHERE content IS NOT NULL", []
    for col, val in (("course_id", course_id), ("status", status), ("curriculum_id", curriculum_id),
                     ("language", language)):
        if val:
            sql += f" AND {col}=?"
            args.append(val)
    if q:
        sql += " AND (content LIKE ? OR strategy LIKE ?)"
        args += [f"%{q}%"] * 2
    sql += " ORDER BY coalesce(updated_at, created_at) DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with connect(db_path) as conn:
        return [_campaign_row(r) for r in conn.execute(sql, args)]


def list_campaign_versions(campaign_id: str, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM campaign_versions WHERE campaign_id=? ORDER BY version", (campaign_id,))
        return [{**dict(r), "content": json.loads(r["content"])} for r in rows]


def list_handoffs_to(to_agent: str, db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM handoff_events WHERE to_agent=? ORDER BY created_at DESC, rowid DESC",
                            (to_agent,))
        return [dict(r) for r in rows]


def list_messages(session_id: str, limit: int = 500, db_path: Optional[Path] = None) -> list[dict]:
    """A whole conversation, oldest first (the chat page restores it from the database)."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT role, content, sources, created_at FROM chat_messages WHERE session_id=? "
                            "ORDER BY message_id LIMIT ?", (session_id, limit)).fetchall()
        return [{**dict(r), "sources": json.loads(r["sources"] or "[]")} for r in rows]


def claim_session(session_id: str, user_id: str, db_path: Optional[Path] = None) -> bool:
    """Make user_id the owner of a conversation (new, or from before accounts). False if another account owns it."""
    now = _now()
    with connect(db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO chat_sessions (session_id, created_at, updated_at, user_id) "
                     "VALUES (?,?,?,?)", (session_id, now, now, user_id))
        conn.execute("UPDATE chat_sessions SET user_id=? WHERE session_id=? AND user_id IS NULL", (user_id, session_id))
        row = conn.execute("SELECT user_id FROM chat_sessions WHERE session_id=?", (session_id,)).fetchone()
        return row["user_id"] == user_id


def session_owner(session_id: str, db_path: Optional[Path] = None) -> Optional[str]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT user_id FROM chat_sessions WHERE session_id=?", (session_id,)).fetchone()
        return row["user_id"] if row else None


def list_sessions(user_id: str, limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    """One account's conversations, newest first, titled by their first message."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT s.session_id, s.language, s.stage, s.created_at, s.updated_at,
                      (SELECT content FROM chat_messages m WHERE m.session_id = s.session_id AND m.role = 'user'
                       ORDER BY m.message_id LIMIT 1) AS title,
                      (SELECT count(*) FROM chat_messages m WHERE m.session_id = s.session_id) AS messages
               FROM chat_sessions s WHERE s.user_id = ?
                 AND EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.session_id)
               ORDER BY s.updated_at DESC LIMIT ?""", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Accounts and login tokens (rex_ai/core/auth.py)
# --------------------------------------------------------------------------- #
def create_user(user: dict, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT INTO users (user_id, email, name, role, password_hash, created_at) VALUES (?,?,?,?,?,?)",
                     (user["user_id"], user["email"], user["name"], user["role"], user["password_hash"],
                      user["created_at"]))


def get_user(user_id: Optional[str] = None, email: Optional[str] = None, db_path: Optional[Path] = None) -> Optional[dict]:
    col, val = ("user_id", user_id) if user_id else ("email", (email or "").lower())
    with connect(db_path) as conn:
        row = conn.execute(f"SELECT * FROM users WHERE {col}=?", (val,)).fetchone()
        return dict(row) if row else None


def list_users(db_path: Optional[Path] = None) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT user_id, email, name, role, created_at, last_login_at FROM users ORDER BY created_at")]


def count_users(role: Optional[str] = None, db_path: Optional[Path] = None) -> int:
    sql, args = "SELECT count(*) AS n FROM users", ()
    if role:
        sql, args = sql + " WHERE role=?", (role,)
    with connect(db_path) as conn:
        return conn.execute(sql, args).fetchone()["n"]


def set_user_role(user_id: str, role: str, db_path: Optional[Path] = None) -> bool:
    with connect(db_path) as conn:
        return conn.execute("UPDATE users SET role=? WHERE user_id=?", (role, user_id)).rowcount > 0


def touch_login(user_id: str, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE users SET last_login_at=? WHERE user_id=?", (_now(), user_id))


def save_token(token_hash: str, user_id: str, expires_at: str, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (_now(),))          # tidy up expired logins
        conn.execute("INSERT INTO auth_tokens VALUES (?,?,?,?)", (token_hash, user_id, _now(), expires_at))


def user_for_token(token_hash: str, db_path: Optional[Path] = None) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT u.* FROM auth_tokens t JOIN users u ON u.user_id = t.user_id "
                           "WHERE t.token_hash=? AND t.expires_at > ?", (token_hash, _now())).fetchone()
        return dict(row) if row else None


def delete_token(token_hash: str, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash=?", (token_hash,))


# --------------------------------------------------------------------------- #
# Stored data files (rex_ai/core/filestore.py)
# --------------------------------------------------------------------------- #
def file_index(db_path: Optional[Path] = None) -> dict[str, str]:
    """{path: sha256} of every stored file."""
    with connect(db_path) as conn:
        return {r["path"]: r["sha256"] for r in conn.execute("SELECT path, sha256 FROM stored_files")}


def put_file(path: str, content: bytes, sha256: str, db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO stored_files VALUES (?,?,?,?,?)",
                     (path, content, sha256, len(content), _now()))


def get_file(path: str, db_path: Optional[Path] = None) -> Optional[bytes]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT content FROM stored_files WHERE path=?", (path,)).fetchone()
        return bytes(row["content"]) if row else None


# --------------------------------------------------------------------------- #
# Status and migration (python -m rex_ai.core.db ...)
# --------------------------------------------------------------------------- #
def counts(db_path: Optional[Path] = None) -> dict[str, int]:
    with connect(db_path) as conn:
        return {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in tables()}


def migrate_from_sqlite(sqlite_path: Path) -> dict[str, int]:
    """Copy every row of a local SQLite database into the current (shared) database. Rows that already exist are
    kept as they are, so it is safe to run twice. Returns rows copied per table."""
    if backend() != "postgres":
        raise RuntimeError("Set REX_DATABASE_URL to the shared PostgreSQL database first")
    init_db()
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    copied: dict[str, int] = {}
    try:
        existing_tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        with _direct() as conn:
            for table, info in tables().items():          # schema order: parents before children
                if table not in existing_tables:
                    continue
                have = {r["name"] for r in src.execute(f"PRAGMA table_info({table})")}
                serial = _SERIAL_TABLES.get(table)
                cols = [c for c in info["columns"] if c in have and c != serial]
                rows = [tuple(r[c] for c in cols) for r in src.execute(f"SELECT {', '.join(cols)} FROM {table}")]
                if rows and serial:
                    # auto-numbered rows (messages, reviews, logs...): the old numbers may already be taken in the
                    # shared database, so new ones are given and a row counts as present when every column matches
                    same = " AND ".join(f"{c} IS NOT DISTINCT FROM ?" for c in cols)
                    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join('?' * len(cols))} "
                                     f"WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE {same})",
                                     [row + row for row in rows])
                elif rows:
                    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
                                     f" ON CONFLICT DO NOTHING", rows)
                copied[table] = len(rows)
    finally:
        src.close()
    return copied


_SERIAL_TABLES = {"chat_messages": "message_id", "reviews": "review_id", "campaign_variants": "variant_id",
                  "app_logs": "log_id"}                  # table -> auto-numbered key


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="REX AI database: status, migration of a local SQLite history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="which database is used and how many rows each table holds")
    sub.add_parser("init", help="create the tables and load the academy data")
    m = sub.add_parser("migrate", help="copy a local SQLite database into the shared database")
    m.add_argument("--from", dest="source", default=str(config.ROOT_DIR / "rex_ai.db"))
    sub.add_parser("push-files", help="store this checkout's data files in the shared database, replacing the "
                                      "stored copies (use after editing data/*.json by hand)")
    args = ap.parse_args(argv)
    print(f"Database: {describe()}")
    init_db(seed=args.cmd == "init")
    if args.cmd == "migrate":
        for table, n in migrate_from_sqlite(Path(args.source)).items():
            print(f"  {table:<24} {n} row(s) copied")
    elif args.cmd == "push-files":
        from rex_ai.core import filestore
        print(f"  {filestore.push()} file(s) stored")
        return
    for table, n in counts().items():
        print(f"  {table:<24} {n}")


if __name__ == "__main__":
    _main()
