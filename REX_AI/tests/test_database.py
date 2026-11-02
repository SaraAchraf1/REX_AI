"""Shared database (owner: Member 3). Run: pytest -q tests/test_database.py

Every action is stored through rex_ai/core/db.py, on PostgreSQL when REX_DATABASE_URL is set, else SQLite. These tests
check that every SQL statement the agents run translates to valid PostgreSQL (parsed with PostgreSQL's own grammar via
pglast), the file store, chat history and the migration command. To run the WHOLE suite on a real PostgreSQL server:

    $env:REX_TEST_DATABASE_URL="postgresql://user:password@host:5432/dbname"; pytest -q
"""
import re
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from rex_ai import config
from rex_ai.core import db, filestore

pglast = pytest.importorskip("pglast", reason="pip install pglast to check the PostgreSQL translation")


def _parse_pg(sql: str) -> None:
    n = iter(range(1, 10_000))
    pglast.parse_sql(re.sub(r"%s", lambda _: f"${next(n)}", sql))


@pytest.fixture(scope="module")
def recorded_sql():
    """Every distinct SQL statement run while the API serves a flow through all five agents."""
    seen: set[str] = set()
    original = db.connect

    class Recorder:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, args=()):
            seen.add(sql)
            return self.conn.execute(sql, args)

        def executemany(self, sql, rows):
            seen.add(sql)
            return self.conn.executemany(sql, rows)

        def executescript(self, sql):
            return self.conn.executescript(sql)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    @contextmanager
    def recording(db_path=None):
        with original(db_path) as conn:
            yield Recorder(conn)

    db.connect = recording
    try:
        from rex_ai.api.main import app
        c = TestClient(app)
        c.post("/chat", json={"session_id": "db-1", "message": "I want to learn data analysis, beginner, online"})
        c.post("/leads", json={"session_id": "db-1", "name": "Sara", "email": "s@example.com",
                               "interested_course_ids": ["DS-101"], "consent": True})
        c.get("/chat/db-1/history")
        c.post("/rfp/run", json={})
        c.get("/rfp", params={"q": "training", "open_only": True})
        c.get("/rfp/alerts")
        cur = c.post("/curriculum", json={"course_title": "Software Testing Fundamentals", "skill_level": "Intermediate",
                                          "target_audience": "Junior developers", "total_hours": 30}).json()
        cid = cur["curriculum_id"]
        c.get("/curriculum", params={"department": "Software", "q": "testing", "min_hours": 1, "framework": "ISTQB"})
        c.post(f"/curriculum/{cid}/revise", json={"reviewer": "QA", "instructions": "set hours to 32"})
        c.post(f"/curriculum/{cid}/review", json={"action": "approved", "reviewer": "Dean"})
        c.get(f"/curriculum/{cid}/history")
        doc = c.post("/documents", json={"doc_type": "course_description", "curriculum_id": cid}).json()
        c.patch(f"/documents/{doc['document_id']}", json={"reviewer": "QA", "title": "Edited"})
        c.post(f"/documents/{doc['document_id']}/review", json={"action": "approved", "reviewer": "Dean"})
        c.get("/documents", params={"q": "Edited", "doc_type": "course_description"})
        c.get("/documents/handoffs")
        camp = c.post("/campaigns", json={"course_id": "DS-101"})
        if camp.status_code == 201:
            camp_id = camp.json()["campaign_id"]
            c.post(f"/campaigns/{camp_id}/review", json={"action": "approved", "reviewer": "Dean"})
            c.get(f"/campaigns/{camp_id}/history")
        c.get("/campaigns", params={"q": "data"})
        from rex_ai.core import auth
        u = auth.signup("Rec", f"rec-{id(seen)}@academy.example", "correct-horse")
        _, token = auth.login(u["email"], "correct-horse")
        auth.user_for_token(token)
        auth.set_role(u["user_id"], "user")
        db.count_users("admin")
        db.list_users()
        db.claim_session("rec-chat", u["user_id"])
        db.session_owner("rec-chat")
        db.list_sessions(u["user_id"])
        auth.logout(token)
        db.put_file("catalog/x.txt", b"x", filestore._sha(b"x"))
        db.file_index()
        db.get_file("catalog/x.txt")
        db.counts()
        with original() as conn:
            conn.execute("DELETE FROM stored_files")
    finally:
        db.connect = original
    return seen


def test_every_statement_translates_to_valid_postgresql(recorded_sql):
    statements = [s for s in recorded_sql if not s.lstrip().upper().startswith("PRAGMA")]
    assert len(statements) > 40                                         # the flow really touched every agent
    for sql in statements:
        pg = db._pg_sql(sql)
        assert "?" not in pg and "INSERT OR" not in pg.upper() and "rowid" not in pg.lower(), pg
        _parse_pg(pg)


def test_schema_translates_to_valid_postgresql():
    script = db._pg_schema(db.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert "AUTOINCREMENT" not in script and "PRAGMA" not in script and "BYTEA" in script
    pglast.parse_sql(script)
    for table, columns in db._ADDED_COLUMNS.items():
        for name, decl in columns.items():
            pglast.parse_sql(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {db._pg_schema(decl)}")


def test_upserts_use_the_primary_key():
    t = db.tables()
    assert t["courses"]["pk"] == ["course_id"] and len(t["courses"]["columns"]) == 27
    assert t["curriculum_versions"]["pk"] == ["curriculum_id", "version"]
    assert "target_audience" in t["curricula"]["columns"]                # several columns on one line
    sql = db._pg_sql("INSERT OR REPLACE INTO curriculum_versions VALUES (?,?,?,?,?,?)")
    assert "ON CONFLICT (curriculum_id, version) DO UPDATE SET content=EXCLUDED.content" in sql
    assert db._pg_sql("INSERT OR IGNORE INTO course_instructors VALUES (?,?)").endswith("ON CONFLICT DO NOTHING")
    assert db._pg_sql("SELECT * FROM rfps WHERE title LIKE ? ORDER BY created_at DESC, rowid DESC") == \
        "SELECT * FROM rfps WHERE title ILIKE %s ORDER BY created_at DESC"


def test_backend_selection_and_password_is_hidden(monkeypatch):
    assert db.backend() == ("postgres" if config.DATABASE_URL else "sqlite")
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://rex:s3cret@db.example.com:5432/rex")
    assert db.backend() == "postgres"
    assert "s3cret" not in db.describe() and "rex:***@db.example.com" in db.describe()


def test_chat_history_is_reloaded_from_the_database(chat):
    chat("Hello, I want a data course", session_id="hist-1")
    chat("I am a beginner", session_id="hist-1")
    msgs = db.list_messages("hist-1")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "Hello, I want a data course"
    from rex_ai.api.main import app
    body = TestClient(app).get("/chat/hist-1/history").json()
    assert body["found"] and len(body["messages"]) == 4
    assert TestClient(app).get("/chat/nope-123/history").json() == {
        "session_id": "nope-123", "found": False, "language": None, "stage": None, "messages": []}


def test_uploaded_files_are_shared_between_servers(monkeypatch, tmp_path):
    """Server A stores its uploads in the database; server B (another data folder) restores them."""
    monkeypatch.setattr(config, "SHARED_FILES", True)
    server_a = config.DATA_DIR
    upload = server_a / "catalog" / "shared_test_upload.csv"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_text("course_id,title_en\nZZ-901,Shared course\n", encoding="utf-8")
    try:
        assert filestore.push() >= 1
        assert filestore.push() == 0                                    # unchanged files are not stored again
        stored = db.file_index()
        assert "catalog/shared_test_upload.csv" in stored and "courses.json" in stored
        assert not any(p.startswith("backup/") for p in stored)

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)              # server B: empty data folder
        assert filestore.pull(force=True) == len(stored)
        assert (tmp_path / "catalog" / "shared_test_upload.csv").read_text(encoding="utf-8").startswith("course_id")
        assert filestore.pull(force=True) == 0                          # already up to date
        assert filestore.pull() == 0                                    # throttled
        (tmp_path / "courses.json").write_text("{}", encoding="utf-8")  # a stale local copy is replaced
        assert filestore.pull(force=True) == 1 and (tmp_path / "courses.json").stat().st_size > 2

        db.put_file("../outside.txt", b"x", filestore._sha(b"x"))       # never written outside data/
        filestore.pull(force=True)
        assert not (tmp_path.parent / "outside.txt").exists()
    finally:
        upload.unlink(missing_ok=True)
        with db.connect() as conn:
            conn.execute("DELETE FROM stored_files")


def test_file_store_is_off_without_a_shared_database():
    assert config.SHARED_FILES is False
    assert filestore.push() == 0 and filestore.pull(force=True) == 0


def test_migrate_requires_postgres():
    if db.backend() == "postgres":
        pytest.skip("covered by test_migrate_sqlite_history_into_postgres")
    with pytest.raises(RuntimeError, match="REX_DATABASE_URL"):
        db.migrate_from_sqlite(config.DB_PATH)


@pytest.mark.skipif(not config.DATABASE_URL, reason="needs REX_TEST_DATABASE_URL (a PostgreSQL server)")
def test_migrate_sqlite_history_into_postgres(tmp_path, monkeypatch):
    src = tmp_path / "old.db"
    url = config.DATABASE_URL
    monkeypatch.setattr(config, "DATABASE_URL", "")
    db.init_db(src)
    db.save_session("old-1", "en", {}, "discovery", db_path=src)
    db.add_message("old-1", "user", "hello from the old laptop", db_path=src)
    db.save_review("curriculum", "CUR-OLD", "approved", "Dean", db_path=src)
    monkeypatch.setattr(config, "DATABASE_URL", url)
    copied = db.migrate_from_sqlite(src)
    assert copied["chat_messages"] == 1 and copied["reviews"] == 1
    assert db.list_messages("old-1")[0]["content"] == "hello from the old laptop"
    assert db.migrate_from_sqlite(src)["chat_messages"] == 1           # safe to run twice
    assert len(db.list_messages("old-1")) == 1
    db.save_review("curriculum", "CUR-NEW", "approved", "Dean")        # numbering continues after copied ids
