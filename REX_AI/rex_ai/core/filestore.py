"""Data files that the app adds or changes, kept in the shared database so every server has them (owner: Member 3).

Uploads land on the disk of the server that received them (data/catalog/, data/documents/, data/rfp/...), and a
catalogue merge rewrites data/*.json there. With a shared database (REX_SHARED_FILES, on by default when
REX_DATABASE_URL is set):

  push()   after every upload / merge: files whose content differs from the stored copy are stored
  pull()   on start-up and then at most every REX_FILE_SYNC_SECONDS: stored files missing or different on this
           server are written to its data/ folder, so the next request merges / indexes them as usual

The database copy wins. To publish data files edited by hand in a checkout, run
`python -m rex_ai.core.db push-files`.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

from rex_ai import config
from rex_ai.core import db

log = logging.getLogger("rex.files")

# relative to data/: what uploads and merges write
MIRRORED = ("*.json", "catalog/**/*", "documents/**/*", "rfp/feeds/**/*", "rfp/documents/**/*")
SKIP_PARTS = {"backup", "__pycache__"}
_lock = threading.Lock()
_last_pull = 0.0


def enabled() -> bool:
    return config.SHARED_FILES


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_files() -> dict[str, Path]:
    root = config.DATA_DIR
    out = {}
    for pattern in MIRRORED:
        for p in root.glob(pattern):
            rel = p.relative_to(root)
            if p.is_file() and not p.name.startswith("~$") and not SKIP_PARTS & set(rel.parts):
                out[rel.as_posix()] = p
    return out


def push() -> int:
    """Store the local files whose content differs from the database copy. Returns how many were stored."""
    if not enabled():
        return 0
    stored = db.file_index()
    n = 0
    for rel, path in local_files().items():
        data = path.read_bytes()
        sha = _sha(data)
        if stored.get(rel) != sha:
            db.put_file(rel, data, sha)
            n += 1
    if n:
        log.info("Stored %d data file(s) in the shared database", n)
    return n


def pull(force: bool = False) -> int:
    """Write stored files that are missing or different here. Throttled; never raises. Returns files written."""
    global _last_pull
    if not enabled() or (not force and time.monotonic() - _last_pull < config.FILE_SYNC_SECONDS):
        return 0
    with _lock:
        _last_pull = time.monotonic()
        try:
            stored = db.file_index()
        except Exception as exc:                          # e.g. tables not created yet
            log.warning("Could not read stored files: %s", exc)
            return 0
        local = local_files()
        n = 0
        root = config.DATA_DIR.resolve()
        for rel, sha in stored.items():
            target = (config.DATA_DIR / rel).resolve()
            if root not in target.parents:                # never write outside data/
                continue
            if rel in local and _sha(local[rel].read_bytes()) == sha:
                continue
            data = db.get_file(rel)
            if data is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            n += 1
        if n:
            log.info("Restored %d data file(s) from the shared database", n)
        return n
