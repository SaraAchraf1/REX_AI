"""Merge academy catalogue files (PDF, Word, Excel, CSV, TXT) into REX's course data.

    python scripts/import_catalog.py                       # every new/changed file in data/catalog/
    python scripts/import_catalog.py path/to/file.pdf      # one file
    python scripts/import_catalog.py --dry-run             # check only

Always merges: existing courses are kept, a course ID that already exists is updated. The catalogue format
and details are in rex_ai/core/catalog_import.py. The API also runs this on startup for data/catalog/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rex_ai.core.catalog_import import *  # noqa: E402,F401,F403
from rex_ai.core.catalog_import import _int, _match, _split, main  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())