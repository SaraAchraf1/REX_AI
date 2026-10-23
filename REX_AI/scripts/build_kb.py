"""Validates the course data and exports the chatbot knowledge base to data/kb/kb_chunks.jsonl.

Run from the repo root:  python scripts/build_kb.py
- Loads every data file through the Pydantic models, so a bad field fails here, not in the demo.
- Writes one JSON line per chunk with its metadata (department, skill level, audience, industry,
  duration, country, accreditation alignment, source_id, last_updated). A vector store
  (Chroma/FAISS) can ingest this file directly in Sprint 2.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rex_ai.agents.rex.knowledge_base import load_knowledge_base  # noqa: E402

kb = load_knowledge_base()
out = ROOT / "data" / "kb" / "kb_chunks.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for c in kb.chunks:
        f.write(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata},
                           ensure_ascii=False, default=str) + "\n")

print(f"Data status: {kb.meta.get('status')}")
print(f"{len(kb.courses)} courses | {len(kb.instructors)} instructors | {len(kb.schedules)} cohorts | "
      f"{len(kb.promotions)} promotions | {len(kb.faqs)} FAQs")
print(f"{len(kb.chunks)} chunks -> {out.relative_to(ROOT)}")
for dept, n in sorted(Counter(c.department for c in kb.courses.values()).items()):
    print(f"  {dept:<26} {n} course(s)")
