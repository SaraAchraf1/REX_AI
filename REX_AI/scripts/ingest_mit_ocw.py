"""Download the MIT OCW 6.7960 Deep Learning (Fall 2024) lecture notes and ingest them into the shared RAG.

Run from the repo root:
    python scripts/ingest_mit_ocw.py                      # all lectures -> collection "curriculum_refs"
    python scripts/ingest_mit_ocw.py --lectures 1 8 21    # only some lectures
    python scripts/ingest_mit_ocw.py --list-url <another OCW lecture-notes list page>

The scraper reads the OCW "lecture notes" list page, follows every PDF link (direct .pdf links or
/resources/... pages), downloads the PDFs to data/raw/<course>/ (skipped if already there) and ingests
them with metadata so the Curriculum, Document and Marketing agents can filter and cite them.
Content: MIT OpenCourseWare, CC BY-NC-SA 4.0 - fine for this academic prototype, cite the source.
"""
import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rex_ai.core.rag import get_rag  # noqa: E402

DEFAULT_URL = "https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/lists/lecture-notes/"
# Browser-like headers: some networks/CDNs reset connections from unknown user agents.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36 REX-AI-graduation-project",
    "Accept": "text/html,application/pdf,*/*",
}
COURSE_BASE = "https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024"
# Fallback when the list page can't be reached (lectures 22 and 25 are not published by MIT).
KNOWN_LECTURES = {
    1: "Introduction to Deep Learning", 2: "How to Train a Neural Net", 3: "Approximation Theory",
    4: "Architectures for Grids", 5: "Graph Neural Networks", 6: "Neural Network Generalization",
    7: "Scaling Rules for Optimization", 8: "Transformers", 9: "Hacker's Guide to Deep Learning",
    10: "Memory and Sequence Modeling", 11: "Representation Learning I",
    12: "Similarity-Based Representation Learning", 13: "Architectural Bias on Representations",
    14: "Deep Generative Models I", 15: "Deep Generative Models II", 16: "Deep Generative Models III",
    17: "Out-of-Distribution Generalization", 18: "Transfer Learning I", 19: "Transfer Learning II",
    20: "Scaling Laws", 21: "Language Models", 23: "Metrized Deep Learning",
    24: "Inference Methods for Deep Learning",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def http_get(url: str, retries: int = 4, **kw) -> requests.Response:
    """GET with retries and backoff (connection resets / timeouts are common on large OCW files)."""
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=kw.pop("timeout", 60), **kw)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  network error ({exc.__class__.__name__}), retry {attempt}/{retries - 1} in {wait}s...")
            time.sleep(wait)


def known_lectures() -> list[dict]:
    return [{"lecture": n, "title": t,
             "pdf_url": f"{COURSE_BASE}/mit6_7960_f24_lec{n}.pdf",
             "page_url": f"{COURSE_BASE}/resources/mit6_7960_f24_lec{n}_pdf/"} for n, t in KNOWN_LECTURES.items()]


def find_lectures(list_url: str) -> list[dict]:
    """Return [{lecture, title, pdf_url, page_url}] from an OCW list page."""
    soup = BeautifulSoup(http_get(list_url).text, "html.parser")
    lectures: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(list_url, a["href"])
        m = re.search(r"lec(\d+)", href)
        if not m or "/courses/" not in href:
            continue
        item = lectures.setdefault(m.group(1), {"lecture": int(m.group(1))})
        text = a.get_text(" ", strip=True)
        if href.lower().endswith(".pdf"):
            item["pdf_url"] = href
        elif "/resources/" in href:
            item["page_url"] = href
            if text and not re.match(r"^pdf\b", text, re.I):
                item["title"] = text
    # resource pages without a direct link: open them and take the download link
    for item in lectures.values():
        if "pdf_url" not in item and "page_url" in item:
            page = BeautifulSoup(http_get(item["page_url"]).text, "html.parser")
            link = page.find("a", href=re.compile(r"\.pdf$", re.I))
            if link:
                item["pdf_url"] = urljoin(item["page_url"], link["href"])
    return sorted((v for v in lectures.values() if "pdf_url" in v), key=lambda v: v["lecture"])


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(1, 5):
        try:
            with http_get(url, retries=1, timeout=180, stream=True) as r, tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except (requests.ConnectionError, requests.Timeout) as exc:   # dropped mid-download
            if attempt == 4:
                raise
            print(f"  download interrupted ({exc.__class__.__name__}), retrying...")
            time.sleep(2 ** attempt)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-url", default=DEFAULT_URL)
    ap.add_argument("--collection", default="curriculum_refs")
    ap.add_argument("--lectures", nargs="*", type=int, help="lecture numbers to ingest (default: all)")
    ap.add_argument("--out", default=None, help="download folder (default data/raw/<course-slug>)")
    ap.add_argument("--offline", action="store_true",
                    help="don't read the list page; use the built-in lecture list (and PDFs already downloaded)")
    args = ap.parse_args()

    slug = re.search(r"/courses/([^/]+)/", args.list_url).group(1)
    out_dir = Path(args.out) if args.out else ROOT / "data" / "raw" / slug
    is_default_course = args.list_url.startswith(COURSE_BASE)
    if args.offline and is_default_course:
        lectures = known_lectures()
    else:
        try:
            lectures = find_lectures(args.list_url)
        except requests.RequestException as exc:
            if not is_default_course:
                raise SystemExit(f"Could not read {args.list_url}: {exc}")
            print(f"Could not read the OCW list page ({exc.__class__.__name__}) - using the built-in lecture list.")
            lectures = known_lectures()
    if args.lectures:
        lectures = [lec for lec in lectures if lec["lecture"] in args.lectures]
    print(f"Found {len(lectures)} lecture PDFs on {args.list_url}")

    rag = get_rag(args.collection)
    total = 0
    for lec in lectures:
        name = lec.get("title", "")
        # OCW titles sometimes already start with "Lecture N:" - don't prefix twice
        title = name if re.match(r"^lecture\s*\d+", name, re.I) else f"Lecture {lec['lecture']}: {name}".rstrip(": ")
        try:
            pdf = out_dir / Path(lec["pdf_url"]).name
            if not (pdf.exists() and pdf.stat().st_size > 0):
                if args.offline:
                    print(f"  - {title}: not downloaded yet (skipped in --offline mode)")
                    continue
                pdf = download(lec["pdf_url"], pdf)
        except Exception as exc:
            print(f"  ! {title}: download failed ({exc})")
            continue
        n = rag.ingest_file(pdf, metadata={
            "title": title,
            "source_id": "SRC-MIT-OCW-6.7960",
            "url": lec.get("page_url", lec["pdf_url"]),
            "publisher": "MIT OpenCourseWare",
            "course": "6.7960 Deep Learning, Fall 2024",
            "lecture": lec["lecture"],
            "department": "AI",
            "industry": "Software & AI",
            "skill_level": "Advanced",
            "doc_kind": "lecture_notes",
            "license": "CC BY-NC-SA 4.0",
            "country": "INTL",
        })
        total += n
        print(f"  {title:<55} {n:>4} chunks" if n else f"  {title:<55} unchanged")
    print(f"\nDone: {total} new chunks. Stats: {rag.stats()['chunks']} chunks in '{args.collection}'.")
    print('Try: python -m rex_ai.core.rag query "how does self-attention work" --collection curriculum_refs')


if __name__ == "__main__":
    main()