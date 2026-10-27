"""PDF export for any generated document (brief, Agent 4: "export documents to PDF"), English or Arabic.

Reuses the Curriculum Agent's Writer (fonts, Arabic shaping and right-to-left tables), so every REX document has the
same look. A document renders its sections in order, then its sources (traceability) and review history. Until a
reviewer approves it, every page carries a DRAFT watermark.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer, Table, TableStyle

from rex_ai.agents.curriculum.pdf import (BODY_W, BRAND, BRAND_LIGHT, MARGIN, MUTED, PAGE_H, PAGE_W, Writer, _AR,
                                          _visual)

LABELS = {
    "en": {"status": "Status", "version": "version", "id": "Document ID", "generated": "Generated",
           "course": "Course", "for": "Prepared for", "sources": "Sources and traceability",
           "review": "Review and approval", "no_reviews": "Not reviewed yet - this document is a draft awaiting review.",
           "draft": "DRAFT - NOT APPROVED", "page": "Page", "notes": "Notes", "confidential": "CONFIDENTIAL",
           "footer": "Alignment is not accreditation.",
           "status_names": {"draft": "Draft", "pending_approval": "Pending approval", "approved": "Approved",
                            "changes_requested": "Changes requested", "rejected": "Rejected"}},
    "ar": {"status": "الحالة", "version": "الإصدار", "id": "رقم المستند", "generated": "تاريخ الإنشاء",
           "course": "الدورة", "for": "مقدم إلى", "sources": "المصادر وإمكانية التتبع",
           "review": "المراجعة والاعتماد الداخلي", "no_reviews": "لم تتم المراجعة بعد - هذا المستند مسودة بانتظار المراجعة.",
           "draft": "مسودة - غير معتمدة", "page": "صفحة", "notes": "ملاحظات", "confidential": "سري",
           "footer": "المواءمة ليست اعتماداً رسمياً.",
           "status_names": {"draft": "مسودة", "pending_approval": "بانتظار الموافقة", "approved": "معتمد داخلياً",
                            "changes_requested": "مطلوب تعديلات", "rejected": "مرفوض"}},
}
CONFIDENTIAL = {"question_bank"}


def _v(text: str) -> str:
    return _visual(text) if _AR.search(text) else text


def _decor(canvas, doc, w: Writer, d: dict, L: dict, approved: bool):
    canvas.saveState()
    canvas.setFillColor(BRAND)
    canvas.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont(w.bold, 8.5)
    canvas.drawString(MARGIN, PAGE_H - 6 * mm, "REX AI  ·  Andalusia Academy")
    canvas.setFont(w.font, 8.5)
    head = d["metadata"].get("type_title") or d["title"]
    if d["doc_type"] in CONFIDENTIAL:
        head = f"{L['confidential']} · {head}"
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 6 * mm, _v(head[:80]))
    canvas.setFillColor(MUTED)
    canvas.setFont(w.font, 7.5)
    canvas.drawString(MARGIN, 10 * mm, f"{d['document_id']} · v{d.get('version', 1)} · {d.get('status', 'draft')}")
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, _v(L["footer"]))
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, _v(f"{L['page']} {doc.page}"))
    if not approved:
        canvas.setFillColor(colors.Color(0.55, 0.29, 0.23, alpha=0.08))
        canvas.setFont(w.bold, 46)
        canvas.translate(PAGE_W / 2, PAGE_H / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, _v(L["draft"]))
    canvas.restoreState()


def _table(w: Writer, t: dict) -> Table:
    cols = t["columns"]
    rel = t.get("widths") or [1] * len(cols)
    widths = [BODY_W * x / sum(rel) for x in rel]
    rows = [[w.p(c, "head", widths[i]) for i, c in enumerate(cols)]]
    for r in t["rows"]:                               # a cell with line breaks becomes one paragraph per line
        rows.append([[w.p(x, "cell", widths[i]) for x in str(c).split("\n")] if "\n" in str(c)
                     else w.p(c, "cell", widths[i]) for i, c in enumerate(r)])
    return w.table(rows, widths)


def _notes_box(w: Writer, notes: list[str]) -> Table:
    box = Table([[[w.p(n, "note", BODY_W - 14) for n in notes]]], colWidths=[BODY_W])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, BRAND), ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
                             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return box


def _section(w: Writer, s: dict) -> list:
    head = w.p(s["heading"], "h1" if s.get("level", 1) == 1 else "h2")
    body: list = [w.p(x) for x in s.get("paragraphs") or []]
    body += w.bullets(s["bullets"], numbered=s.get("numbered", False)) if s.get("bullets") else []
    if s.get("table") and s["table"].get("columns"):
        body.append(_table(w, s["table"]))
    if s.get("notes"):
        body += [Spacer(1, 3), _notes_box(w, s["notes"])]
    # keep a heading with the start of its content (a long table may still break across pages)
    first = body[:1] if body and isinstance(body[0], Table) else body[:2]
    return [KeepTogether([head, *first]), *body[len(first):]]


def render_document_pdf(doc: dict, out_path: Path, reviews: Optional[list[dict]] = None) -> Path:
    w = Writer(doc.get("language") or "en")
    L = LABELS[w.lang]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    approved = doc.get("status") == "approved"
    m = doc.get("metadata") or {}
    pdf = SimpleDocTemplate(str(out_path), pagesize=(PAGE_W, PAGE_H), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=16 * mm, bottomMargin=17 * mm, title=doc["title"],
                            author="REX AI Document Agent", subject=m.get("type_title") or doc["doc_type"])
    status = L["status_names"].get(doc.get("status", "draft"), doc.get("status", "draft"))
    sub = [m.get("type_title") or doc["doc_type"], f"{L['status']}: {status}",
           f"{L['version']} {doc.get('version', 1)}", f"{L['id']}: {doc['document_id']}",
           f"{L['generated']}: {date.today().isoformat()}"]
    flow: list = [w.p(doc["title"], "title"), w.p(" · ".join(sub), "subtitle")]
    if m.get("course_title") and m["course_title"] != doc["title"]:
        flow.append(w.p(f"{L['course']}: {m['course_title']}", "subtitle"))
    for s in doc.get("sections") or []:
        flow += _section(w, s)
    flow.append(w.p(L["sources"], "h1"))
    flow += w.bullets([f"{s['source_id']} - {s['title']}" + (f" ({s['record_id']})" if s.get("record_id") else "")
                       for s in doc.get("sources") or []])
    flow.append(w.p(L["review"], "h1"))
    if reviews:
        flow += w.bullets([f"{r['created_at'][:16].replace('T', ' ')} - {r['reviewer']}: {r['action']}"
                           + (f" - {r['comment']}" if r.get("comment") else "") for r in reviews])
    else:
        flow.append(w.p(L["no_reviews"], "note"))
    pdf.build(flow, onFirstPage=lambda c, d: _decor(c, d, w, doc, L, approved),
              onLaterPages=lambda c, d: _decor(c, d, w, doc, L, approved))
    return out_path

