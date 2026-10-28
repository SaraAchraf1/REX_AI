"""Professional curriculum PDF (brief, Agent 3 FR 24), English or Arabic.

    path = render_curriculum_pdf(curriculum, gap_report, other_reports, reviews, out_path)

Sections: metadata table (the brief's knowledge-base fields), objectives, outcomes, modules with topics, practical
work and hours, assessment plan, capstone, accreditation-alignment report (coverage, competency table, gaps,
recommendations, disclaimer), sources (traceability) and review history. Until a reviewer approves it, every page
carries a DRAFT watermark.

Arabic: letters are joined with arabic_reshaper and ordered with python-bidi; lines are wrapped BEFORE reordering
so multi-line paragraphs read top to bottom. Needs a TTF font with Arabic glyphs (REX_PDF_FONT, or Arial / Tahoma /
DejaVu Sans / Noto Naskh found on the system).
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rex_ai import config
from rex_ai.agents.curriculum.generator import DEPARTMENT_AR, INDUSTRY_AR, LEVEL_AR

log = logging.getLogger("rex.curriculum.pdf")

BRAND = colors.HexColor("#8B4A3A")
BRAND_LIGHT = colors.HexColor("#F6EFEA")
GRID = colors.HexColor("#D9C9BF")
TEXT = colors.HexColor("#2B2B2B")
MUTED = colors.HexColor("#6B6B6B")
STATUS_COLORS = {"covered": colors.HexColor("#2E7D32"), "partial": colors.HexColor("#B26A00"),
                 "missing": colors.HexColor("#C62828")}
_AR = re.compile(r"[\u0600-\u06FF]")
COUNTRIES = {"en": {"EG": "Egypt", "SA": "Saudi Arabia", "INTL": "International"},
             "ar": {"EG": "\u0645\u0635\u0631", "SA": "\u0627\u0644\u0645\u0645\u0644\u0643\u0629 \u0627\u0644\u0639\u0631\u0628\u064A\u0629 \u0627\u0644\u0633\u0639\u0648\u062F\u064A\u0629", "INTL": "\u062F\u0648\u0644\u064A"}}
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
BODY_W = PAGE_W - 2 * MARGIN

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"),
    ("/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf", "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]

LABELS = {
    "en": {
        "doc": "Course Curriculum", "status": "Status", "version": "version", "id": "Curriculum ID",
        "meta": "Course information", "department": "Department", "audience": "Target audience",
        "level": "Skill level", "industry": "Industry", "duration": "Course duration", "hours": "hours",
        "weeks": "weeks", "country": "Country", "framework": "Accreditation framework (alignment only)",
        "prereq": "Prerequisites", "sources_date": "Data sources & update date", "objectives": "Course objectives",
        "outcomes": "Learning outcomes", "outcomes_intro": "By the end of the course, learners will be able to:",
        "structure": "Course structure", "module": "Module and topics", "practice": "Practical work",
        "h": "Hours", "total": "Total", "exercises": "Practical exercises", "assessments": "Assessment plan",
        "assessment": "Assessment", "type": "Type", "weight": "Weight", "capstone": "Capstone project",
        "alignment": "Accreditation alignment report", "coverage": "Coverage", "competency": "Competency",
        "result": "Result", "evidence": "Evidence", "missing_topics": "Missing topics",
        "missing_outcomes": "Missing learning outcomes", "recommendations": "Recommendations",
        "other_checks": "Other alignment checks", "sources": "Sources and traceability",
        "review": "Review and approval", "no_reviews": "Not reviewed yet - this curriculum is a draft awaiting review.",
        "draft": "DRAFT - NOT APPROVED", "generated": "Generated", "page": "Page",
        "none": "None", "covered": "covered", "partial": "partial", "missing": "missing",
        "footer_note": "Alignment is not accreditation.",
        "status_names": {"draft": "Draft", "pending_approval": "Pending approval", "approved": "Approved",
                         "changes_requested": "Changes requested", "rejected": "Rejected"},
    },
    "ar": {
        "doc": "المنهج التدريبي", "status": "الحالة", "version": "الإصدار", "id": "رقم المنهج",
        "meta": "معلومات الدورة", "department": "القسم", "audience": "الفئة المستهدفة", "level": "المستوى",
        "industry": "القطاع", "duration": "مدة الدورة", "hours": "ساعة", "weeks": "أسابيع", "country": "الدولة",
        "framework": "إطار الاعتماد (مواءمة فقط)", "prereq": "المتطلبات السابقة",
        "sources_date": "مصادر البيانات وتاريخ التحديث", "objectives": "أهداف الدورة", "outcomes": "نواتج التعلم",
        "outcomes_intro": "في نهاية الدورة سيكون المتدرب قادراً على أن:", "structure": "هيكل الدورة",
        "module": "الوحدة والموضوعات", "practice": "التطبيق العملي", "h": "الساعات", "total": "الإجمالي",
        "exercises": "التمارين العملية", "assessments": "خطة التقييم", "assessment": "التقييم", "type": "النوع",
        "weight": "الوزن", "capstone": "المشروع الختامي", "alignment": "تقرير المواءمة مع إطار الاعتماد",
        "coverage": "نسبة التغطية", "competency": "الكفاءة", "result": "النتيجة", "evidence": "الدليل",
        "missing_topics": "موضوعات ناقصة", "missing_outcomes": "نواتج تعلم ناقصة", "recommendations": "التوصيات",
        "other_checks": "فحوص مواءمة أخرى", "sources": "المصادر وإمكانية التتبع", "review": "المراجعة والاعتماد الداخلي",
        "no_reviews": "لم تتم المراجعة بعد - هذا المنهج مسودة بانتظار المراجعة.", "draft": "مسودة - غير معتمدة",
        "generated": "تاريخ الإنشاء", "page": "صفحة", "none": "لا يوجد", "covered": "مغطى", "partial": "جزئي",
        "missing": "ناقص", "footer_note": "المواءمة ليست اعتماداً رسمياً.",
        "status_names": {"draft": "مسودة", "pending_approval": "بانتظار الموافقة", "approved": "معتمد داخلياً",
                         "changes_requested": "مطلوب تعديلات", "rejected": "مرفوض"},
    },
}


# ============================================================================ fonts and text shaping
_FONTS: Optional[tuple[str, str]] = None


def fonts() -> tuple[str, str]:
    """(regular, bold) font names, registered once."""
    global _FONTS
    if _FONTS:
        return _FONTS
    candidates = [(config.PDF_FONT, config.PDF_FONT_BOLD or config.PDF_FONT)] if config.PDF_FONT else []
    for regular, bold in candidates + FONT_CANDIDATES:
        if regular and Path(regular).exists():
            try:
                pdfmetrics.registerFont(TTFont("RexSans", regular))
                pdfmetrics.registerFont(TTFont("RexSans-Bold", bold if bold and Path(bold).exists() else regular))
                _FONTS = ("RexSans", "RexSans-Bold")
                return _FONTS
            except Exception as exc:
                log.warning("Font %s could not be loaded: %s", regular, exc)
    log.warning("No TTF font with Arabic glyphs found; set REX_PDF_FONT. Using Helvetica (no Arabic).")
    _FONTS = ("Helvetica", "Helvetica-Bold")
    return _FONTS


def _visual(text: str) -> str:
    """Logical Arabic text -> joined letters in visual order (one line)."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def _wrap_rtl(text: str, font: str, size: float, width: float) -> list[str]:
    """Wrap in logical order (so line 1 is the start of the sentence), then reorder each line."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and pdfmetrics.stringWidth(_visual(trial), font, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return [_visual(line) for line in lines]


class Writer:
    def __init__(self, language: str):
        self.lang = language if language in LABELS else "en"
        self.L = LABELS[self.lang]
        self.font, self.bold = fonts()
        self.rtl = self.lang == "ar"
        align = TA_RIGHT if self.rtl else TA_LEFT

        def style(name: str, **kw) -> ParagraphStyle:
            return ParagraphStyle(name, **{"fontName": self.font, "textColor": TEXT, "alignment": align, **kw})

        b = self.bold
        self.styles = {
            "title": style("title", fontName=b, fontSize=20, leading=25, textColor=BRAND, spaceAfter=4),
            "subtitle": style("subtitle", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10),
            "h1": style("h1", fontName=b, fontSize=13.5, leading=18, textColor=BRAND, spaceBefore=12, spaceAfter=5),
            "h2": style("h2", fontName=b, fontSize=11, leading=15, spaceBefore=6, spaceAfter=3),
            "body": style("body", fontSize=9.5, leading=13.5, spaceAfter=3),
            "cell": style("cell", fontSize=8.6, leading=11.5),
            "cellb": style("cellb", fontName=b, fontSize=8.6, leading=11.5),
            "head": style("head", fontName=b, fontSize=8.8, leading=11.5, textColor=colors.white),
            "note": style("note", fontSize=8.3, leading=11, textColor=MUTED),
        }

    def p(self, text, style: str = "body", width: float = BODY_W, color=None) -> Paragraph:
        text = str(text if text is not None else "")
        st = self.styles[style]
        if _AR.search(text):
            pad = 12 if style in ("cell", "cellb", "head") else 4
            # keep a margin: if Paragraph re-wrapped a visual-order line, the words would come out reversed
            lines = _wrap_rtl(text, st.fontName, st.fontSize, max((width - pad) * 0.96, 40))
            markup = "<br/>".join(escape(x) for x in lines)
            st = ParagraphStyle(f"{st.name}-rtl", parent=st, alignment=TA_RIGHT)
        else:
            markup = escape(text)
        if color is not None:
            markup = f'<font color="{color.hexval()}">{markup}</font>'
        return Paragraph(markup, st)

    def bullets(self, items: list[str], numbered: bool = False) -> list:
        out = []
        for i, item in enumerate(items, start=1):
            out.append(self.p(f"{f'{i}.' if numbered else '•'} {item}", "body"))
        return out or [self.p(self.L["none"], "note")]

    def table(self, rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
        if self.rtl:                                           # first column on the right for Arabic
            rows = [list(reversed(r)) for r in rows]
            widths = list(reversed(widths))
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                 ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), BRAND))
        if zebra:
            style += [("BACKGROUND", (0, r), (-1, r), BRAND_LIGHT) for r in range(1 if header else 0, len(rows), 2)]
        t.setStyle(TableStyle(style))
        return t


# ============================================================================ document
def _page_decor(canvas, doc, w: Writer, cur: dict, approved: bool):
    canvas.saveState()
    canvas.setFillColor(BRAND)
    canvas.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont(w.bold, 8.5)
    canvas.drawString(MARGIN, PAGE_H - 6 * mm, "REX AI  ·  Curriculum & Accreditation Alignment")
    canvas.setFont(w.font, 8.5)
    title = cur["course_title"]
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 6 * mm, _visual(title[:70]) if _AR.search(title) else title[:70])
    canvas.setFillColor(MUTED)
    canvas.setFont(w.font, 7.5)
    left = f"{cur['curriculum_id']} · v{cur.get('version', 1)} · {cur.get('status', 'draft')}"
    canvas.drawString(MARGIN, 10 * mm, left)
    note = w.L["footer_note"]
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, _visual(note) if _AR.search(note) else note)
    page = f"{w.L['page']} {doc.page}"
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, _visual(page) if _AR.search(page) else page)
    if not approved:
        canvas.setFillColor(colors.Color(0.55, 0.29, 0.23, alpha=0.08))
        canvas.setFont(w.bold, 46)
        canvas.translate(PAGE_W / 2, PAGE_H / 2)
        canvas.rotate(35)
        mark = w.L["draft"]
        canvas.drawCentredString(0, 0, _visual(mark) if _AR.search(mark) else mark)
    canvas.restoreState()


def _metadata_rows(w: Writer, cur: dict) -> list[list]:
    L, m = w.L, cur.get("metadata") or {}
    hours = cur.get("training_hours") or m.get("duration_hours")
    weeks = m.get("duration_weeks")
    duration = f"{hours:g} {L['hours']}" + (f" · {weeks} {L['weeks']}" if weeks else "")
    fw = m.get("accreditation_framework") or L["none"]
    sources = ", ".join(m.get("data_sources") or []) or L["none"]
    if m.get("update_date"):
        sources += f" · {m['update_date']}"
    industry = m.get("industry")
    industry = ", ".join(industry) if isinstance(industry, list) else industry
    department, level = m.get("department"), m.get("skill_level")
    if w.rtl:
        department = DEPARTMENT_AR.get(department, department)
        level = LEVEL_AR.get(level, level)
        industry = INDUSTRY_AR.get(industry, industry)
    country = COUNTRIES[w.lang].get(m.get("country"), m.get("country"))
    rows = [(L["department"], department or L["none"]), (L["audience"], cur.get("target_audience")),
            (L["level"], level or L["none"]), (L["industry"], industry or L["none"]),
            (L["duration"], duration), (L["country"], country or L["none"]), (L["framework"], fw),
            (L["prereq"], "; ".join(cur.get("prerequisites") or []) or L["none"]), (L["sources_date"], sources)]
    widths = [52 * mm, BODY_W - 52 * mm]
    return [[w.p(k, "cellb", widths[0]), w.p(v, "cell", widths[1])] for k, v in rows]


def _structure(w: Writer, cur: dict) -> Table:
    L = w.L
    widths = [9 * mm, 88 * mm, BODY_W - 9 * mm - 88 * mm - 15 * mm, 15 * mm]
    rows = [[w.p("#", "head", widths[0]), w.p(L["module"], "head", widths[1]), w.p(L["practice"], "head", widths[2]),
             w.p(L["h"], "head", widths[3])]]
    for i, mod in enumerate(cur.get("modules") or [], start=1):
        topics = "; ".join(mod.get("topics") or [])
        cell = [w.p(mod["title"], "cellb", widths[1]), w.p(topics, "cell", widths[1])]
        practice = [w.p(e, "cell", widths[2]) for e in mod.get("exercises") or []] or [w.p("-", "cell")]
        rows.append([w.p(str(i), "cell", widths[0]), cell, practice,
                     w.p(f"{float(mod.get('hours') or 0):g}", "cell", widths[3])])
    total = sum(float(m.get("hours") or 0) for m in cur.get("modules") or [])
    rows.append([w.p("", "cell"), w.p(L["total"], "cellb", widths[1]), w.p("", "cell"),
                 w.p(f"{total:g}", "cellb", widths[3])])
    return w.table(rows, widths)


def _assessments(w: Writer, cur: dict) -> Table:
    L = w.L
    widths = [BODY_W - 60 * mm, 35 * mm, 25 * mm]
    rows = [[w.p(L["assessment"], "head", widths[0]), w.p(L["type"], "head", widths[1]),
             w.p(L["weight"], "head", widths[2])]]
    for a in cur.get("assessments") or []:
        rows.append([w.p(a["name"], "cell", widths[0]), w.p(a["type"], "cell", widths[1]),
                     w.p(f"{float(a['weight_pct']):g}%", "cell", widths[2])])
    return w.table(rows, widths)


def _alignment(w: Writer, report: dict, others: list[dict]) -> list:
    L = w.L
    flow = [w.p(L["alignment"], "h1"),
            w.p(f"{report['framework']}  ·  {L['coverage']}: {report['coverage_pct']:g}%", "h2")]
    widths = [55 * mm, 20 * mm, BODY_W - 75 * mm]
    rows = [[w.p(L["competency"], "head", widths[0]), w.p(L["result"], "head", widths[1]),
             w.p(L["evidence"], "head", widths[2])]]
    for it in report.get("items") or []:
        rows.append([w.p(it["competency"], "cell", widths[0]),
                     w.p(L.get(it["status"], it["status"]), "cellb", widths[1], STATUS_COLORS.get(it["status"])),
                     w.p(it.get("evidence") or "", "cell", widths[2])])
    if len(rows) > 1:
        flow.append(w.table(rows, widths))
    for key, label in (("missing_topics", L["missing_topics"]), ("missing_outcomes", L["missing_outcomes"])):
        if report.get(key):
            flow += [w.p(label, "h2"), *w.bullets(report[key])]
    if report.get("recommendations"):
        flow += [w.p(L["recommendations"], "h2"), *w.bullets(report["recommendations"])]
    box = Table([[w.p(report.get("disclaimer", ""), "note", BODY_W - 12)]], colWidths=[BODY_W])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, BRAND), ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
                             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    flow += [Spacer(1, 6), box]
    if others:
        flow.append(w.p(L["other_checks"], "h2"))
        flow += w.bullets([f"{o['framework']}: {o['coverage_pct']:g}%"
                           + (f" - {L['missing']}: {', '.join((o.get('missing_topics') or []) + (o.get('missing_outcomes') or []))[:180]}"
                              if (o.get("missing_topics") or o.get("missing_outcomes")) else "") for o in others])
    return flow


def render_curriculum_pdf(cur: dict, out_path: Path, report: Optional[dict] = None,
                          other_reports: Optional[list[dict]] = None, reviews: Optional[list[dict]] = None) -> Path:
    w = Writer(cur.get("language") or "en")
    L = w.L
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    approved = cur.get("status") == "approved"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=16 * mm, bottomMargin=17 * mm, title=cur["course_title"],
                            author="REX AI Curriculum Agent", subject=L["doc"])
    status = L["status_names"].get(cur.get("status", "draft"), cur.get("status", "draft"))
    generated = date.today().isoformat()
    flow = [w.p(cur["course_title"], "title"),
            w.p(f"{L['doc']} · {L['status']}: {status} · {L['version']} {cur.get('version', 1)} · "
                f"{L['id']}: {cur['curriculum_id']} · {L['generated']}: {generated}", "subtitle"),
            w.p(L["meta"], "h1"), w.table(_metadata_rows(w, cur), [52 * mm, BODY_W - 52 * mm], header=False),
            w.p(L["objectives"], "h1"), *w.bullets(cur.get("objectives") or []),
            w.p(L["outcomes"], "h1"), w.p(L["outcomes_intro"], "body"),
            *w.bullets(cur.get("learning_outcomes") or [], numbered=True),
            w.p(L["structure"], "h1"), _structure(w, cur)]
    module_ex = {e for m in cur.get("modules") or [] for e in m.get("exercises") or []}
    extra = [e for e in cur.get("practical_exercises") or [] if e not in module_ex]
    if extra:
        flow += [w.p(L["exercises"], "h1"), *w.bullets(extra)]
    flow += [KeepTogether([w.p(L["assessments"], "h1"), _assessments(w, cur)]),
             w.p(L["capstone"], "h1"), w.p(cur.get("capstone_project") or L["none"])]
    if report:
        flow += _alignment(w, report, other_reports or [])
    flow.append(w.p(L["sources"], "h1"))
    flow += w.bullets([f"{s['source_id']} - {s['title']}" + (f" ({s['record_id']})" if s.get("record_id") else "")
                       for s in cur.get("sources") or []])
    flow.append(w.p(L["review"], "h1"))
    if reviews:
        flow += w.bullets([f"{r['created_at'][:16].replace('T', ' ')} - {r['reviewer']}: {r['action']}"
                           + (f" - {r['comment']}" if r.get("comment") else "") for r in reviews])
    else:
        flow.append(w.p(L["no_reviews"], "note"))
    doc.build(flow, onFirstPage=lambda c, d: _page_decor(c, d, w, cur, approved),
              onLaterPages=lambda c, d: _page_decor(c, d, w, cur, approved))
    return out_path
