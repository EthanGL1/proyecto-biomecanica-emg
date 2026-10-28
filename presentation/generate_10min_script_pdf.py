"""Render the canonical Canva speaker-script Markdown as a print-ready PDF."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Canva_10_minute_speaker_script_EN.md"
OUTPUT = ROOT / "Canva_10_minute_speaker_script_EN_F59.pdf"
SLIDE_RE = re.compile(r"^## Slide (\d+) - (.+?) \((\d{2}:\d{2})\)$", re.MULTILINE)


def _parse_script(text: str) -> tuple[str, list[dict[str, str]], str]:
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    note_match = re.search(r"^\*\*Demonstration note \(not spoken\):\*\* (.+)$", text, re.MULTILINE)
    matches = list(SLIDE_RE.finditer(text))
    if not title_match or not note_match or len(matches) != 10:
        raise ValueError("Expected title, demonstration note, and exactly 10 timed slide sections.")

    slides: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : block_end].strip()
        cue_match = re.search(r"^\*\*Visual cue:\*\* (.+)$", block, re.MULTILINE)
        script_match = re.search(r"^\*\*Script:\*\*\s*\n+(.+)$", block, re.MULTILINE | re.DOTALL)
        if not cue_match or not script_match:
            raise ValueError(f"Slide {match.group(1)} is missing a cue or script.")
        speech = " ".join(part.strip() for part in script_match.group(1).split("\n\n") if part.strip())
        slides.append(
            {
                "number": match.group(1),
                "title": match.group(2),
                "time": match.group(3),
                "cue": cue_match.group(1).strip(),
                "speech": speech,
            }
        )

    total_seconds = sum(int(s["time"].split(":")[0]) * 60 + int(s["time"].split(":")[1]) for s in slides)
    if total_seconds != 600:
        raise ValueError(f"Slide timings total {total_seconds} seconds, expected 600.")
    return title_match.group(1), slides, note_match.group(1)


def _paragraph_markup(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def _draw_page(canvas, document) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D8E2EA"))
    canvas.setLineWidth(0.6)
    canvas.line(0.72 * inch, 0.58 * inch, width - 0.72 * inch, 0.58 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526170"))
    canvas.drawString(0.72 * inch, 0.38 * inch, "GROUP 4 | sEMG | F59 | 2026-10-28")
    canvas.drawRightString(width - 0.72 * inch, 0.38 * inch, f"Page {document.page}")
    canvas.restoreState()


def render() -> Path:
    text = SOURCE.read_text(encoding="utf-8")
    title, slides, demo_note = _parse_script(text)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#15324B"), alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(name="DeckSub", parent=styles["Normal"], fontSize=9.5, leading=13, textColor=colors.HexColor("#536476"), spaceAfter=4))
    styles.add(ParagraphStyle(name="Note", parent=styles["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#4C3B12"), backColor=colors.HexColor("#FFF4D6"), borderColor=colors.HexColor("#E2B94F"), borderWidth=0.5, borderPadding=7, spaceBefore=4, spaceAfter=9))
    styles.add(ParagraphStyle(name="SlideTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#0E5963"), spaceBefore=2, spaceAfter=2))
    styles.add(ParagraphStyle(name="SlideTime", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=colors.HexColor("#0E5963"), alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name="Cue", parent=styles["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#657585"), spaceAfter=5))
    styles.add(ParagraphStyle(name="Speech", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2, leading=14.3, textColor=colors.HexColor("#1F2933"), spaceAfter=8, allowWidows=0, allowOrphans=0))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter,
        rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.62 * inch, bottomMargin=0.78 * inch,
        title=title, author="Group 4", subject="Timed 10-minute scientific presentation script",
    )
    story = [
        Paragraph("Group 4 | Scientific presentation", styles["DocTitle"]),
        Paragraph("Quadriceps sEMG during unilateral knee extension | existing Canva deck | 10:00 assigned time", styles["DeckSub"]),
        Paragraph(f"<b>Demo scope:</b> {_paragraph_markup(demo_note)}", styles["Note"]),
        Paragraph("Slide times: 00:30, 00:40, 00:50, 00:55, 00:50, 01:00, 01:00, 01:00, 01:00, 02:15. The assigned total is 10:00.", styles["DeckSub"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D8E2EA"), spaceBefore=3, spaceAfter=9),
    ]

    for slide in slides:
        number = int(slide["number"])
        if number in {3, 5, 7, 9, 10}:
            story.append(PageBreak())
        title_table = Table(
            [[Paragraph(f"Slide {number} - {_paragraph_markup(slide['title'])}", styles["SlideTitle"]), Paragraph(slide["time"], styles["SlideTime"])]],
            colWidths=[5.58 * inch, 0.72 * inch],
        )
        title_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        story.extend([
            title_table,
            Paragraph(f"<b>Visual cue:</b> {_paragraph_markup(slide['cue'])}", styles["Cue"]),
            Paragraph(_paragraph_markup(slide["speech"]), styles["Speech"]),
            HRFlowable(width="100%", thickness=0.45, color=colors.HexColor("#E2E8ED"), spaceBefore=2, spaceAfter=7),
        ])
    doc.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    return OUTPUT


if __name__ == "__main__":
    print(render())
