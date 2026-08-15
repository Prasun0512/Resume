from __future__ import annotations

from pathlib import Path

from docx import Document
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Prasun_Kumar_AI_ML_Architect_Resume.docx"
PDF = ROOT / "Prasun_Kumar_AI_ML_Architect_Resume.pdf"

SECTION_HEADINGS = {
    "PROFESSIONAL SUMMARY",
    "CORE SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "SELECTED AI ARCHITECTURE PROJECTS",
    "PUBLIC GITHUB PORTFOLIO",
    "EDUCATION AND CERTIFICATION",
    "AWARDS AND RECOGNITION",
    "THOUGHT LEADERSHIP",
}


def get_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "Name",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            textColor=colors.HexColor("#102033"),
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "headline": ParagraphStyle(
            "Headline",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=11,
            textColor=colors.HexColor("#0B5C74"),
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "contact": ParagraphStyle(
            "Contact",
            parent=base["Normal"],
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#425466"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.white,
            backColor=colors.HexColor("#0B5C74"),
            borderPadding=3.5,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "role": ParagraphStyle(
            "Role",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=10.2,
            textColor=colors.HexColor("#102033"),
            spaceBefore=4,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=9.5,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=2.8,
        ),
        "compact": ParagraphStyle(
            "Compact",
            parent=base["BodyText"],
            fontSize=7.3,
            leading=8.8,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=2.2,
        ),
    }


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def is_role_line(text: str) -> bool:
    role_markers = (
        "Technical Lead, AI and ML |",
        "Senior Software Engineer |",
        "Software Engineer |",
        "Associate IT Applications Specialist |",
        "Master Level Intern |",
    )
    return any(text.startswith(marker) for marker in role_markers)


def build_story() -> list:
    styles = get_styles()
    paragraphs = [p.text.strip() for p in Document(DOCX).paragraphs if p.text.strip()]
    story: list = []

    for index, text in enumerate(paragraphs):
        clean = escape(text)
        if index == 0:
            story.append(Paragraph(clean, styles["name"]))
        elif index == 1:
            story.append(Paragraph(clean, styles["headline"]))
        elif index == 2:
            story.append(Paragraph(clean, styles["contact"]))
        elif text in SECTION_HEADINGS:
            story.append(Paragraph(clean, styles["section"]))
        elif is_role_line(text):
            story.append(Paragraph(clean, styles["role"]))
        elif text.endswith(":") and len(text) < 75:
            story.append(Paragraph(f"<b>{clean}</b>", styles["body"]))
        elif text.startswith(("enterprise-", "agentic-", "document-", "email-", "ai-", "mlflow-")):
            story.append(Paragraph(clean, styles["compact"]))
        else:
            story.append(Paragraph(clean, styles["body"]))

    return story


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(width - 0.45 * inch, 0.32 * inch, f"Prasun Kumar - AI/ML Architect Resume | Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    doc = BaseDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=0.42 * inch,
        rightMargin=0.42 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.42 * inch,
        title="Prasun Kumar AI ML Architect Resume",
        author="Prasun Kumar",
        subject="AI/ML Technical Lead and GenAI Solution Architect Resume",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="resume", frames=[frame], onPage=header_footer)])
    doc.build(build_story())


if __name__ == "__main__":
    build_pdf()
