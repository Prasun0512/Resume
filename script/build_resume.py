"""Build the public, single-column resume from reviewed public facts only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#1A2430")
ACCENT = colors.HexColor("#126477")
MUTED = colors.HexColor("#4B5563")


def build(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    styles = {
        "name": ParagraphStyle("Name", fontName="Helvetica-Bold", fontSize=23, leading=26, textColor=INK, spaceAfter=5),
        "headline": ParagraphStyle("Headline", fontName="Helvetica-Bold", fontSize=9.7, leading=13, textColor=ACCENT, spaceAfter=4),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.6, leading=12.7, textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8.6, leading=11.3, textColor=MUTED, spaceAfter=4),
        "section": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=10.7, leading=14, textColor=ACCENT, spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "role": ParagraphStyle("Role", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=2, keepWithNext=True),
        "bullet": ParagraphStyle("Bullet", fontName="Helvetica", fontSize=9.6, leading=12.7, textColor=INK, leftIndent=9, firstLineIndent=-9, spaceAfter=3),
    }

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def link(url, label):
        return f'<a href="{escape(url)}" color="#126477">{escape(label)}</a>'

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 24, "Prasun Kumar | AI/ML Engineering and Technical Leadership")
        canvas.drawRightString(A4[0] - 42, 24, str(doc.page))

    story = [p(escape(data["name"]).upper(), "name"), p(escape(data["headline"]), "headline"),
             p(escape(data["current_role"])), p(escape(data["contact"]), "small"),
             p(" | ".join(link(x["url"], x["label"]) for x in data["links"]), "small"),
             p("PROFESSIONAL SUMMARY", "section"), p(escape(data["summary"])),
             p("CORE EXPERTISE", "section")]
    for skill in data["skills"]:
        story.append(p(f'<b>{escape(skill["label"])}:</b> {escape(skill["text"])}'))
    story.append(p("PROFESSIONAL EXPERIENCE", "section"))
    for job in data["employment"]:
        group = [p(escape(job["role"]), "role"),
                 p(f'{escape(job["company"])} | {escape(job["dates"])}', "small")]
        group.extend(p("- " + escape(b), "bullet") for b in job["bullets"])
        story.append(KeepTogether(group))

    story.extend([PageBreak(), p("SELECTED PROFESSIONAL PROJECTS", "section"),
                  p("Prior-role enterprise work. Public links are sanitized reference implementations, not employer source code.", "small")])
    for project in data["projects"]:
        group = [p(escape(project["name"]), "role")]
        group.extend(p("- " + escape(b), "bullet") for b in project["bullets"])
        if project["repo"]:
            group.append(p("Public reference: " + link("https://github.com/Prasun0512/" + project["repo"], project["repo"]), "small"))
        story.append(KeepTogether(group))
    story.extend([p("INDEPENDENT GITHUB PROJECTS", "section"),
                  p("Personal portfolio implementations, separate from professional client delivery.", "small")])
    for project in data["independent"]:
        story.append(p('<b>' + link("https://github.com/Prasun0512/" + project["repo"], project["name"]) + '</b> - ' + escape(project["text"])))
    story.append(p("EDUCATION AND RECOGNITION", "section"))
    for line in data["education"]:
        story.append(p(escape(line)))
    story.append(p(link(data["credential"], "ML and AI credential verification") + " | " + escape(data["recognition"]), "small"))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=34, bottomMargin=39, title="Prasun Kumar - AI ML Engineering and Leadership Resume",
                            author="Prasun Kumar", subject="Professional AI ML projects, IBM employment and independent portfolio work")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "resume/profile.json")
    parser.add_argument("--out", type=Path, default=ROOT / "Prasun_Kumar_AI_ML_Architect_Resume.pdf")
    args = parser.parse_args()
    build(args.source, args.out)
