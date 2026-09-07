"""Styled, local document generation with format-specific validation."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


OUTPUT_DIR = Path(__file__).with_name("generated_files").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_FORMATS = {"pdf", "docx", "txt", "md", "pptx", "xlsx", "csv", "json"}
_TEAL = "087E70"
_INK = "18212B"
_PALE = "EAF3F0"


@dataclass(frozen=True)
class Artifact:
    path: Path
    media_type: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"filename": self.path.name, "path": str(self.path), "media_type": self.media_type, "size_bytes": self.size_bytes}


def _safe_filename(filename: str, extension: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(str(filename or "output")).name).strip(" .") or "output"
    if Path(stem).suffix.lower() != f".{extension}":
        stem = f"{Path(stem).stem}.{extension}"
    return OUTPUT_DIR / stem


def _paragraphs(content: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n\s*\n", content or "") if item.strip()]


def _rows(content: str) -> list[list[str]]:
    rows = [row for row in csv.reader((content or "").splitlines()) if any(cell.strip() for cell in row)]
    return rows or [["Content"], [content.strip() or "No content supplied"]]


def derive_clean_title(question: str, content: str = "") -> tuple[str, str, str]:
    """
    Derives a clean document title, sanitized content (without duplicated title heading),
    and a filesystem-safe slug for the filename.
    Prevents raw prompt instructions (e.g. 'Give me the code to...') from becoming the main heading.
    """
    # 0. Check if user explicitly asked for a title in quotes, e.g. titled "Q3 Report"
    quoted_title_match = re.search(r'(?:title|titled|named)\s+["\']([^"\']+)["\']', question or "", re.I)
    if quoted_title_match:
        explicit_title = quoted_title_match.group(1).strip()
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', explicit_title.lower()).strip('_')
        slug_parts = [p for p in slug.split('_') if p][:5]
        safe_name = '_'.join(slug_parts) if slug_parts else 'document'
        return explicit_title, content, safe_name

    # 1. Check content for leading heading (# Heading or Title: ...)
    content_lines = (content or "").splitlines()
    title_from_content = ""
    new_content_lines = []
    found_heading = False
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        if not found_heading and i < 5 and (stripped.startswith("#") or stripped.lower().startswith("title:")):
            if stripped.startswith("#"):
                title_from_content = re.sub(r"^#+\s*", "", stripped).strip()
            else:
                title_from_content = re.sub(r"^title:\s*", "", stripped, flags=re.I).strip()
            title_from_content = re.sub(r"[\*`_]", "", title_from_content).strip()
            # If the heading is not just the raw prompt itself
            if len(title_from_content) > 2 and not title_from_content.lower().startswith(("give me", "write code to", "user request", "prompt:")):
                found_heading = True
                continue  # Skip this line from content so it isn't rendered twice
        new_content_lines.append(line)

    if found_heading and title_from_content:
        clean_title = title_from_content
        cleaned_content = "\n".join(new_content_lines).strip()
    else:
        cleaned_content = content
        q = (question or "").strip()
        q = re.sub(r"^[/!]agent\s+", "", q, flags=re.I)
        # Remove trailing file generation requests
        q = re.sub(r"[\s,]+(and\s+)?(also\s+)?(generate|generatet|save|create|make|export)(t)?(\s+it)?(\s+as|\s+in)?(\s+a|\s+the|\s+he)?\s+(pdf|word|docx|doc|excel|xlsx|csv|json|powerpoint|pptx|file|document)\b.*$", "", q, flags=re.I)
        q = re.sub(r"[\s,]+(also\s+)?generate(t)?\s+(he|the|a)?\s*(pdf|word|docx|excel|xlsx|csv|file)\b.*$", "", q, flags=re.I)
        # Remove leading command phrases
        q = re.sub(r"^(give\s+me(\s+the|\s+a)?|write(\s+a|\s+the)?|create(\s+a|\s+the)?|generate(\s+a|\s+the)?|show(\s+me)?|please\s+|can\s+you\s+|how\s+to\s+)", "", q, flags=re.I).strip()
        q = re.sub(r"^(code\s+to\s+|script\s+to\s+|program\s+to\s+|solution\s+for\s+)", "", q, flags=re.I).strip()
        q = re.sub(r"^(print\s+the\s+|print\s+)", "", q, flags=re.I).strip()
        q = q.strip(".?! ")
        if q:
            words = q.split()
            clean_title = " ".join(w.capitalize() if w.lower() not in {"in", "of", "to", "for", "a", "an", "the", "and", "from", "with"} else w.lower() for w in words)
            if clean_title:
                clean_title = clean_title[0].upper() + clean_title[1:]
        else:
            clean_title = "Agent Document"

    clean_title = re.sub(r"^#+\s*", "", clean_title).strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", clean_title.lower()).strip("_")
    slug_parts = [p for p in slug.split("_") if p][:5]
    safe_name = "_".join(slug_parts) if slug_parts else "document"
    return clean_title, cleaned_content, safe_name


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pdf_chrome(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor(f"#{_TEAL}"))
    canvas.rect(0, height - 0.38 * inch, width, 0.38 * inch, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor(f"#{_INK}"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.68 * inch, 0.42 * inch, "Agent OTG | Local document")
    canvas.drawRightString(width - 0.68 * inch, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def _generate_pdf(title: str, content: str, path: Path) -> None:
    base = getSampleStyleSheet()
    title_style = ParagraphStyle("AgentTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=colors.HexColor(f"#{_INK}"), spaceAfter=18)
    heading_style = ParagraphStyle("AgentHeading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor(f"#{_TEAL}"), spaceBefore=12, spaceAfter=7)
    body_style = ParagraphStyle("AgentBody", parent=base["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, textColor=colors.HexColor(f"#{_INK}"), spaceAfter=9)
    bullet_style = ParagraphStyle("AgentBullet", parent=body_style, leftIndent=16, firstLineIndent=-10)
    code_style = ParagraphStyle("AgentCode", fontName="Courier", fontSize=8.5, leading=11, backColor=colors.HexColor("#F1F5F4"), borderColor=colors.HexColor("#C5D7D2"), borderWidth=0.5, borderPadding=8, spaceBefore=4, spaceAfter=10)
    story = [Spacer(1, 0.18 * inch), Paragraph(_escape(title), title_style)]
    code_open = False
    code_lines: list[str] = []
    for raw in (content or "").splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if code_open:
                story.append(Preformatted("\n".join(code_lines) or " ", code_style))
                code_lines = []
            code_open = not code_open
            continue
        if code_open:
            code_lines.append(line)
            continue
        if not line.strip():
            story.append(Spacer(1, 3))
        elif line.lstrip().startswith("#"):
            story.append(Paragraph(_escape(line.lstrip("# ")), heading_style))
        elif line.lstrip().startswith(("- ", "* ")):
            story.append(Paragraph(_escape(line.lstrip()[2:]), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(_escape(line), body_style))
    if code_lines:
        story.append(Preformatted("\n".join(code_lines), code_style))
    SimpleDocTemplate(str(path), pagesize=A4, title=title, leftMargin=0.68 * inch, rightMargin=0.68 * inch, topMargin=0.72 * inch, bottomMargin=0.7 * inch).build(story, onFirstPage=_pdf_chrome, onLaterPages=_pdf_chrome)


def _set_cell_shading(cell, fill: str) -> None:
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shade)


def _generate_docx(title: str, content: str, path: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin, section.bottom_margin = Inches(0.72), Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size, normal.font.color.rgb = "Aptos", Pt(10.5), RGBColor.from_string(_INK)
    normal.paragraph_format.space_after = Pt(8)
    heading = doc.styles["Heading 1"]
    heading.font.name, heading.font.size, heading.font.color.rgb = "Aptos Display", Pt(16), RGBColor.from_string(_TEAL)
    title_p = doc.add_paragraph(style="Title")
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = title_p.add_run(title)
    run.font.name, run.font.size, run.font.bold, run.font.color.rgb = "Aptos Display", Pt(24), True, RGBColor.from_string(_INK)
    for paragraph in _paragraphs(content):
        if paragraph.startswith("#"):
            doc.add_heading(paragraph.lstrip("# "), level=1)
        elif paragraph.startswith(("- ", "* ")):
            for line in paragraph.splitlines():
                doc.add_paragraph(line.lstrip("-* "), style="List Bullet")
        else:
            doc.add_paragraph(paragraph)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Agent OTG | Local document")
    footer_run.font.size, footer_run.font.color.rgb = Pt(8), RGBColor.from_string("66747A")
    doc.save(path)


def _generate_pptx(title: str, content: str, path: Path) -> None:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor as PptRGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches as PptInches, Pt as PptPt
    except ImportError as exc:
        raise RuntimeError("PPTX generation requires python-pptx. Install project requirements first.") from exc
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = PptInches(13.333), PptInches(7.5)
    sections = _paragraphs(content) or ["No content supplied"]
    for index, section in enumerate(sections):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        background = slide.background.fill
        background.solid(); background.fore_color.rgb = PptRGBColor(246, 248, 247)
        band = slide.shapes.add_shape(1, 0, 0, presentation.slide_width, PptInches(0.28))
        band.fill.solid(); band.fill.fore_color.rgb = PptRGBColor(8, 126, 112); band.line.fill.background()
        heading = slide.shapes.add_textbox(PptInches(0.75), PptInches(0.7), PptInches(11.8), PptInches(0.8)).text_frame
        heading.clear(); title_run = heading.paragraphs[0].add_run(); title_run.text = title if index == 0 else section.splitlines()[0].lstrip("# ")
        title_run.font.name, title_run.font.size, title_run.font.bold, title_run.font.color.rgb = "Aptos Display", PptPt(28), True, PptRGBColor(24, 33, 43)
        body = slide.shapes.add_textbox(PptInches(0.85), PptInches(1.75), PptInches(11.3), PptInches(4.8)).text_frame
        body.clear(); body.word_wrap = True
        lines = section.splitlines() if index else [line for line in section.splitlines() if line.strip()]
        for position, line in enumerate(lines):
            paragraph = body.paragraphs[0] if position == 0 else body.add_paragraph()
            paragraph.text = line.lstrip("- ")
            paragraph.level = 0
            paragraph.font.name, paragraph.font.size, paragraph.font.color.rgb = "Aptos", PptPt(17), PptRGBColor(24, 33, 43)
            paragraph.space_after = PptPt(10)
        footer = slide.shapes.add_textbox(PptInches(0.85), PptInches(7.0), PptInches(11.3), PptInches(0.2)).text_frame.paragraphs[0]
        footer.text, footer.alignment = f"Agent OTG | {index + 1}", PP_ALIGN.RIGHT
        footer.font.size, footer.font.color.rgb = PptPt(8), PptRGBColor(102, 116, 122)
    presentation.save(path)


def _generate_xlsx(title: str, content: str, path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = re.sub(r"[\\/*?:\[\]]", "", title)[:31] or "Sheet1"
    rows = _rows(content)
    for row in rows:
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = Font(name="Aptos", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=_TEAL)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in range(1, sheet.max_column + 1):
        values = [str(sheet.cell(row, column).value or "") for row in range(1, sheet.max_row + 1)]
        sheet.column_dimensions[get_column_letter(column)].width = min(max(max(map(len, values), default=10) + 2, 12), 42)
    workbook.save(path)
    if load_workbook(path, read_only=True).active.max_row < 1:
        raise RuntimeError("Generated XLSX has no rows")


def _media_type(extension: str) -> str:
    return {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "csv": "text/csv", "json": "application/json", "md": "text/markdown", "txt": "text/plain"}[extension]


def create_artifact(title: str, content: str, file_format: str, filename: str | None = None) -> Artifact:
    extension = str(file_format).lower().lstrip(".")
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{file_format}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")

    # Prevent prompt strings from becoming the document title
    t_lower = (title or "").lower().strip()
    if any(t_lower.startswith(p) for p in ["give me", "write a", "write code", "create a", "generate a", "generate the", "/agent", "please"]) or "also generate" in t_lower:
        clean_t, clean_c, safe_slug = derive_clean_title(title, content)
        title = clean_t
        content = clean_c
        if not filename or any(filename.lower().startswith(p) for p in ["give me", "write a", "create a", "/agent"]):
            filename = f"{safe_slug}.{extension}"

    path = _safe_filename(filename or title or "output", extension)
    if extension == "pdf":
        _generate_pdf(title, content, path)
        if len(PdfReader(str(path)).pages) < 1:
            raise RuntimeError("Generated PDF has no pages")
    elif extension == "docx":
        _generate_docx(title, content, path)
        if not Document(path).paragraphs:
            raise RuntimeError("Generated DOCX has no paragraphs")
    elif extension == "pptx":
        _generate_pptx(title, content, path)
    elif extension == "xlsx":
        _generate_xlsx(title, content, path)
    elif extension == "csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
    elif extension == "json":
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {"title": title, "content": content}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Artifact validation failed for {path.name}")
    return Artifact(path=path, media_type=_media_type(extension), size_bytes=path.stat().st_size)


def resolve_artifact(filename: str) -> Path:
    path = (OUTPUT_DIR / Path(filename).name).resolve()
    if path.parent != OUTPUT_DIR or not path.is_file():
        raise FileNotFoundError(filename)
    return path
