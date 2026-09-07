"""
tools.py - Real tool implementations callable by the LangGraph agent and router.

All output files are saved to the generated_files/ subdirectory.
Tools registered in TOOL_FUNCTIONS / TOOL_SCHEMAS at the bottom.

NOTE: ChromaDB / RAG tools have been removed (no-RAG mode).
"""

import os
import csv
import html as _html_module

# ── Output directory ──────────────────────────────────────────────────────────
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def _clean_path(path: str) -> str:
    """Strip surrounding quotes and whitespace from a path string."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


def _safe_filename(name: str, ext: str) -> str:
    """Return a safe output path under _OUTPUT_DIR with the given extension."""
    safe = os.path.basename(_clean_path(name)) if name else f"output{ext}"
    # Ensure correct extension
    if not safe.lower().endswith(ext):
        safe += ext
    return os.path.join(_OUTPUT_DIR, safe)


# ══════════════════════════════════════════════════════════════════════════════
# WORD DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

def write_docx(title: str, content: str, filename: str = "output.docx") -> str:
    """
    Write a Word (.docx) document.
    Parses basic Markdown: # headings, - bullet lists, ```code blocks```.
    """
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        return "Error: python-docx is not installed. Run: pip install python-docx"

    output_path = _safe_filename(filename, ".docx")
    doc = Document()

    if title and title.strip():
        doc.add_heading(title.strip(), level=1)

    in_code = False
    code_lines: list = []

    def _flush_code(lines: list):
        if lines:
            p = doc.add_paragraph("\n".join(lines))
            p.style = "No Spacing"
            for run in p.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(9)

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                _flush_code(code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        elif stripped:
            doc.add_paragraph(stripped)

    if in_code and code_lines:
        _flush_code(code_lines)

    try:
        doc.save(output_path)
        return f"✅ Word document saved: {output_path}"
    except Exception as exc:
        return f"Error saving Word document: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# PLAIN TEXT FILE
# ══════════════════════════════════════════════════════════════════════════════

def write_txt(title: str, content: str, filename: str = "output.txt") -> str:
    """Write a plain text file with an optional title header."""
    output_path = _safe_filename(filename, ".txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if title and title.strip():
                f.write(title.strip() + "\n")
                f.write("=" * len(title.strip()) + "\n\n")
            f.write(content)
        return f"✅ Text file saved: {output_path}"
    except Exception as exc:
        return f"Error saving text file: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# CSV FILE
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(headers: list, rows: list, filename: str = "output.csv") -> str:
    """
    Write a CSV file. headers is a list of column names.
    rows is a list of lists (each inner list is one row).
    """
    output_path = _safe_filename(filename, ".csv")
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return f"✅ CSV file saved: {output_path} ({len(rows)} rows)"
    except Exception as exc:
        return f"Error saving CSV file: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_styled_html(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: letter;
        margin: 2cm;
    }}
    body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.6;
        color: #1e293b;
    }}
    h1 {{
        font-size: 20pt;
        color: #0f172a;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
        margin-top: 0;
    }}
    h2 {{
        font-size: 14pt;
        color: #1e3a8a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 18px;
    }}
    h3 {{
        font-size: 11pt;
        color: #0f172a;
    }}
    blockquote {{
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 8px 14px;
        margin: 14px 0;
        color: #1e40af;
    }}
    table {{
        width: 100%;
        margin: 14px 0;
        border-collapse: collapse;
    }}
    th {{
        background-color: #1e293b;
        color: #ffffff;
        padding: 8px;
        text-align: left;
    }}
    td {{
        padding: 8px;
        border-bottom: 1px solid #e2e8f0;
    }}
    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px;
        font-family: Courier, monospace;
        font-size: 8.5pt;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
    code {{
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 4px;
        font-family: Courier, monospace;
        font-size: 9pt;
    }}
    ul, ol {{
        margin: 8px 0;
        padding-left: 20px;
    }}
    li {{
        margin: 4px 0;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def generate_pdf_from_text(title: str, content: str, filename: str = "output.pdf") -> str:
    """
    Generate a styled PDF document from plain text or Markdown content.
    Uses xhtml2pdf (rich formatting) if available, falls back to ReportLab.
    """
    output_path = _safe_filename(filename, ".pdf")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Attempt 1: xhtml2pdf with Markdown rendering ─────────────────────────
    try:
        import markdown as _md_lib
        from xhtml2pdf import pisa

        full_md = f"# {title}\n\n{content}" if title and not content.strip().startswith("#") else content
        body_html = _md_lib.markdown(full_md, extensions=["tables", "fenced_code"])
        html_content = _build_styled_html(body_html)

        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file, path=current_dir)

        if not pisa_status.err:
            return f"✅ Styled PDF saved: {output_path}"
        # Fall through to ReportLab if pisa reported an error
    except ImportError:
        pass  # xhtml2pdf/markdown not installed — try ReportLab
    except Exception:
        pass  # xhtml2pdf failed for some other reason — try ReportLab

    # ── Attempt 2: ReportLab fallback ─────────────────────────────────────────
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(
            output_path,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )
        styles = getSampleStyleSheet()
        code_style = ParagraphStyle(
            "Code",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=4,
            spaceAfter=4,
        )
        story = []

        if title and title.strip():
            story.append(Paragraph(_html_module.escape(title.strip()), styles["Title"]))
            story.append(Spacer(1, 12))

        in_code = False
        code_buf: list = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    if code_buf:
                        story.append(Preformatted("\n".join(code_buf), code_style))
                        story.append(Spacer(1, 6))
                    code_buf = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_buf.append(line)
                continue

            if stripped:
                if stripped.startswith("### "):
                    story.append(Paragraph(_html_module.escape(stripped[4:]), styles["Heading3"]))
                elif stripped.startswith("## "):
                    story.append(Paragraph(_html_module.escape(stripped[3:]), styles["Heading2"]))
                elif stripped.startswith("# "):
                    story.append(Paragraph(_html_module.escape(stripped[2:]), styles["Heading1"]))
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    story.append(Paragraph("• " + _html_module.escape(stripped[2:]), styles["Normal"]))
                else:
                    story.append(Paragraph(_html_module.escape(stripped), styles["Normal"]))
                story.append(Spacer(1, 4))

        if in_code and code_buf:
            story.append(Preformatted("\n".join(code_buf), code_style))

        doc.build(story)
        return f"✅ PDF saved (ReportLab): {output_path}"

    except ImportError:
        return "Error: Neither xhtml2pdf nor reportlab is installed. Run: pip install reportlab"
    except Exception as exc:
        return f"Error generating PDF: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# PDF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all readable text from a PDF file on disk."""
    try:
        import pdfplumber
    except ImportError:
        return "Error: pdfplumber is not installed. Run: pip install pdfplumber"

    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return f"Error: file not found — {clean_p}"
    try:
        with pdfplumber.open(clean_p) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        return text if text else "No text could be extracted from this PDF."
    except Exception as exc:
        return f"Error reading PDF: {exc}"


def merge_pdfs(pdf_paths, output_filename: str = "merged.pdf") -> str:
    """Merge multiple PDF files into a single PDF."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    import json as _json
    if isinstance(pdf_paths, str):
        try:
            parsed = _json.loads(pdf_paths)
            if isinstance(parsed, list):
                pdf_paths = parsed
        except Exception:
            pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]

    cleaned = [_clean_path(p) for p in pdf_paths]
    missing = [p for p in cleaned if not os.path.isfile(p)]
    if missing:
        return f"Error: files not found: {missing}"
    try:
        writer = PdfWriter()
        for path in cleaned:
            writer.append(path)
        output_path = _safe_filename(output_filename, ".pdf")
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"✅ Merged {len(cleaned)} PDF(s) into {output_path}"
    except Exception as exc:
        return f"Error merging PDFs: {exc}"


def split_pdf(pdf_path: str, output_folder: str) -> str:
    """Split a PDF into separate files, one per page."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    clean_p = _clean_path(pdf_path)
    clean_out = _clean_path(output_folder)
    if not os.path.isfile(clean_p):
        return f"Error: file not found — {clean_p}"
    os.makedirs(clean_out, exist_ok=True)
    try:
        reader = PdfReader(clean_p)
        base = os.path.splitext(os.path.basename(clean_p))[0]
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            out = os.path.join(clean_out, f"{base}_page_{i + 1}.pdf")
            with open(out, "wb") as f:
                writer.write(f)
        return f"✅ Split into {len(reader.pages)} page(s) in {clean_out}"
    except Exception as exc:
        return f"Error splitting PDF: {exc}"


def rotate_pdf_pages(pdf_path: str, degrees, output_filename: str) -> str:
    """Rotate all pages in a PDF by a given number of degrees."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return f"Error: file not found — {clean_p}"
    try:
        deg = int(degrees)
        reader = PdfReader(clean_p)
        writer = PdfWriter()
        for page in reader.pages:
            page.rotate(deg)
            writer.add_page(page)
        output_path = _safe_filename(output_filename, ".pdf")
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"✅ Rotated all pages by {deg}°, saved to {output_path}"
    except Exception as exc:
        return f"Error rotating PDF: {exc}"


def get_pdf_page_count(pdf_path: str) -> str:
    """Get the number of pages in a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return f"Error: file not found — {clean_p}"
    try:
        reader = PdfReader(clean_p)
        return f"✅ {len(reader.pages)} page(s) in '{os.path.basename(clean_p)}'"
    except Exception as exc:
        return f"Error reading PDF: {exc}"


def list_generated_files() -> str:
    """List all files in the generated_files directory."""
    try:
        files = os.listdir(_OUTPUT_DIR)
        if not files:
            return "The generated_files/ directory is currently empty."
        lines = [f"Files in generated_files/ ({len(files)} total):"]
        for fname in sorted(files):
            fpath = os.path.join(_OUTPUT_DIR, fname)
            size = os.path.getsize(fpath)
            lines.append(f"  • {fname}  ({size:,} bytes)")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing files: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY — imported by router.py and langgraph_agent.py
# NOTE: RAG/ChromaDB tools intentionally excluded (no-RAG mode).
# ══════════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS: dict = {
    "write_docx":             write_docx,
    "write_txt":              write_txt,
    "write_csv":              write_csv,
    "generate_pdf_from_text": generate_pdf_from_text,
    "extract_pdf_text":       extract_pdf_text,
    "merge_pdfs":             merge_pdfs,
    "split_pdf":              split_pdf,
    "rotate_pdf_pages":       rotate_pdf_pages,
    "get_pdf_page_count":     get_pdf_page_count,
    "list_generated_files":   list_generated_files,
}

TOOL_SCHEMAS: list = [
    {"type": "function", "function": {
        "name": "write_docx",
        "description": "Write a Word (.docx) document with a title and body content and save it to disk. Supports Markdown formatting (headings, bullets, code blocks).",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Document heading or title"},
            "content":  {"type": "string", "description": "Main body text (Markdown supported). Use blank lines between paragraphs."},
            "filename": {"type": "string", "description": "Output filename, e.g. report.docx"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "write_txt",
        "description": "Write a plain text (.txt) file with a title and body content.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Title shown at the top of the file"},
            "content":  {"type": "string", "description": "Main body text"},
            "filename": {"type": "string", "description": "Output filename, e.g. notes.txt"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "write_csv",
        "description": "Write data to a CSV file. Provide column headers and rows of data.",
        "parameters": {"type": "object", "properties": {
            "headers":  {"type": "array", "items": {"type": "string"}, "description": "List of column header names"},
            "rows":     {"type": "array", "items": {"type": "array"}, "description": "List of rows; each row is a list of values"},
            "filename": {"type": "string", "description": "Output filename, e.g. data.csv"},
        }, "required": ["headers", "rows"]},
    }},
    {"type": "function", "function": {
        "name": "generate_pdf_from_text",
        "description": "Generate a styled PDF document from plain text or Markdown content with a title. Use this whenever the user asks to save output as PDF.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Document title shown at the top of the PDF"},
            "content":  {"type": "string", "description": "Main body text or Markdown. Separate sections with blank lines."},
            "filename": {"type": "string", "description": "Output filename, e.g. report.pdf"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "extract_pdf_text",
        "description": "Extract all readable text from a PDF file on disk.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Path to the PDF file"},
        }, "required": ["pdf_path"]},
    }},
    {"type": "function", "function": {
        "name": "merge_pdfs",
        "description": "Merge multiple PDF files into a single PDF and save it to disk.",
        "parameters": {"type": "object", "properties": {
            "pdf_paths":       {"type": "array", "items": {"type": "string"}, "description": "List of PDF file paths in order"},
            "output_filename": {"type": "string", "description": "Filename for the merged PDF"},
        }, "required": ["pdf_paths"]},
    }},
    {"type": "function", "function": {
        "name": "split_pdf",
        "description": "Split a PDF into separate files, one per page.",
        "parameters": {"type": "object", "properties": {
            "pdf_path":      {"type": "string", "description": "Path to the input PDF"},
            "output_folder": {"type": "string", "description": "Folder to save the single-page PDFs"},
        }, "required": ["pdf_path", "output_folder"]},
    }},
    {"type": "function", "function": {
        "name": "rotate_pdf_pages",
        "description": "Rotate all pages in a PDF by a given number of degrees and save.",
        "parameters": {"type": "object", "properties": {
            "pdf_path":        {"type": "string",  "description": "Path to the input PDF"},
            "degrees":         {"type": "integer", "description": "Degrees to rotate (90, 180, or 270)"},
            "output_filename": {"type": "string",  "description": "Output filename for the rotated PDF"},
        }, "required": ["pdf_path", "degrees", "output_filename"]},
    }},
    {"type": "function", "function": {
        "name": "get_pdf_page_count",
        "description": "Get the number of pages in a PDF file.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Path to the PDF file"},
        }, "required": ["pdf_path"]},
    }},
    {"type": "function", "function": {
        "name": "list_generated_files",
        "description": "List all files that have been generated and saved in the output directory.",
        "parameters": {"type": "object", "properties": {}},
    }},
]
