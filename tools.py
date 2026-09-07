import os
import json
from artifacts import OUTPUT_DIR, SUPPORTED_FORMATS, create_artifact

def _clean_path(path: str) -> str:
    """Clean surrounding quotes and whitespace from paths."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


# ── write_docx tool ──────────────────────────────────────────────────────────
_OUTPUT_DIR = str(OUTPUT_DIR)
_DOCX_OUTPUT_DIR = _OUTPUT_DIR


def write_docx(title: str, content: str, filename: str = "output.docx") -> str:
    artifact = create_artifact(title, content, "docx", filename)
    return f"Saved document to {artifact.path}"


# ── RAG tools ────────────────────────────────────────────────────────────────
# IMPORTANT:
# The old direct Chroma implementation has intentionally been removed.
# RAG is now handled by the ./rag package. Imports are lazy so that the
# main application can still start and serve non-RAG requests if the RAG
# dependencies/model are unavailable.

def search_documents(query: str, n_results: int = 5) -> str:
    """Search the RAG knowledge base and return a grounded answer with sources."""
    try:
        from rag.pipeline import answer
        result = answer(query, top_k=n_results)
        sources = result.get("sources", [])

        if not sources:
            return result.get(
                "answer",
                "I couldn't find sufficient information in the provided knowledge base."
            )

        source_lines = []
        for source in sources:
            line = f"- {source.get('source', 'unknown')}"
            if source.get("page") is not None:
                line += f", page {source['page']}"
            if source.get("sheet") is not None:
                line += f", sheet {source['sheet']}"
            source_lines.append(line)

        return result["answer"] + "\n\nSources:\n" + "\n".join(source_lines)

    except Exception as exc:
        return (
            "Knowledge base is currently unavailable. "
            f"RAG error: {exc}"
        )


def ingest_document(text: str, source: str, chunk_size: int = 500) -> str:
    """
    Backward-compatible text-ingestion tool.

    `chunk_size` is retained only so existing callers/tool schemas do not
    break. The new RAG pipeline controls chunking automatically using its
    configured content-aware splitter.
    """
    try:
        from langchain_core.documents import Document
        from rag.splitter import split_documents
        from rag.vectorstore import add_documents

        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "file_name": source,
                "content_type": "text",
            },
        )
        chunks = split_documents([doc])
        count = add_documents(chunks)
        return (
            f"Ingested {count} chunk(s) from '{source}' into the "
            "knowledge base."
        )
    except Exception as exc:
        return f"Knowledge base unavailable. Ingestion error: {exc}"


def ingest_file(file_path: str, replace_existing: bool = True) -> str:
    """Ingest PDF/DOCX/TXT/MD/CSV/XLS/XLSX/JSON/image into the RAG KB."""
    try:
        from rag.pipeline import ingest_path
        result = ingest_path(file_path, replace_existing=replace_existing)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return f"File ingestion failed: {exc}"


# ── PDF tools ────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    import pdfplumber
    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return f"Error: file not found - {clean_p}"
    try:
        with pdfplumber.open(clean_p) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages).strip() or "No text could be extracted from this PDF."
    except Exception as exc:
        return f"Error reading PDF: {exc}"


def ocr_image(image_path: str) -> str:
    """Extract text from a local image using a locally installed Tesseract runtime."""
    path = _clean_path(image_path)
    if not os.path.isfile(path):
        return f"Error: file not found - {path}"
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(path)).strip()
        return text or "No readable text was found in the image."
    except ImportError:
        return "OCR unavailable. Install pytesseract and the local Tesseract OCR runtime."
    except Exception as exc:
        return f"OCR failed: {exc}"


def merge_pdfs(pdf_paths, output_filename: str = "merged.pdf") -> str:
    from pypdf import PdfWriter
    if isinstance(pdf_paths, str):
        try:
            parsed = json.loads(pdf_paths)
            if isinstance(parsed, list):
                pdf_paths = parsed
        except Exception:
            pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]

    cleaned_paths = [_clean_path(p) for p in pdf_paths]
    missing = [p for p in cleaned_paths if not os.path.isfile(p)]
    if missing:
        return f"Error: files not found: {missing}"
    try:
        writer = PdfWriter()
        for path in cleaned_paths:
            writer.append(path)
        safe_name = os.path.basename(_clean_path(output_filename or "merged.pdf")) or "merged.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        output_path = os.path.join(_OUTPUT_DIR, safe_name)
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"Merged {len(cleaned_paths)} PDF(s) into {output_path}"
    except Exception as exc:
        return f"Error merging PDFs: {exc}"


def generate_pdf_from_text(title: str, content: str, filename: str = "output.pdf") -> str:
    artifact = create_artifact(title, content, "pdf", filename)
    return f"Saved PDF to {artifact.path}"


def create_file(title: str, content: str, file_format: str, filename: str = "output") -> dict:
    """Create and validate a local PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, or JSON artifact."""
    artifact = create_artifact(title, content, file_format, filename)
    return artifact.as_dict()


def split_pdf(pdf_path: str, output_folder: str = "split_pages") -> str:
    from pypdf import PdfReader, PdfWriter
    clean_p = _clean_path(pdf_path)
    clean_out = _clean_path(output_folder or "split_pages")
    if not os.path.isabs(clean_out):
        clean_out = os.path.join(_OUTPUT_DIR, clean_out)
    if not os.path.isfile(clean_p):
        return f"Error: file not found - {clean_p}"
    os.makedirs(clean_out, exist_ok=True)
    try:
        reader = PdfReader(clean_p)
        base = os.path.splitext(os.path.basename(clean_p))[0]
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            out = os.path.join(clean_out, f"{base}_page_{i+1}.pdf")
            with open(out, "wb") as f:
                writer.write(f)
        return f"Split into {len(reader.pages)} pages in {clean_out}"
    except Exception as exc:
        return f"Error splitting PDF: {exc}"


def rotate_pdf_pages(pdf_path: str, degrees: int = 90, output_filename: str = "rotated.pdf") -> str:
    from pypdf import PdfReader, PdfWriter
    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return f"Error: file not found - {clean_p}"
    try:
        deg = int(degrees)
        reader = PdfReader(clean_p)
        writer = PdfWriter()
        for page in reader.pages:
            page.rotate(deg)
            writer.add_page(page)

        safe_name = os.path.basename(_clean_path(output_filename or "rotated.pdf")) or "rotated.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        output_path = os.path.join(_OUTPUT_DIR, safe_name)
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"Rotated pages by {deg} degrees, saved to {output_path}"
    except Exception as exc:
        return f"Error rotating PDF: {exc}"


def get_pdf_page_count(pdf_path: str) -> int:
    from pypdf import PdfReader
    clean_p = _clean_path(pdf_path)
    if not os.path.isfile(clean_p):
        return -1
    try:
        reader = PdfReader(clean_p)
        return len(reader.pages)
    except Exception:
        return -1


# ── File listing ─────────────────────────────────────────────────────────────
# Alias for backward-compatibility with ask.py's handle_list_files import


def list_generated_files() -> str:
    """Return a formatted string listing all files in generated_files/."""
    if not os.path.isdir(_DOCX_OUTPUT_DIR):
        return "No generated_files/ directory found."
    files = sorted(os.listdir(_DOCX_OUTPUT_DIR))
    if not files:
        return "No generated files yet. Use /agent to create Word docs, PDFs, etc."
    lines = []
    for fname in files:
        fpath = os.path.join(_DOCX_OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        size_str = f"{size:,} bytes" if size < 1024 else f"{size // 1024:,} KB"
        lines.append(f"  • {fname}  ({size_str})")
    return f"Generated files ({len(files)} total in generated_files/):\n" + "\n".join(lines)


# ── Registry ─────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict = {
    "create_file": create_file,
    "write_docx": write_docx,
    "search_documents": search_documents,
    "ingest_document": ingest_document,
    "ingest_file": ingest_file,
    "extract_pdf_text": extract_pdf_text,
    "ocr_image": ocr_image,
    "merge_pdfs": merge_pdfs,
    "generate_pdf_from_text": generate_pdf_from_text,
    "split_pdf": split_pdf,
    "rotate_pdf_pages": rotate_pdf_pages,
    "get_pdf_page_count": get_pdf_page_count,
}


TOOL_SCHEMAS: list = [
    {"type": "function", "function": {
        "name": "ocr_image",
        "description": "Extract readable text from a local image using offline Tesseract OCR.",
        "parameters": {"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Path to a PNG, JPEG, or WEBP image"},
        }, "required": ["image_path"]},
    }},
    {"type": "function", "function": {
        "name": "create_file",
        "description": "Create and validate a local PDF, DOCX, TXT, Markdown, PPTX, XLSX, CSV, or JSON artifact.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Document title"},
            "content": {"type": "string", "description": "Complete content for the file"},
            "file_format": {"type": "string", "enum": ["pdf", "docx", "txt", "md", "pptx", "xlsx", "csv", "json"]},
            "filename": {"type": "string", "description": "Output filename"},
        }, "required": ["title", "content", "file_format"]},
    }},
    {"type": "function", "function": {
        "name": "write_docx",
        "description": "Write a Word (.docx) document with a title and body content and save it to disk.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Document heading or title"},
            "content": {"type": "string", "description": "Main body text, use blank lines between paragraphs"},
            "filename": {"type": "string", "description": "Output filename, e.g. report.docx"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the internal knowledge base using the RAG pipeline and answer using retrieved context.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "n_results": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Maximum number of retrieved chunks"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "ingest_document",
        "description": "Add text to the internal knowledge base. The RAG pipeline automatically chooses chunking.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Full text to ingest"},
            "source": {"type": "string", "description": "Label identifying the source"},
        }, "required": ["text", "source"]},
    }},
    {"type": "function", "function": {
        "name": "ingest_file",
        "description": "Ingest a PDF, DOCX, TXT, Markdown, CSV, Excel, JSON, or supported image file into the knowledge base.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Path to the file on disk"},
            "replace_existing": {"type": "boolean", "description": "Replace previously indexed chunks from the same source"},
        }, "required": ["file_path"]},
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
            "pdf_paths": {"type": "array", "items": {"type": "string"}, "description": "List of paths to PDFs, in order"},
            "output_filename": {"type": "string", "description": "Filename for the merged PDF"},
        }, "required": ["pdf_paths"]},
    }},
    {"type": "function", "function": {
        "name": "generate_pdf_from_text",
        "description": "Generate a PDF document from plain text with a title.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Document title"},
            "content": {"type": "string", "description": "Main body text, separate paragraphs with blank lines"},
            "filename": {"type": "string", "description": "Output filename, e.g. report.pdf"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "split_pdf",
        "description": "Split a PDF into separate files, one per page.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Path to the input PDF"},
            "output_folder": {"type": "string", "description": "Folder to save the single-page PDFs"},
        }, "required": ["pdf_path", "output_folder"]},
    }},
    {"type": "function", "function": {
        "name": "rotate_pdf_pages",
        "description": "Rotate all pages in a PDF by a given number of degrees.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Path to the input PDF"},
            "degrees": {"type": "integer", "description": "Degrees to rotate (e.g., 90, 180, 270)"},
            "output_filename": {"type": "string", "description": "Output filename for rotated PDF"},
        }, "required": ["pdf_path", "degrees", "output_filename"]},
    }},
    {"type": "function", "function": {
        "name": "get_pdf_page_count",
        "description": "Get the number of pages in a PDF file.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Path to the PDF file"},
        }, "required": ["pdf_path"]},
    }},
]
