import os

def _clean_path(path: str) -> str:
    """Clean surrounding quotes and whitespace from paths."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)

# ── write_docx tool ──────────────────────────────────────────────────────────
# pip install python-docx
from docx import Document

_DOCX_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_files")
os.makedirs(_DOCX_OUTPUT_DIR, exist_ok=True)


def write_docx(title: str, content: str, filename: str = "output.docx") -> str:
    safe_name = os.path.basename(_clean_path(filename))
    if not safe_name.lower().endswith(".docx"):
        safe_name += ".docx"
    output_path = os.path.join(_DOCX_OUTPUT_DIR, safe_name)
    doc = Document()
    doc.add_heading(title, level=1)
    for paragraph in content.split("\n\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph)
    doc.save(output_path)
    return f"Saved document to {output_path}"


# ── ChromaDB / RAG tools ─────────────────────────────────────────────────────
# pip install chromadb
import chromadb

try:
    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    _collection    = _chroma_client.get_or_create_collection("knowledge_base")
    _chroma_ready  = True
except Exception as _chroma_err:
    _chroma_ready = False
    print(f"[tools.py] WARNING: ChromaDB failed to initialise: {_chroma_err}")
    print("[tools.py]          search_documents / ingest_document will return errors.")


def search_documents(query: str, n_results: int = 3) -> str:
    if not _chroma_ready:
        return "Knowledge base unavailable (ChromaDB failed to initialise)."
    results = _collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No relevant documents found."
    return "\n---\n".join(docs)


def ingest_document(text: str, source: str, chunk_size: int = 500) -> str:
    if not _chroma_ready:
        return "Knowledge base unavailable (ChromaDB failed to initialise)."
    chunks    = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    ids       = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    _collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return f"Ingested {len(chunks)} chunk(s) from '{source}' into the knowledge base."


# ── PDF tools ────────────────────────────────────────────────────────────────
# pip install pdfplumber pypdf


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


def merge_pdfs(pdf_paths, output_filename: str = "merged.pdf") -> str:
    from pypdf import PdfWriter
    # Handle string passed as JSON array or comma separated
    if isinstance(pdf_paths, str):
        import json
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
        safe_name = os.path.basename(_clean_path(output_filename))
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        output_path = os.path.join(_DOCX_OUTPUT_DIR, safe_name)
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"Merged {len(cleaned_paths)} PDF(s) into {output_path}"
    except Exception as exc:
        return f"Error merging PDFs: {exc}"


def generate_pdf_from_text(title: str, content: str, filename: str = "output.pdf") -> str:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    safe_name = os.path.basename(_clean_path(filename))
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    output_path = os.path.join(_DOCX_OUTPUT_DIR, safe_name)
    
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    
    for paragraph in content.split("\n\n"):
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), styles['Normal']))
            story.append(Spacer(1, 12))
            
    doc.build(story)
    return f"Saved PDF to {output_path}"


def split_pdf(pdf_path: str, output_folder: str) -> str:
    from pypdf import PdfReader, PdfWriter
    clean_p = _clean_path(pdf_path)
    clean_out = _clean_path(output_folder)
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


def rotate_pdf_pages(pdf_path: str, degrees, output_filename: str) -> str:
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
        
        safe_name = os.path.basename(_clean_path(output_filename))
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        output_path = os.path.join(_DOCX_OUTPUT_DIR, safe_name)
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



# ── Registry ─────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict = {
    "write_docx":       write_docx,
    "search_documents": search_documents,
    "ingest_document":  ingest_document,
    "extract_pdf_text": extract_pdf_text,
    "merge_pdfs":       merge_pdfs,
    "generate_pdf_from_text": generate_pdf_from_text,
    "split_pdf":        split_pdf,
    "rotate_pdf_pages": rotate_pdf_pages,
    "get_pdf_page_count": get_pdf_page_count,
}

TOOL_SCHEMAS: list = [
    {"type": "function", "function": {
        "name": "write_docx",
        "description": "Write a Word (.docx) document with a title and body content and save it to disk.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string",  "description": "Document heading or title"},
            "content":  {"type": "string",  "description": "Main body text, use blank lines between paragraphs"},
            "filename": {"type": "string",  "description": "Output filename, e.g. report.docx"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the internal knowledge base for passages relevant to a query.",
        "parameters": {"type": "object", "properties": {
            "query":     {"type": "string",  "description": "Natural-language search query"},
            "n_results": {"type": "integer", "description": "Number of results to return (default 3)"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "ingest_document",
        "description": "Add a piece of text to the knowledge base so it can be searched later.",
        "parameters": {"type": "object", "properties": {
            "text":       {"type": "string",  "description": "The full text to ingest"},
            "source":     {"type": "string",  "description": "A label identifying the source"},
            "chunk_size": {"type": "integer", "description": "Characters per chunk (default 500)"},
        }, "required": ["text", "source"]},
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
            "pdf_paths":       {"type": "array", "items": {"type": "string"}, "description": "List of paths to PDFs, in order"},
            "output_filename": {"type": "string", "description": "Filename for the merged PDF"},
        }, "required": ["pdf_paths"]},
    }},
    {"type": "function", "function": {
        "name": "generate_pdf_from_text",
        "description": "Generate a PDF document from plain text with a title.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string",  "description": "Document title"},
            "content":  {"type": "string",  "description": "Main body text, separate paragraphs with blank lines"},
            "filename": {"type": "string",  "description": "Output filename, e.g. report.pdf"},
        }, "required": ["title", "content"]},
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
        "description": "Rotate all pages in a PDF by a given number of degrees.",
        "parameters": {"type": "object", "properties": {
            "pdf_path":        {"type": "string",  "description": "Path to the input PDF"},
            "degrees":         {"type": "integer", "description": "Degrees to rotate (e.g., 90, 180, 270)"},
            "output_filename": {"type": "string",  "description": "Output filename for rotated PDF"},
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
