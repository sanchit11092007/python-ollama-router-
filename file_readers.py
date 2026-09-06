import os
import json
import pandas as pd
import pymupdf as fitz  # PyMuPDF (fitz alias kept for compatibility)
from docx import Document


def convert_excel_to_json(file_path: str, sheet_name=0) -> str:
    """Reads an Excel file and returns it as a clean JSON string."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.where(pd.notnull(df), None)
    json_data = df.to_json(orient="records", date_format="iso", force_ascii=False)
    parsed = json.loads(json_data)
    return json.dumps(parsed, indent=2)


def convert_pdf_to_json(file_path: str) -> str:
    """Extracts text from a PDF, page by page, as structured JSON."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    doc = fitz.open(file_path)
    document_data = {
        "metadata": doc.metadata,
        "total_pages": len(doc),
        "pages": [],
    }
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        document_data["pages"].append({
            "page_number": page_num + 1,
            "content": text if text else None,
        })
    doc.close()
    return json.dumps(document_data, indent=2, ensure_ascii=False)


def process_word(file_path: str) -> str:
    """Reads a Word document and returns all its text as one string."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _clean_path(path: str) -> str:
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


def process_file(file_path: str) -> str:
    """
    Looks at a file's extension and reads it with the right function above.
    Returns plain text/JSON that can be handed to any of your AI models.
    """
    file_path = _clean_path(file_path)
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    extension = file_path.lower()

    if extension.endswith(".xlsx") or extension.endswith(".xls"):
        return convert_excel_to_json(file_path)
    elif extension.endswith(".docx"):
        return process_word(file_path)
    elif extension.endswith(".pdf"):
        return convert_pdf_to_json(file_path)
    else:
        return f"Error: unsupported file type for {file_path}"
