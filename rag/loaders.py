"""File -> LangChain Document normalization.

We deliberately return Documents, not JSON strings. Embedding models need text;
JSON is useful as a transport/structured-data format but is not a prerequisite.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").replace("\x00", "").splitlines()]
    # Preserve paragraph boundaries while removing excessive blank lines.
    out, blank = [], 0
    for line in lines:
        if line.strip():
            out.append(line.strip())
            blank = 0
        elif blank == 0:
            out.append("")
            blank = 1
    return "\n".join(out).strip()


def _base_metadata(path: str) -> dict:
    p = Path(path)
    stat = p.stat()
    return {
        "source": str(p),
        "file_name": p.name,
        "file_type": p.suffix.lower().lstrip("."),
        "document_id": hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:24],
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime,
    }


def _load_pdf(path: str) -> list[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    docs = PyMuPDFLoader(path).load()
    base = _base_metadata(path)
    out = []
    for i, d in enumerate(docs, 1):
        text = _clean_text(d.page_content)
        if not text:
            continue
        meta = {**base, **d.metadata, "page": i, "content_type": "text"}
        out.append(Document(page_content=text, metadata=meta))
    return out


def _load_docx(path: str) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    docs = Docx2txtLoader(path).load()
    base = _base_metadata(path)
    return [
        Document(page_content=_clean_text(d.page_content), metadata={**base, **d.metadata, "content_type": "text"})
        for d in docs if _clean_text(d.page_content)
    ]


def _load_text(path: str) -> list[Document]:
    from langchain_community.document_loaders import TextLoader
    docs = TextLoader(path, autodetect_encoding=True).load()
    base = _base_metadata(path)
    return [
        Document(page_content=_clean_text(d.page_content), metadata={**base, "content_type": "text"})
        for d in docs if _clean_text(d.page_content)
    ]


def _load_json(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return [Document(page_content=text, metadata={**_base_metadata(path), "content_type": "json"})]


def _load_csv(path: str) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader
    docs = CSVLoader(path).load()
    base = _base_metadata(path)
    return [
        Document(page_content=_clean_text(d.page_content), metadata={**base, **d.metadata, "content_type": "table"})
        for d in docs if _clean_text(d.page_content)
    ]


def _load_excel(path: str) -> list[Document]:
    # Tables are kept as meaningful text for embeddings. We do NOT stringify a
    # whole workbook into one giant JSON blob.
    import pandas as pd
    workbook = pd.ExcelFile(path)
    base = _base_metadata(path)
    out: list[Document] = []
    for sheet in workbook.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.dropna(how="all")
        if df.empty:
            continue
        df = df.where(pd.notnull(df), "")
        headers = [str(c) for c in df.columns]
        for row_number, row in enumerate(df.itertuples(index=False, name=None), 2):
            pairs = [f"{headers[i]}: {row[i]}" for i in range(len(headers)) if str(row[i]).strip()]
            if pairs:
                text = f"Sheet: {sheet}\nRow: {row_number}\n" + "\n".join(pairs)
                out.append(Document(
                    page_content=text,
                    metadata={**base, "sheet": str(sheet), "row": row_number, "content_type": "table"},
                ))
    return out


def _load_image(path: str) -> list[Document]:
    # Image bytes are not embedded directly in the text vector store. They are
    # converted to a searchable visual description by Qwen-VL during ingestion.
    import base64
    import ollama
    from .config import OLLAMA_BASE_URL, RAG_VISION_MODEL

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    client = ollama.Client(host=OLLAMA_BASE_URL)
    prompt = (
        "Describe this image for enterprise knowledge-base retrieval. "
        "Extract visible text, labels, tables, diagrams, entities, relationships, "
        "numbers, and important visual facts. Do not invent anything."
    )
    response = client.chat(model=RAG_VISION_MODEL, messages=[{
        "role": "user", "content": prompt, "images": [b64]
    }])
    text = _clean_text(response["message"]["content"])
    return [Document(page_content=text, metadata={**_base_metadata(path), "content_type": "image", "modality": "image"})] if text else []


def load_file(path: str) -> list[Document]:
    path = os.path.normpath(str(path).strip().strip('"').strip("'"))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = Path(path).suffix.lower()
    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_text,
        ".csv": _load_csv,
        ".json": _load_json,
        ".xlsx": _load_excel,
        ".xls": _load_excel,
        ".png": _load_image,
        ".jpg": _load_image,
        ".jpeg": _load_image,
        ".webp": _load_image,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](path)


def load_files(paths: Iterable[str]) -> list[Document]:
    all_docs: list[Document] = []
    for path in paths:
        all_docs.extend(load_file(path))
    return all_docs
