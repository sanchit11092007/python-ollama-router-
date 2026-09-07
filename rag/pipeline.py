from __future__ import annotations

import json
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from .config import (
    ENABLE_QUERY_EXPANSION,
    OLLAMA_BASE_URL,
    RAG_LLM_MODEL,
)
from .loaders import load_file, load_files
from .prompts import RAG_PROMPT
from .retriever import retrieve, retrieve_expanded
from .splitter import split_documents
from .vectorstore import add_documents, delete_source


# ── LLM (cached) ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatOllama:
    return ChatOllama(model=RAG_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def _get_chain():
    return RAG_PROMPT | _get_llm() | StrOutputParser()


# ── Query expansion ───────────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """
    Use the LLM to rewrite the user query into 2 semantic variants.

    This significantly improves retrieval recall because the same information
    can be expressed in many ways in source documents. Falls back to the
    original query if the LLM call fails or returns invalid JSON.

    Example:
        Input:  "refund policy"
        Output: ["refund policy",
                 "return and refund conditions",
                 "how to get a refund or return a product"]
    """
    if not ENABLE_QUERY_EXPANSION:
        return [query]

    prompt = (
        "Generate exactly 2 alternative phrasings of this search query to help "
        "find relevant information in a document knowledge base.\n"
        "Reply with ONLY a JSON array of 2 strings, no explanation, no markdown.\n"
        f"Original query: {query}\n"
        'Example format: ["alternative phrasing one", "alternative phrasing two"]'
    )
    try:
        response = _get_llm().invoke(prompt)
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        variants = json.loads(raw)
        if isinstance(variants, list) and variants:
            all_q = [query] + [str(v).strip() for v in variants if str(v).strip()]
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = []
            for q in all_q:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique[:3]   # original + up to 2 variants
    except Exception:
        pass
    return [query]


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    if not docs:
        return "[NO_RELEVANT_CONTEXT]"
    blocks = []
    for i, doc in enumerate(docs, 1):
        m      = doc.metadata
        source = m.get("file_name") or m.get("source", "unknown")
        page   = m.get("page")
        location = f", page {page}" if page else ""
        blocks.append(f"[{i}] source: {source}{location}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_paths(paths: list[str], replace_existing: bool = True) -> dict:
    raw_docs = load_files(paths)
    if replace_existing:
        sources = {d.metadata.get("source") for d in raw_docs if d.metadata.get("source")}
        for source in sources:
            delete_source(source)
    chunks = split_documents(raw_docs)
    count  = add_documents(chunks)
    return {
        "files":            len(paths),
        "documents_loaded": len(raw_docs),
        "chunks_indexed":   count,
    }


def ingest_path(path: str, replace_existing: bool = True) -> dict:
    return ingest_paths([path], replace_existing=replace_existing)


# ── RAG Answer (with query expansion loop) ───────────────────────────────────

def answer(
    query: str,
    access_filter: dict | None = None,
    strategy: str | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Full RAG pipeline with optimized query expansion loop:

      1. Expand query → 1–3 semantic variants
      2. Retrieve for ALL variants, deduplicate by content hash
      3. If expansion gave 0 docs, fall back to direct similarity search
      4. Format context and invoke the LLM chain
      5. Return answer + sources + strategy used

    The expansion loop dramatically improves recall without requiring
    MultiQueryRetriever or any extra dependencies.
    """
    # ── Per-request top_k override (backward compat with search_documents tool) ──
    from . import config as _c
    old_top_k = None
    if top_k is not None:
        old_top_k    = _c.TOP_K
        _c.TOP_K     = max(1, min(int(top_k), 20))

    try:
        # Step 1 — Expand query into semantic variants
        queries = _expand_query(query)

        # Step 2 — Multi-query retrieval with deduplication
        if len(queries) > 1:
            docs, selected_strategy = retrieve_expanded(
                queries, strategy=strategy, access_filter=access_filter
            )
        else:
            docs, selected_strategy = retrieve(
                query, strategy=strategy, access_filter=access_filter
            )

        # Step 3 — Fallback: if expanded retrieval found nothing, try plain similarity
        if not docs and len(queries) > 1:
            docs, selected_strategy = retrieve(
                query, strategy="similarity", access_filter=access_filter
            )

    finally:
        if old_top_k is not None:
            _c.TOP_K = old_top_k

    # Step 4 — No docs at all → early return
    if not docs:
        return {
            "answer":             "I couldn't find sufficient information in the provided knowledge base.",
            "sources":            [],
            "retrieval_strategy": selected_strategy,
            "documents":          [],
        }

    # Step 5 — Invoke LLM with formatted context
    result = _get_chain().invoke({"question": query, "context": _format_docs(docs)})

    sources = []
    for d in docs:
        sources.append({
            "source":      d.metadata.get("file_name") or d.metadata.get("source", "unknown"),
            "page":        d.metadata.get("page"),
            "document_id": d.metadata.get("document_id"),
            "chunk_index": d.metadata.get("chunk_index"),
        })

    return {
        "answer":             result,
        "sources":            sources,
        "retrieval_strategy": selected_strategy,
        "documents":          docs,
    }
