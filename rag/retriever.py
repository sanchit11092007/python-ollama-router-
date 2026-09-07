from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import ENABLE_MULTI_QUERY, FETCH_K, MAX_CONTEXT_DOCS, OLLAMA_BASE_URL, RAG_LLM_MODEL, TOP_K
from .vectorstore import get_vector_store


@lru_cache(maxsize=1)
def get_query_llm():
    return ChatOllama(model=RAG_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def choose_strategy(query: str) -> str:
    q = query.lower().strip()
    if any(x in q for x in ["compare", "difference", "different", "alternatives", "pros and cons", "various", "all methods", "types of"]):
        return "mmr"
    if any(x in q for x in ["ambiguous", "another way", "different wording", "what does this mean", "why does", "how does"]):
        return "multi_query" if ENABLE_MULTI_QUERY else "similarity"
    # Precise identifiers/numbers are usually better served by direct similarity.
    if re.search(r"\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\d{4,}\b", query):
        return "similarity"
    if len(q.split()) > 22:
        return "multi_query" if ENABLE_MULTI_QUERY else "mmr"
    return "similarity"


def _base_retriever(strategy: str):
    store = get_vector_store()
    if strategy == "mmr":
        return store.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})
    return store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


def retrieve(query: str, strategy: str | None = None, access_filter: dict | None = None) -> tuple[list[Document], str]:
    strategy = strategy or choose_strategy(query)
    try:
        if strategy == "multi_query" and ENABLE_MULTI_QUERY:
            from langchain_community.retrievers import MultiQueryRetriever
            base = _base_retriever("mmr")
            if access_filter:
                base = get_vector_store().as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "filter": access_filter})
            retriever = MultiQueryRetriever.from_llm(retriever=base, llm=get_query_llm())
        else:
            retriever = _base_retriever(strategy)
            if access_filter:
                retriever = get_vector_store().as_retriever(
                    search_type="mmr" if strategy == "mmr" else "similarity",
                    search_kwargs={"k": TOP_K, **({"fetch_k": FETCH_K} if strategy == "mmr" else {}), "filter": access_filter},
                )
        docs = retriever.invoke(query)
        return docs[:MAX_CONTEXT_DOCS], strategy
    except Exception:
        # Reliability rule: advanced retrieval must never make RAG unavailable.
        try:
            fallback = _base_retriever("similarity")
            if access_filter:
                fallback = get_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": TOP_K, "filter": access_filter})
            return fallback.invoke(query)[:MAX_CONTEXT_DOCS], "similarity_fallback"
        except Exception:
            return [], "unavailable"


def retrieve_expanded(
    queries: list[str],
    strategy: str | None = None,
    access_filter: dict | None = None,
) -> tuple[list[Document], str]:
    """
    Retrieve documents for multiple query variants and merge results,
    deduplicating by MD5 hash of page_content.

    This is the backend for the query-expansion optimization loop in pipeline.py.
    It gives much higher recall than a single-query retrieval because the same
    information may be phrased differently across source documents.
    """
    seen: set[str] = set()
    all_docs: list[Document] = []
    last_strategy = "similarity"

    for q in queries:
        try:
            docs, strat = retrieve(q, strategy=strategy, access_filter=access_filter)
            last_strategy = strat
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode("utf-8", errors="replace")).hexdigest()
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        except Exception:
            continue

    return all_docs[:MAX_CONTEXT_DOCS], last_strategy or "unavailable"
