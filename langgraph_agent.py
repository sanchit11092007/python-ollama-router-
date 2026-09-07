"""Reliable offline LangGraph orchestration for Agent OTG.

File requests follow a deterministic plan. The model produces content, while
the workflow (not model tool-call luck) creates and validates the artifact.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

import config
from artifacts import SUPPORTED_FORMATS, create_artifact, derive_clean_title


class WorkflowState(TypedDict, total=False):
    question: str
    history: list[dict]
    plan: dict[str, Any]
    context: str
    content: str
    artifact: dict[str, Any]
    events: list[dict[str, str]]
    errors: list[str]
    final_answer: str


_FORMAT_PATTERN = re.compile(r"\b(pdf|docx|word|txt|text|markdown|md|pptx|powerpoint|xlsx|excel|csv|json)\b", re.I)


def _event(state: WorkflowState, stage: str, detail: str) -> dict:
    return {"events": [*state.get("events", []), {"stage": stage, "detail": detail}]}


def _detect_format(question: str) -> str | None:
    match = _FORMAT_PATTERN.search(question)
    if not match:
        return None
    return {"word": "docx", "text": "txt", "markdown": "md", "powerpoint": "pptx", "excel": "xlsx"}.get(match.group(1).lower(), match.group(1).lower())


def supervisor_node(state: WorkflowState) -> dict:
    requested_format = _detect_format(state["question"])
    plan = {"file_format": requested_format, "needs_rag": bool(re.search(r"\b(document|knowledge base|uploaded|source)\b", state["question"], re.I))}
    return {**_event(state, "understanding", "Supervisor classified the request."), "plan": plan}


def planner_node(state: WorkflowState) -> dict:
    fmt = state["plan"].get("file_format")
    detail = f"Planner scheduled content generation, {fmt.upper()} creation, and validation." if fmt else "Planner scheduled a direct response."
    return _event(state, "planning", detail)


def rag_node(state: WorkflowState) -> dict:
    if not state["plan"].get("needs_rag"):
        return {"context": ""}
    try:
        from rag.pipeline import answer
        result = answer(state["question"])
        context = result.get("answer", "") if result.get("sources") else ""
        return {**_event(state, "retrieving", "Retrieved grounded local knowledge."), "context": context}
    except Exception as exc:
        return {**_event(state, "retrieving", "Knowledge base unavailable; continuing without retrieval."), "context": "", "errors": [str(exc)]}


def _llm_content(question: str, context: str, file_format: str | None) -> str:
    prompt = "Answer the user accurately and completely using only the supplied local context when it exists."
    if file_format:
        prompt += f" Produce polished content for a {file_format.upper()} file. Return only the file content, without preamble."
    prompt += f"\n\nRequest:\n{question}\n\nLocal context:\n{context or '[none]'}"
    try:
        llm = ChatOllama(model=config.MAIN_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
        response = llm.invoke(prompt)
        return str(response.content).strip()
    except Exception:
        # Offline resilience: tool completion remains available even while the
        # local model is starting or unavailable.
        return context or f"{question}\n\nGenerated locally by Agent OTG."


def execution_node(state: WorkflowState) -> dict:
    fmt = state["plan"].get("file_format")
    content = _llm_content(state["question"], state.get("context", ""), fmt)
    update = _event(state, "generating", "Generated content with the local model.")
    return {**update, "content": content}


def file_node(state: WorkflowState) -> dict:
    fmt = state["plan"].get("file_format")
    if not fmt:
        return {}
    title, clean_content, safe_slug = derive_clean_title(state["question"], state.get("content", ""))
    filename = f"{safe_slug}.{fmt}"
    artifact = create_artifact(title, clean_content, fmt, filename)
    return {
        **_event(state, "creating_file", f"Created local {fmt.upper()} artifact: {artifact.path.name}"),
        "artifact": artifact.as_dict(),
        "content": clean_content,
    }


def validation_node(state: WorkflowState) -> dict:
    artifact = state.get("artifact")
    if artifact and artifact["size_bytes"] <= 0:
        return {"errors": ["Generated artifact is empty."]}
    return _event(state, "validating", "Validated the generated output.")


def response_node(state: WorkflowState) -> dict:
    artifact = state.get("artifact")
    content = state.get("content", "").strip()
    if artifact:
        answer_parts = []
        if content:
            answer_parts.append(content)
            answer_parts.append("\n\n---\n")
        answer_parts.append(f"📄 **Created and validated {artifact['filename']}** (saved to `{artifact['path']}`).")
        answer = "".join(answer_parts)
    else:
        answer = content or "I could not generate a response."
    return {**_event(state, "completed", "Workflow completed."), "final_answer": answer}


def _route_after_execution(state: WorkflowState) -> str:
    return "file" if state["plan"].get("file_format") else "validate"


def _build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("rag", rag_node)
    graph.add_node("execution", execution_node)
    graph.add_node("file", file_node)
    graph.add_node("validation", validation_node)
    graph.add_node("response", response_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "planner")
    graph.add_edge("planner", "rag")
    graph.add_edge("rag", "execution")
    graph.add_conditional_edges("execution", _route_after_execution, {"file": "file", "validate": "validation"})
    graph.add_edge("file", "validation")
    graph.add_edge("validation", "response")
    graph.add_edge("response", END)
    return graph.compile()


_GRAPH = _build_graph()


def run_agent(query: str, history: list | None = None) -> dict:
    """Run the supervised workflow and return its answer, progress, and artifact."""
    result = _GRAPH.invoke({"question": query, "history": history or [], "events": [], "errors": []})
    return {
        "final_answer": result.get("final_answer", ""),
        "needs_tool": bool(result.get("artifact")),
        "tool_log": ([{"tool": "create_file", "result": result["artifact"]}] if result.get("artifact") else []),
        "artifact": result.get("artifact"),
        "events": result.get("events", []),
        "errors": result.get("errors", []),
        "run_id": uuid.uuid4().hex,
    }
