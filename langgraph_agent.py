"""
langgraph_agent.py — Reliable offline LangGraph orchestration for Agent OTG.

All output files are saved to ~/Downloads/AgentOTG/.

Key design decisions:
- Individual try/except error shielding around every tool call: if one tool fails,
  it returns a clean error string like "Tool 'X' failed: <reason>" instead of
  crashing the whole graph or returning nothing.
- Pre-validation of required arguments and types before calling functions to prevent TypeErrors.
- Tool logging through the router/main.py event callback system so all executions and
  failures are visible in the terminal.
- When an artifact is created: show ONLY the file confirmation card, never dump raw content
  unless explicitly requested.
- Code requests are routed to CODER_MODEL and forbidden from generating PDF compilation scripts.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

import config
from artifacts import derive_clean_title, OUTPUT_DIR
import tools


class WorkflowState(TypedDict, total=False):
    question: str
    history: list[dict]
    plan: dict[str, Any]
    context: str
    content: str
    answer_text: str
    artifact: dict[str, Any]
    events: list[dict[str, str]]
    errors: list[str]
    final_answer: str
    tool_log: list[dict]
    needs_tool: bool


_FORMAT_PATTERN = re.compile(
    r"\b(pdf|docx|word|txt|text|markdown|md|pptx|powerpoint|xlsx|excel|csv|json)\b", re.I
)

# Phrases that indicate the user wants to SEE the content AND get the file
_WANTS_BOTH_PATTERN = re.compile(
    r"\b(also show|show me|print|display|and show|with content|with preview|preview)\b", re.I
)

# This is deliberately narrower than ``_FORMAT_PATTERN``.  Mentioning a PDF
# ("summarise this PDF") is not a request to create one.  These patterns are
# the deterministic safety net for explicit artifact requests.
_FILE_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pptx", r"\b(?:(?:generate|create|make|build|export|produce)\s+(?:a\s+)?|(?:provide|deliver|return).{0,40}\b(?:as|in)\s+)(?:powerpoint|pptx|ppt|slides?)\b"),
    ("xlsx", r"\b(?:(?:generate|create|make|build|export|produce)\s+(?:an?\s+)?|(?:provide|deliver|return).{0,40}\b(?:as|in)\s+)(?:excel|xlsx|spreadsheet)\b"),
    ("docx", r"\b(?:(?:generate|create|make|build|export|produce|write)\s+(?:a\s+)?|(?:provide|deliver|return).{0,40}\b(?:as|in)\s+)(?:word(?:\s+doc(?:ument)?)?|docx)\b"),
    ("pdf", r"\b(?:(?:generate|create|make|build|export|produce|save)\s+(?:a\s+)?|(?:provide|deliver|return).{0,40}\b(?:as|in)\s+)pdf\b"),
    ("image", r"\b(?:generate|create|make|draw)\s+(?:an?\s+)?image\b"),
)


def detect_file_intent(user_query: str) -> dict[str, str | None]:
    """Return an explicit requested output type, without asking an LLM.

    The result is intentionally a small, stable contract so both the terminal
    and API agent use the same decision.  ``file_format`` is ``None`` when the
    prompt only refers to a file rather than requesting one.
    """
    query = user_query if isinstance(user_query, str) else ""
    for file_format, pattern in _FILE_INTENT_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return {
                "file_format": file_format,
                "tool_name": "generate_image" if file_format == "image" else "create_file",
                "matched_text": match.group(0),
            }
    return {"file_format": None, "tool_name": None, "matched_text": None}


def _non_file_question(question: str, intent: dict[str, str | None]) -> str:
    """Remove only the explicit artifact clause before normal answer generation."""
    matched = intent.get("matched_text") or ""
    if not matched:
        return question
    cleaned = re.sub(re.escape(matched), "", question, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:also|and)\s*(?:it|this)?\s*(?:as|into|to)?\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;")
    return cleaned or "Please provide the requested answer."


def _file_topic(question: str) -> str:
    """Extract the subject, never the instruction to create a file.

    The previous expression removed everything after ``create a Word
    document``.  That left an empty subject and triggered the Python fallback,
    which is why a laptop request could become a Python document.
    """
    topic = (question or "").strip()
    topic = re.sub(
        r"^\s*(?:please\s+)?(?:generate|create|make|build|export|produce|write)\s+"
        r"(?:an?\s+)?(?:pdf|word(?:\s+doc(?:ument)?)?|docx|excel|xlsx|spreadsheet|"
        r"powerpoint|pptx|ppt|slides?)\s*(?:file|document|report|presentation)?\s*"
        r"(?:about|on|for|covering)?\s*",
        "", topic, flags=re.I,
    )
    topic = re.sub(r"^(?:the\s+)?topic\s+(?:of\s+|on\s+)?", "", topic, flags=re.I)
    # Handles: "Give me factorial code, also generate a PDF".
    topic = re.sub(
        r"(?:,?\s+(?:and\s+)?(?:also\s+)?)?(?:generate|create|make|build|export|produce|save)"
        r"(?:\s+it)?(?:\s+as|\s+in|\s+to)?(?:\s+an?|\s+the)?\s+"
        r"(?:pdf|word(?:\s+doc(?:ument)?)?|docx|excel|xlsx|spreadsheet|powerpoint|pptx|ppt|slides?)\b.*$",
        "", topic, flags=re.I,
    )
    return topic.strip(" ,.;") or "the requested topic"


def _event(state: WorkflowState, stage: str, detail: str) -> dict:
    return {"events": [*state.get("events", []), {"stage": stage, "detail": detail}]}


def _detect_format(question: str) -> str | None:
    match = _FORMAT_PATTERN.search(question)
    if not match:
        return None
    return {
        "word": "docx", "text": "txt", "markdown": "md",
        "powerpoint": "pptx", "excel": "xlsx",
    }.get(match.group(1).lower(), match.group(1).lower())


# Map schemas by function name for pre-validation
TOOL_SCHEMAS_MAP: dict[str, dict] = {
    s["function"]["name"]: s["function"] for s in tools.TOOL_SCHEMAS
}


def _log_tool_call(tool_name: str, args: dict, result: str) -> None:
    """Log tool call through existing main.py / router logging system so it is visible in terminal."""
    try:
        import router
        router._log_event("tool_call", {"tool": tool_name, "args": args, "result": result})
    except Exception:
        pass


def validate_tool_arguments(tool_name: str, args: dict) -> tuple[bool, str, dict]:
    """
    Validate that all required arguments for a tool are present and are the
    correct type BEFORE calling the function, preventing raw TypeErrors.
    """
    schema = TOOL_SCHEMAS_MAP.get(tool_name)
    if not schema:
        return True, "", args

    params_meta = schema.get("parameters", {})
    required = params_meta.get("required", [])
    props = params_meta.get("properties", {})
    cleaned_args = dict(args)

    for req_field in required:
        if req_field not in cleaned_args or cleaned_args[req_field] is None:
            return (
                False,
                f"Tool '{tool_name}' failed: Missing required argument '{req_field}'. Please provide {req_field}.",
                {},
            )

        val = cleaned_args[req_field]
        expected_type = props.get(req_field, {}).get("type")

        if expected_type == "string":
            if not isinstance(val, str) or not str(val).strip():
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a non-empty string.",
                    {},
                )
            cleaned_args[req_field] = str(val).strip()

        elif expected_type == "integer":
            try:
                cleaned_args[req_field] = int(val)
            except (ValueError, TypeError):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be an integer.",
                    {},
                )

        elif expected_type == "array":
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        cleaned_args[req_field] = parsed
                    else:
                        cleaned_args[req_field] = [v.strip() for v in val.split(",") if v.strip()]
                except Exception:
                    cleaned_args[req_field] = [v.strip() for v in val.split(",") if v.strip()]
            elif not isinstance(val, list):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a list.",
                    {},
                )

        elif expected_type == "boolean":
            if isinstance(val, str):
                cleaned_args[req_field] = val.lower() in ("true", "1", "yes")
            elif not isinstance(val, bool):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a boolean.",
                    {},
                )

    return True, "", cleaned_args


def execute_tool_safely(tool_name: str, raw_args: dict) -> tuple[bool, str, dict]:
    """
    Validate required arguments and execute individual tool call inside its own try/except.
    Returns (success: bool, message: str, cleaned_args: dict).
    """
    tool_fn = getattr(tools, tool_name, None)
    if not tool_fn:
        tool_fn = tools.TOOL_FUNCTIONS.get(tool_name)
    if not tool_fn:
        err = f"Tool '{tool_name}' failed: Tool is not registered in tools.py."
        _log_tool_call(tool_name, raw_args, err)
        return False, err, raw_args

    # Validate required arguments and types before invoking
    valid, val_err, clean_args = validate_tool_arguments(tool_name, raw_args)
    if not valid:
        _log_tool_call(tool_name, raw_args, val_err)
        return False, val_err, raw_args

    # Execute individual tool call safely
    try:
        res = tool_fn(**clean_args)
        res_str = str(res)
        _log_tool_call(tool_name, clean_args, res_str)
        return True, res_str, clean_args
    except Exception as exc:
        err_msg = f"Tool '{tool_name}' failed: {exc}"
        _log_tool_call(tool_name, clean_args, err_msg)
        return False, err_msg, clean_args


def _detect_tool_action(question: str) -> tuple[str | None, dict]:
    """Detect if question is requesting a specific non-file-creation utility tool."""
    q = question.strip()
    ql = q.lower()

    # 1. AI Image generation
    if re.search(r"\b(generate (an )?image|create (an )?image|make (an )?image|draw (an )?image|generate_image)\b", ql):
        prompt = re.sub(
            r"^(generate|create|make|draw)\s+(an\s+)?image\s+(of\s+|for\s+)?",
            "", q, flags=re.I,
        ).strip()
        if prompt:
            return "generate_image", {"prompt": prompt}

    # 2. List files in output dir
    if re.search(r"\b(list (my )?files|show (my )?files|what files|generated files)\b", ql):
        return "list_generated_files", {}

    # 3. Get PDF page count
    if re.search(r"\b(page count|how many pages|count pages|number of pages)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "get_pdf_page_count", {"pdf_path": path}

    # 4. Extract text from PDF
    if re.search(r"\b(extract text|read pdf|read text from pdf|dump text from)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "extract_pdf_text", {"pdf_path": path}

    # 5. Rotate PDF
    if re.search(r"\b(rotate pdf|rotate pages?|rotate)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        deg_m = re.search(r"\b(90|180|270)\b", q)
        degrees = int(deg_m.group(1)) if deg_m else 90
        if m:
            path = m.group(1) or m.group(2)
            return "rotate_pdf_pages", {"pdf_path": path, "degrees": degrees}

    # 6. Split PDF
    if re.search(r"\b(split pdf|split pages?|separate pages)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "split_pdf", {"pdf_path": path}

    # 7. Merge PDFs
    if re.search(r"\b(merge pdfs?|combine pdfs?|join pdfs?)\b", ql):
        paths = re.findall(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        found = [p[0] or p[1] for p in paths]
        if len(found) >= 2:
            return "merge_pdfs", {"pdf_paths": found}

    # 8. OCR image
    if re.search(r"\b(ocr|extract text from image|read (text from )?image)\b", ql):
        m = re.search(r'["\']([^"\']+\.(png|jpe?g|webp|bmp))["\']|([A-Za-z0-9_\-\\/:.]+\.(png|jpe?g|webp|bmp))', q, re.I)
        if m:
            path = m.group(1) or m.group(3)
            return "ocr_image", {"image_path": path}

    return None, {}


def supervisor_node(state: WorkflowState) -> dict:
    started = time.time()
    q = state["question"]
    direct_tool, tool_args = _detect_tool_action(q)
    file_intent = detect_file_intent(q)
    requested_format = file_intent["file_format"] if not direct_tool else None

    needs_rag = bool(re.search(
        r"\b(document|knowledge base|uploaded|source|my file|from the doc|in the pdf|search docs?)\b",
        q, re.I,
    )) and not direct_tool

    # User wants content preview in terminal AND the file
    show_preview = bool(_WANTS_BOTH_PATTERN.search(q))

    plan = {
        "direct_tool":     direct_tool,
        "tool_args":       tool_args,
        "file_format":     requested_format,
        "file_intent":     file_intent,
        "needs_rag":       needs_rag,
        "show_preview":    show_preview,
    }

    action_label = f"Tool: {direct_tool}" if direct_tool else f"Format: {requested_format or 'direct'}"
    result = {
        **_event(state, "understanding",
                 f"{action_label} | RAG: {needs_rag} | Preview: {show_preview}"),
        "plan": plan,
    }
    result["events"] = [*result["events"], {"stage": "timing", "detail": f"classification: {round(time.time() - started, 3)}s"}]
    return result


def planner_node(state: WorkflowState) -> dict:
    plan = state["plan"]
    if plan.get("direct_tool"):
        detail = f"Executing specialized tool: {plan['direct_tool']}."
    elif plan.get("file_format"):
        fmt = plan["file_format"]
        detail = f"Generating content → building {fmt.upper()} → validating."
    else:
        detail = "Generating direct response."
    return _event(state, "planning", detail)


def rag_node(state: WorkflowState) -> dict:
    if not state["plan"].get("needs_rag"):
        return {"context": ""}
    try:
        from rag.pipeline import answer
        result = answer(state["question"])
        if result.get("sources"):
            return {
                **_event(state, "retrieving", f"Retrieved {len(result['sources'])} source(s)."),
                "context": result.get("answer", ""),
            }
        return {
            **_event(state, "retrieving", "No relevant documents found."),
            "context": "",
        }
    except Exception as exc:
        return {
            **_event(state, "retrieving", f"KB unavailable: {exc}"),
            "context": "",
            "errors": [str(exc)],
        }


def _llm_content(question: str, context: str, file_format: str | None) -> str:
    """
    Generate the ACTUAL content for the document — not instructions about how
    to create the document, not the filename, and NEVER code to generate a PDF.
    """
    is_code_request = bool(re.search(
        r"\b(code|script|program|function|class|algorithm|python|sql|javascript|java|c\+\+|html|css|implementation)\b",
        question, re.I
    ))

    subject = _file_topic(question) if file_format else question.strip().strip(".,;")

    subject = re.sub(r"^(give\s+me(\s+the|\s+a)?|write(\s+a|\s+the)?|generate(\s+a|\s+the)?|create(\s+a|\s+the)?)\s+", "", subject, flags=re.I).strip()

    if not subject or subject.lower() in ("code", "a code", "the code"):
        subject = "Production Python programming implementation, algorithm, and architectural guide"

    # Select optimal model: CODER_MODEL for code tasks, MAIN_MODEL for others
    model_to_use = config.CODER_MODEL if is_code_request else config.MAIN_MODEL

    prompt_parts = []

    if context:
        prompt_parts.append(
            f"Use this knowledge-base context to answer accurately:\n\n{context}\n\n---\n\n"
        )

    if is_code_request:
        prompt_parts.append(
            f"Provide a complete, production-grade technical guide and implementation for: {subject}\n\n"
            "Format your response using structured markdown:\n"
            "  # [System or Algorithm Title]\n"
            "  ## Architectural Overview\n"
            "  ## Implementation\n"
            "  ```python\n"
            "  # clean, executable, well-commented code here\n"
            "  ```\n"
            "  ## Complexity & Performance Analysis\n"
            "  ## Usage Examples & Test Cases\n\n"
            "CRITICAL RULES:\n"
            "- Write the ACTUAL requested programming code with clean docstrings and comments.\n"
            "- DO NOT write Python or ReportLab code that creates a PDF/Word/PowerPoint file.\n"
            "- The document engine will compile your output into the target file automatically.\n"
            "- Minimum 400 words with thorough explanations and executable code.\n\n"
        )
    elif file_format:
        if file_format == "xlsx":
            prompt_parts.append(
                f"Create structured tabular data for: {subject}\n\n"
                "Output ONLY a clean CSV table: one header row followed by 10-20 realistic, "
                "internally consistent data rows.\n"
                "Choose useful business/domain columns, use concrete values (not placeholders), "
                "and make each row complete. Use commas as delimiters; quote any value containing a comma.\n"
                "No markdown fences, title, commentary, blank rows, or explanation—only CSV data.\n"
                "Example:\n"
                "Name,Value,Category\n"
                "Row1,100,Type A\n"
                "Row2,200,Type B\n\n"
            )
        elif file_format == "pptx":
            prompt_parts.append(
                f"Create a professional presentation outline on: {subject}\n\n"
                "Structure your response as clearly separated sections, one per slide.\n"
                "Format:\n"
                "# Slide Title\n"
                "- Key point 1\n"
                "- Key point 2\n"
                "- Key point 3\n\n"
                "Create exactly 7 slides. Each slide must have 3-5 concise, specific bullets. "
                "Start with an executive overview and end with practical conclusions. "
                "Do not use placeholders, meta-commentary, or ask questions.\n\n"
            )
        else:  # pdf, docx, txt, md
            prompt_parts.append(
                f"Write a comprehensive, professional document on: {subject}\n\n"
                "Use this exact markdown formatting:\n"
                "  # Main Title (one only)\n"
                "  ## Section Heading\n"
                "  ### Sub-section\n"
                "  - Bullet point\n"
                "  1. Numbered step\n"
                "  > Note: callout text\n"
                "  ```python\n  code here\n  ```\n"
                "  | Col1 | Col2 | Col3 |\n  |------|------|------|\n  | Data | Data | Data |\n\n"
                "CRITICAL RULES:\n"
                f"- The document is ONLY about '{subject}'. Do not introduce unrelated subjects, programming, or Python unless the topic explicitly asks for them.\n"
                "- Write ONLY the document content. Do NOT include meta-text like "
                "'Here is the document', 'I have created', or any preamble.\n"
                "- DO NOT write code that builds or saves PDF/DOCX files.\n"
                "- Do not ask the reader for more details; make sensible professional assumptions where needed.\n"
                "- Use a clear introduction, logically ordered sections, a practical example or comparison where useful, and a concise conclusion.\n"
                "- Make it thorough, factual, and professional — 600-900 words.\n\n"
            )
    else:
        prompt_parts.append(f"Answer accurately and completely:\n\n{subject}\n\n")

    full_prompt = "".join(prompt_parts)

    try:
        llm = ChatOllama(
            model=model_to_use,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.15,
            num_predict=4096,
        )
        response = llm.invoke(full_prompt)
        result = str(response.content).strip()
        if not result:
            raise ValueError("LLM returned empty content")
        return result
    except Exception as exc:
        return (
            f"# {subject}\n\n"
            f"*Content generation encountered an error: {exc}*\n\n"
            f"Generated by Agent OTG | DWE Team"
        )


def execution_node(state: WorkflowState) -> dict:
    started = time.time()
    plan = state["plan"]
    direct_tool = plan.get("direct_tool")
    tool_args   = plan.get("tool_args", {})

    # ── Specialized utility tool execution ─────────────────────────────────
    if direct_tool:
        # This error handling exists specifically so one broken tool call never crashes an entire multi-step agent request.
        success, res_str, clean_args = execute_tool_safely(direct_tool, tool_args)
        tool_entry = {
            "tool":   direct_tool,
            "args":   clean_args,
            "result": res_str,
        }
        status_icon = "✅" if success else "❌"
        return {
            **_event(state, "executing_tool", f"{status_icon} Invoked {direct_tool}"),
            "final_answer": res_str,
            "tool_log": [tool_entry],
            "needs_tool": True,
            "errors": [] if success else [res_str],
        }

    # ── Explicit file intent: answer first, artifact second ────────────────
    # The model no longer decides whether a requested document is created.
    # It only answers the non-file part; file_node invokes create_file itself.
    fmt = plan.get("file_format")
    if fmt:
        intent = plan.get("file_intent", {})
        non_file_question = _non_file_question(state["question"], intent)

        if fmt == "image":
            prompt = re.sub(r"^.*?\bimage\b(?:\s+of|\s+for)?\s*", "", state["question"], flags=re.I).strip()
            success, message, clean_args = execute_tool_safely("generate_image", {"prompt": prompt or state["question"]})
            elapsed = round(time.time() - started, 3)
            return {
                **_event(state, "executing_tool", f"{'✅' if success else '❌'} Deterministic image generation ({elapsed}s)"),
                "events": [*state.get("events", []), {"stage": "executing_tool", "detail": f"{'✅' if success else '❌'} Deterministic image generation"}, {"stage": "timing", "detail": f"tool execution: {elapsed}s"}],
                "final_answer": message,
                "tool_log": [{"tool": "generate_image", "args": clean_args, "result": message}],
                "needs_tool": True,
                "errors": [] if success else [message],
            }

        is_code = bool(re.search(r"\b(code|script|algorithm|python)\b", non_file_question, re.I))
        model_used = config.CODER_MODEL if is_code else config.MAIN_MODEL
        # File formats need format-aware source material.  In particular, an
        # XLSX needs CSV rows and a deck needs slide headings; a generic chat
        # answer is prose and produces poor spreadsheets / one-slide decks.
        file_content = _llm_content(state["question"], state.get("context", ""), fmt)
        # Documents are the deliverable.  Only retain terminal text when the
        # question explicitly requested code; this preserves code+PDF while
        # avoiding a duplicate Word/PDF/spreadsheet body in the terminal.
        answer_text = file_content if is_code else ""
        elapsed = round(time.time() - started, 3)
        return {
            **_event(state, "generating", f"Answer generation: {elapsed}s with {model_used}."),
            "events": [*state.get("events", []), {"stage": "generating", "detail": f"Answer generation with {model_used}"}, {"stage": "timing", "detail": f"final answer generation: {elapsed}s"}],
            "answer_text": answer_text,
            "content": file_content,
        }

    # ── Existing non-file LangGraph / RAG behavior ─────────────────────────
    content = _llm_content(state["question"], state.get("context", ""), fmt)
    is_code = bool(re.search(r"\b(code|script|algorithm|python)\b", state["question"], re.I))
    model_used = config.CODER_MODEL if is_code else config.MAIN_MODEL

    elapsed = round(time.time() - started, 3)
    return {
        **_event(state, "generating", f"Final answer generation: {elapsed}s with {model_used}."),
        "events": [*state.get("events", []), {"stage": "generating", "detail": f"Final answer generation with {model_used}"}, {"stage": "timing", "detail": f"final answer generation: {elapsed}s"}],
        "content": content,
    }


def file_node(state: WorkflowState) -> dict:
    started = time.time()
    fmt = state["plan"].get("file_format")
    if not fmt:
        return {}
    title, clean_content, safe_slug = derive_clean_title(state["question"], state.get("content", ""))
    filename = f"{safe_slug}.{fmt}"
    try:
        # Direct Python tool call: no model tool_call decision exists on this
        # path.  ``create_file`` is the single styled artifact implementation.
        artifact = tools.create_file(title, clean_content, fmt, filename)
        if not isinstance(artifact, dict) or artifact.get("error"):
            raise RuntimeError(artifact.get("error", "create_file returned an invalid result"))
        tool_entry = {
            "tool":   "create_file",
            "args":   {"title": title, "format": fmt, "filename": filename},
            "result": f"Created {artifact['filename']} ({artifact['size_bytes']:,} bytes)",
        }
        _log_tool_call("create_file", tool_entry["args"], tool_entry["result"])
        return {
            **_event(state, "creating_file",
                     f"✅ {fmt.upper()}: {artifact['filename']} ({artifact['size_bytes']:,} bytes) | tool execution: {round(time.time() - started, 3)}s"),
            "events": [*state.get("events", []), {"stage": "creating_file", "detail": f"✅ {fmt.upper()}: {artifact['filename']} ({artifact['size_bytes']:,} bytes)"}, {"stage": "timing", "detail": f"tool execution: {round(time.time() - started, 3)}s"}],
            "artifact":  artifact,
            "content":   clean_content,
            "tool_log":  [tool_entry],
            "needs_tool": True,
        }
    except Exception as exc:
        err_msg = f"Tool 'create_file' failed: {exc}"
        _log_tool_call("create_file", {"title": title, "format": fmt, "filename": filename}, err_msg)
        return {
            **_event(state, "creating_file", f"❌ {err_msg}"),
            "errors": [err_msg],
        }


def validation_node(state: WorkflowState) -> dict:
    artifact = state.get("artifact")
    if artifact:
        if artifact.get("size_bytes", 0) <= 0:
            return {
                **_event(state, "validating", "❌ Artifact is empty."),
                "errors": ["Generated artifact is empty."],
            }
        return _event(state, "validating", f"✅ {artifact['filename']} ({artifact['size_bytes']:,} bytes)")
    return _event(state, "validating", "✅ Direct response validated.")


def response_node(state: WorkflowState) -> dict:
    if state.get("final_answer") and not state.get("artifact"):
        return _event(state, "completed", "Workflow completed.")

    artifact  = state.get("artifact")
    content   = state.get("content", "").strip()
    errors    = state.get("errors", [])
    show_prev = state.get("plan", {}).get("show_preview", False)

    if errors and not artifact:
        answer = (
            f"❌ Agent error: {errors[-1]}\n\n"
            "Please try rephrasing your request."
        )

    elif artifact:
        answer_text = state.get("answer_text", "").strip()
        lines = [
            f"✅ **{artifact['filename']}** created successfully!\n",
            f"📁 **Saved to:** `{artifact['path']}`",
            f"📦 **Size:** {artifact['size_bytes']:,} bytes",
            f"📂 **Folder:** `{OUTPUT_DIR}`",
        ]

        if show_prev and content:
            preview = content[:1500] + ("\n\n…[truncated]" if len(content) > 1500 else "")
            lines = [f"### Preview\n\n{preview}\n\n---\n"] + lines

        # Keep the conversational answer and artifact metadata distinct.  The
        # document body is never injected as a preview unless the user asked.
        answer = (answer_text + "\n\n---\n\n" if answer_text else "") + "\n".join(lines)

    else:
        answer = content or "I could not generate a response. Please try again."

    return {
        **_event(state, "completed", "Workflow done."),
        "final_answer": answer,
    }


def _route_after_execution(state: WorkflowState) -> str:
    if state.get("final_answer"):
        return "validate"
    return "file" if state["plan"].get("file_format") else "validate"


def _build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner",    planner_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("execution",  execution_node)
    graph.add_node("file",       file_node)
    graph.add_node("validation", validation_node)
    graph.add_node("response",   response_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "planner")
    graph.add_edge("planner",    "rag")
    graph.add_edge("rag",        "execution")
    graph.add_conditional_edges(
        "execution", _route_after_execution,
        {"file": "file", "validate": "validation"},
    )
    graph.add_edge("file",       "validation")
    graph.add_edge("validation", "response")
    graph.add_edge("response",   END)
    return graph.compile()


_GRAPH = _build_graph()


def run_agent(query: str, history: list | None = None) -> dict:
    """Run the supervised workflow. Returns answer, artifact info, and event log."""
    result = _GRAPH.invoke({
        "question": query,
        "history":  history or [],
        "events":   [],
        "errors":   [],
        "tool_log": [],
    })
    artifact = result.get("artifact")
    tool_log = result.get("tool_log", [])

    final_answer = result.get("final_answer", "")
    # _llm_content talks directly to ChatOllama, unlike router.get_full_answer.
    # Persist the completed exchange here so agent/file sessions are not empty.
    try:
        import router
        router.add_to_history("user", query)
        router.add_to_history("assistant", final_answer)
    except Exception:
        # The workflow result remains usable even if a local history store is
        # temporarily unavailable; db.py records the actual error state.
        pass

    return {
        "final_answer": final_answer,
        "needs_tool":   bool(tool_log) or bool(artifact),
        "tool_log":     tool_log,
        "artifact":     artifact,
        "events":       result.get("events", []),
        "errors":       result.get("errors", []),
        "run_id":       uuid.uuid4().hex,
    }
