"""
router.py - Smart multi-model router for Agent OTG

MODELS
    CODER_MODEL : code questions — qwen2.5-coder
    MAIN_MODEL  : complex reasoning, essays, analysis — qwen2.5:14b
    FAST_MODEL  : quick answers, greetings, simple math — qwen2.5:7b
    IMAGE_MODEL : vision analysis — qwen2.5vl:7b

CLASSIFICATION — 5 categories (upgraded from 3):
    "code"       → CODER_MODEL
    "simple"     → FAST_MODEL  ← was under-used before
    "complex"    → MAIN_MODEL
    "rag_search" → FAST_MODEL (RAG handles the heavy lifting)
    "agent_task" → handled by LangGraph agent pipeline

SMART ROUTING
    The classifier now uses a richer prompt, length-based heuristics,
    and keyword safety-nets to avoid sending simple questions to the
    expensive 14b model.

SESSION PERSISTENCE
    All messages are stored in SQLite (agent_otg.sqlite3) via db.py.
"""

import json
import os
import re
import ollama

import db as _db

_CURRENT_SESSION_NAME: str = ""
_LOG_CALLBACK = None


def _find_embedded_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Detect tool calls that models emitted as plain JSON text instead of native tool calls."""
    if not text:
        return []
    results = []
    pattern = re.compile(r'```(?:json)?\s*(\{\s*"name"\s*:\s*"[^"]+".*?\})\s*```', re.DOTALL)
    for block in pattern.findall(text):
        try:
            data = json.loads(block)
            name = data.get("name")
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name and isinstance(args, dict):
                results.append((name, args))
        except Exception:
            continue
    if not results:
        unfenced = re.findall(
            r'(\{\s*"name"\s*:\s*"(?:generate_pdf_from_text|create_file|write_docx|merge_pdfs|split_pdf|rotate_pdf_pages|ingest_file|search_documents)"\s*,\s*"arguments"\s*:\s*\{.*?\}(?:\s*,\s*"filename"\s*:\s*"[^"]*")?\s*\})',
            text,
            re.DOTALL,
        )
        for block in unfenced:
            try:
                data = json.loads(block)
                name = data.get("name")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name and isinstance(args, dict):
                    results.append((name, args))
            except Exception:
                continue
    return results


def start_new_session(session_name: str):
    """
    Called at startup to name the current session.
    Creates the session record in the database.
    """
    global _CURRENT_SESSION_NAME
    _CURRENT_SESSION_NAME = session_name
    if _db.is_ready():
        _db.create_session(session_name)


def set_log_callback(cb):
    global _LOG_CALLBACK
    _LOG_CALLBACK = cb


def _log_event(event_type: str, data: dict):
    if _LOG_CALLBACK:
        try:
            _LOG_CALLBACK(event_type, data)
        except Exception:
            pass


def load_system_prompt() -> str:
    """Load the shared system prompt from disk (relative to this file)."""
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are a helpful, honest, on-premise AI assistant for industrial/confidential work."


# ─── Models (read from .env via config.py — swap without touching code) ──────
import config as _cfg
CODER_MODEL = _cfg.CODER_MODEL
MAIN_MODEL  = _cfg.MAIN_MODEL
FAST_MODEL  = _cfg.FAST_MODEL
IMAGE_MODEL = _cfg.IMAGE_MODEL

AVAILABLE_MODELS = {
    "code":       CODER_MODEL,
    "simple":     FAST_MODEL,
    "complex":    MAIN_MODEL,
    "rag_search": FAST_MODEL,   # RAG does the heavy work; model just formats
    "agent_task": MAIN_MODEL,   # Agent tasks use main model for planning
    "image":      IMAGE_MODEL,
}

# ─── Conversation memory ─────────────────────────────────────────────────────
HISTORY   = []
MAX_TURNS = 12


def add_to_history(role: str, content: str):
    HISTORY.append({"role": role, "content": content})
    if len(HISTORY) > MAX_TURNS:
        del HISTORY[:len(HISTORY) - MAX_TURNS]
    if _CURRENT_SESSION_NAME and _db.is_ready():
        ok = _db.save_message(_CURRENT_SESSION_NAME, role, content)
        if not ok:
            print(f"[router] DB save failed for role={role}", flush=True)


def get_history() -> list:
    return HISTORY


def clear_history():
    HISTORY.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SMART CLASSIFIER — 5 categories with confidence-based routing
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are an intelligent routing assistant for Agent OTG.
Read the user's message and classify it into EXACTLY one of 5 categories.

CATEGORIES:
- "code"       → Writing, debugging, fixing, or testing code, scripts, SQL, functions, APIs
- "simple"     → Greetings, small talk, quick factual one-liners, easy math (< 8 words usually)
- "complex"    → Essays, analysis, explanations, research questions, multi-step reasoning
- "rag_search" → User wants to search, query, or ask about UPLOADED documents / knowledge base
                 Signals: "what does the document say", "from my files", "search the KB",
                 "what does the report say", "according to the uploaded", "find in docs",
                 "what does it say about", "from the data", "in the file"
- "agent_task" → User wants to CREATE or GENERATE a FILE (PDF, Word, Excel, CSV, PowerPoint)
                 or needs multi-tool autonomous work
                 Signals: "create a PDF", "generate a Word doc", "make a spreadsheet",
                 "write a report and save it", "generate a file", "create an Excel sheet"

RULES:
1. "simple" ONLY for: greetings (hi/hello/thanks), trivial math, very short factual questions
2. "rag_search" when user references existing documents/files they've uploaded
3. "agent_task" when user explicitly wants a FILE to be created/saved/exported
4. "code" takes priority over "complex" for any programming question
5. "is_multi_part" = true ONLY if message has 2+ clearly distinct, unrelated requests
6. "has_code_and_explain" = true ONLY when BOTH writing code AND explaining it is requested

Reply with ONLY this JSON — nothing else:
{"category": "code"|"simple"|"complex"|"rag_search"|"agent_task", "is_multi_part": true|false, "has_code_and_explain": true|false, "reason": "one short sentence", "confidence": 0.0-1.0}
"""

# ── Keyword safety-nets ───────────────────────────────────────────────────────
_EXPLAIN_KEYWORDS = [
    "explain", "walk me through", "walk through", "how it works",
    "step by step", "step-by-step", "break it down", "describe",
    "elaborate", "detail", "breakdown", "walkthrough",
]
_CODE_KEYWORDS = [
    "code", "script", "function", "program", "write", "build",
    "implement", "create", "generate", "debug", "fix", "bug",
    "test", "class", "method", "api", "sql", "query", "python",
    "javascript", "typescript", "java", "c++", "rust", "go",
    "react", "html", "css", "flask", "fastapi", "django",
]
_SIMPLE_KEYWORDS = [
    "hello", "hi ", "hey ", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "bye", "goodbye", "how are you",
    "what is your name", "who are you", "what time", "what day",
]
_FILE_CREATION_KEYWORDS = [
    "create a pdf", "generate a pdf", "make a pdf", "write a pdf",
    "create a word", "generate a word", "make a word", "write a word",
    "create a doc", "generate a doc",
    "create an excel", "make an excel", "generate an excel",
    "generate a spreadsheet", "create a spreadsheet", "make a spreadsheet",
    "create a csv", "make a csv", "generate a csv",
    "create a report", "generate a report", "save as pdf", "export to pdf",
    "create a presentation", "make a powerpoint", "generate a pptx",
    "create a json", "generate a json file", "generate a json",
    "write a report and save", "create and save", "make and save",
    "make a file", "generate a file", "export a file",
]
_RAG_KEYWORDS = [
    "what does the document", "what does my document", "what does the file",
    "from the uploaded", "in my files", "search the knowledge base",
    "from the knowledge base", "what does the report say", "search my docs",
    "what does the uploaded document", "what does the uploaded file",
    "what does it say about", "find in the document", "look up in",
    "from the uploaded file", "what the document says",
    "according to the document", "according to the report",
    "what does my report", "what does my file",
]


def _looks_like_code_and_explain(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _CODE_KEYWORDS) and any(k in q for k in _EXPLAIN_KEYWORDS)


def _quick_classify(query: str) -> str | None:
    """
    Fast keyword-based pre-screening for obvious cases.
    Returns a category string or None to fall through to LLM classification.
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # Simple: very short greetings / small talk
    if word_count <= 6 and any(k in q for k in _SIMPLE_KEYWORDS):
        return "simple"

    # Very short query with no technical terms → probably simple
    if word_count <= 4 and not any(k in q for k in _CODE_KEYWORDS + _FILE_CREATION_KEYWORDS):
        return "simple"

    # File creation → agent_task
    if any(k in q for k in _FILE_CREATION_KEYWORDS):
        return "agent_task"

    # RAG search → rag_search
    if any(k in q for k in _RAG_KEYWORDS):
        return "rag_search"

    return None  # Let the LLM decide


def classify_question(query: str) -> dict:
    """
    Classify a user query into one of 5 routing categories.
    Uses fast keyword pre-screening first, then LLM classification,
    with keyword safety-nets as final fallback.
    """
    recent = HISTORY[-4:]
    context_text = "\n".join(f"{m['role']}: {m['content'][:120]}" for m in recent)

    # ── Phase 1: fast keyword pre-screen ──────────────────────────────────────
    quick = _quick_classify(query)
    if quick:
        result_data = {
            "category":             quick,
            "model":                pick_model(quick),
            "is_multi_part":        False,
            "has_code_and_explain": _looks_like_code_and_explain(query) if quick == "code" else False,
            "reason":               f"Fast keyword routing → {quick}",
            "confidence":           0.95,
            "routing_method":       "keyword",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Phase 2: LLM classification ───────────────────────────────────────────
    category         = "complex"
    is_multi_part    = False
    has_code_explain = False
    reason           = ""
    confidence       = 0.5

    try:
        response = ollama.chat(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user",   "content": f"Recent chat:\n{context_text}\n\nNew message: {query}"},
            ],
            format="json",
            options={"temperature": 0},
        )
        raw = response["message"]["content"] if isinstance(response, dict) else response.message.content
        parsed        = json.loads(raw)
        category      = parsed.get("category", "complex")
        is_multi_part = bool(parsed.get("is_multi_part", False))
        has_code_explain = bool(parsed.get("has_code_and_explain", False))
        reason        = parsed.get("reason", "")
        confidence    = float(parsed.get("confidence", 0.7))

        # Safety-net: keyword override for obvious code+explain
        if not has_code_explain and _looks_like_code_and_explain(query):
            has_code_explain = True

    except Exception:
        # ── Phase 3: keyword fallback ──────────────────────────────────────
        q_lower = query.lower()
        if any(k in q_lower for k in _FILE_CREATION_KEYWORDS):
            category = "agent_task"
        elif any(k in q_lower for k in _RAG_KEYWORDS):
            category = "rag_search"
        elif any(k in q_lower for k in ["code", "python", "function", "bug", "error", "script", "sql", "debug", "fix", "class", "api"]):
            category = "code"
        elif len(query.split()) <= 8:
            category = "simple"
        else:
            category = "complex"
        has_code_explain = _looks_like_code_and_explain(query)
        is_multi_part    = has_code_explain or (" and " in q_lower and len(query) > 40)
        reason           = "Keyword fallback (LLM classifier unavailable)"
        confidence       = 0.6

    # Validate category
    if category not in AVAILABLE_MODELS:
        category = "complex"

    result_data = {
        "category":             category,
        "model":                pick_model(category),
        "is_multi_part":        is_multi_part,
        "has_code_and_explain": has_code_explain,
        "reason":               reason,
        "confidence":           confidence,
        "routing_method":       "llm",
    }
    _log_event("classify", {"query": query, "result": result_data})
    return result_data


def pick_model(category: str) -> str:
    return AVAILABLE_MODELS.get(category, MAIN_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers to safely read fields from ollama response (dict or object)
# ══════════════════════════════════════════════════════════════════════════════

def _get(obj, *keys, default=None):
    """Safely get nested attribute/key from ollama response objects or dicts."""
    for key in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj if obj is not None else default


def _msg_to_dict(msg) -> dict:
    """Normalise an ollama message (could be dict or object) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    d = {
        "role":    getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", "") or "",
    }
    tc = getattr(msg, "tool_calls", None)
    if tc:
        d["tool_calls"] = tc
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Run a model on a single task (with memory)
# ══════════════════════════════════════════════════════════════════════════════

def _build_messages(query: str) -> list:
    sys_prompt = load_system_prompt()
    return [{"role": "system", "content": sys_prompt}] + HISTORY + [{"role": "user", "content": query}]


def get_full_answer(model: str, query: str) -> str:
    from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
    _log_event("model_start", {"model": model, "query": query})
    messages = _build_messages(query)

    try:
        response = ollama.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
    except Exception as exc:
        _log_event("model_done", {"model": model, "answer": f"[Error] {exc}"})
        err_msg = f"Error calling model '{model}': {exc}"
        add_to_history("user", query)
        add_to_history("assistant", err_msg)
        return err_msg

    tool_calls = _get(response, "message", "tool_calls") or []

    if tool_calls:
        messages.append(_msg_to_dict(_get(response, "message")))

        for tool_call in tool_calls:
            func_name = _get(tool_call, "function", "name")
            args      = _get(tool_call, "function", "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
            else:
                result = f"Error: tool '{func_name}' not registered."

            _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "content": str(result)})

        try:
            response = ollama.chat(model=model, messages=messages)
        except Exception as exc:
            answer = f"Tools executed. Error generating follow-up: {exc}"
            add_to_history("user", query)
            add_to_history("assistant", answer)
            return answer

    answer = _get(response, "message", "content") or ""

    # Fallback: embedded JSON tool calls in text
    if not tool_calls:
        embedded = _find_embedded_tool_calls(answer)
        for func_name, args in embedded:
            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
                _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
                answer += f"\n\n---\n🛠️ **Auto-executed tool** `{func_name}`:\n📄 {result}\n---"

    add_to_history("user",      query)
    add_to_history("assistant", answer)
    _log_event("model_done", {"model": model, "answer": answer})
    return answer


def stream_answer(model: str, query: str):
    from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
    _log_event("stream_start", {"model": model, "query": query})
    messages    = _build_messages(query)
    full_answer = ""

    response_msg: dict = {"role": "assistant", "content": ""}
    tool_calls:   list = []

    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True, tools=TOOL_SCHEMAS):
            chunk_tcs = _get(chunk, "message", "tool_calls") or []
            if chunk_tcs:
                tool_calls.extend(chunk_tcs)

            token = _get(chunk, "message", "content") or ""
            if token:
                full_answer += token
                response_msg["content"] += token
                yield token
    except Exception as exc:
        err = f"\n[Error during streaming: {exc}]"
        yield err
        full_answer += err
        add_to_history("user", query)
        add_to_history("assistant", full_answer)
        _log_event("stream_done", {"model": model, "answer": full_answer})
        return

    if tool_calls:
        response_msg["tool_calls"] = tool_calls
        messages.append(response_msg)

        for tool_call in tool_calls:
            func_name = _get(tool_call, "function", "name")
            args      = _get(tool_call, "function", "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            yield f"\n\n[System: Calling tool '{func_name}'...]\n"

            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
            else:
                result = f"Error: tool '{func_name}' not registered."

            _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "content": str(result)})

        yield "\n"
        try:
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                token = _get(chunk, "message", "content") or ""
                if token:
                    full_answer += token
                    yield token
        except Exception as exc:
            err = f"\n[Error in follow-up stream: {exc}]"
            yield err
            full_answer += err

    elif not tool_calls:
        embedded = _find_embedded_tool_calls(full_answer)
        for func_name, args in embedded:
            if func_name in TOOL_FUNCTIONS:
                yield f"\n\n[System: Auto-executing tool '{func_name}'...]\n"
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
                _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
                yield f"\n---\n🛠️ **Auto-executed tool** `{func_name}`:\n📄 {result}\n---\n"
                full_answer += f"\n\n[Auto-executed '{func_name}': {result}]"

    add_to_history("user",      query)
    add_to_history("assistant", full_answer)
    _log_event("stream_done", {"model": model, "answer": full_answer})


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE: vision model
# ══════════════════════════════════════════════════════════════════════════════

def ask_image(image_b64: str, query: str) -> str:
    _log_event("image_start", {"model": IMAGE_MODEL, "query": query, "image_size_b64": len(image_b64)})
    try:
        response = ollama.chat(
            model=IMAGE_MODEL,
            messages=[{
                "role":    "user",
                "content": query,
                "images":  [image_b64],
            }],
        )
        answer = _get(response, "message", "content") or ""
    except Exception as exc:
        answer = f"Error analyzing image: {exc}"

    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", answer)
    _log_event("image_done", {"model": IMAGE_MODEL, "answer": answer})
    return answer


def stream_image_answer(image_b64: str, query: str):
    _log_event("image_start", {"model": IMAGE_MODEL, "query": query, "image_size_b64": len(image_b64)})
    full_answer = ""
    try:
        for chunk in ollama.chat(
            model=IMAGE_MODEL,
            messages=[{
                "role":    "user",
                "content": query,
                "images":  [image_b64],
            }],
            stream=True,
        ):
            token = _get(chunk, "message", "content") or ""
            if token:
                full_answer += token
                yield token
    except Exception as exc:
        err = f"[Error: {exc}]"
        full_answer += err
        yield err

    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", full_answer)
    _log_event("image_done", {"model": IMAGE_MODEL, "answer": full_answer})


# ══════════════════════════════════════════════════════════════════════════════
# Sequential pipeline: "code + explain" in two ordered steps
# ══════════════════════════════════════════════════════════════════════════════

def run_sequential_tasks(query: str) -> list:
    results = []

    code_prompt = (
        f'The user asked: "{query}"\n\n'
        "Your job for this step: write ONLY the code. "
        "Do not explain it yet — just provide clean, well-commented code."
    )
    code_answer = get_full_answer(CODER_MODEL, code_prompt)
    results.append({
        "step":       1,
        "label":      "Code Generation",
        "model_used": CODER_MODEL,
        "category":   "code",
        "answer":     code_answer,
    })

    explain_prompt = (
        "Now explain the code you just wrote above, step by step. "
        "Be clear and beginner-friendly. Cover what each part does and why."
    )
    explain_answer = get_full_answer(MAIN_MODEL, explain_prompt)
    results.append({
        "step":       2,
        "label":      "Step-by-Step Explanation",
        "model_used": MAIN_MODEL,
        "category":   "complex",
        "answer":     explain_answer,
    })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Split a multi-part question into independent sub-tasks
# ══════════════════════════════════════════════════════════════════════════════

def break_into_tasks(query: str) -> list:
    prompt = f"""The user's message contains MULTIPLE distinct requests bundled together.
Split it into separate, self-contained tasks — do NOT merge them back into one.
Each task's "task" field must be a full standalone instruction (repeat any shared
context so each task makes sense on its own).

Reply with ONLY a JSON list, nothing else, like:
[{{"label": "short title", "task": "the exact self-contained sub-question", "category": "code" | "simple" | "complex"}}]

There must be at least 2 items in the list if the message really contains multiple asks.

Message: {query}
"""
    try:
        response = ollama.chat(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        raw_content = response["message"]["content"] if isinstance(response, dict) else response.message.content
        raw = json.loads(raw_content)

        raw_tasks = raw.get("tasks", raw) if isinstance(raw, dict) else raw

        if not isinstance(raw_tasks, list):
            raise ValueError("Response is not a list")

        tasks = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            label    = str(t.get("label", "Sub-task")).strip() or "Sub-task"
            task_str = str(t.get("task", query)).strip() or query
            category = str(t.get("category", "complex")).strip()
            if category not in AVAILABLE_MODELS:
                category = "complex"
            tasks.append({
                "label":    label,
                "task":     task_str,
                "category": category,
                "model":    pick_model(category),
            })

        if len(tasks) >= 2:
            return tasks

    except Exception:
        pass

    # Small local models occasionally return invalid JSON for an obviously
    # multi-part request.  Do not silently collapse it into one expensive task.
    # 1. Numbered items (e.g. 1) ... 2) ... or 1. ... 2. ...)
    if re.search(r"(?:^|\s)(?:1[\).]|first[,:])\s+.*?(?:\s+(?:2[\).]|second[,:]))\s+", query, re.I | re.DOTALL):
        numbered_parts = [p.strip(" ,.;") for p in re.split(r"(?:^|\s+)\d+[\).]\s+", query) if p.strip(" ,.;")]
        if len(numbered_parts) >= 2:
            tasks = []
            for part in numbered_parts:
                category = _quick_classify(part) or "complex"
                tasks.append({
                    "label": part[:60],
                    "task": part,
                    "category": category,
                    "model": pick_model(category),
                })
            return tasks

    # 2. Bulleted items (e.g. - item 1 \n - item 2)
    if re.search(r"(?:^|\n)\s*[-*•]\s+.*?\n\s*[-*•]\s+", query):
        bullet_parts = [p.strip(" ,.;") for p in re.split(r"(?:^|\n)\s*[-*•]\s+", query) if p.strip(" ,.;")]
        if len(bullet_parts) >= 2:
            tasks = []
            for part in bullet_parts:
                category = _quick_classify(part) or "complex"
                tasks.append({
                    "label": part[:60],
                    "task": part,
                    "category": category,
                    "model": pick_model(category),
                })
            return tasks

    # 3. Delimited or conjoined action splitting
    parts = [p.strip(" ,.;") for p in re.split(
        r"(?:\n+|;|,\s*(?=(?:also\s+)?(?:write|create|generate|make|give|draft|explain)\b)|"
        r"\s+and\s+(?=(?:also\s+)?(?:write|create|generate|make|give|draft|explain)\b))",
        query, flags=re.I,
    ) if p.strip(" ,.;")]
    if len(parts) >= 2:
        tasks = []
        for part in parts:
            category = _quick_classify(part) or "complex"
            tasks.append({
                "label": part[:60],
                "task": part,
                "category": category,
                "model": pick_model(category),
            })
        return tasks

    info = classify_question(query)
    return [{"label": "Full request", "task": query, "category": info["category"], "model": info["model"]}]
