"""
router.py - Decides which model(s) answer, and remembers the conversation.

MODELS
    CODER_MODEL : code questions
    MAIN_MODEL  : the "smart" 14b model -> essays, reasoning, explanations
    FAST_MODEL  : the 7b model -> (a) makes the routing decision itself,
                  (b) answers quick casual chit-chat
    IMAGE_MODEL : vision model -> answers questions about images

HOW A QUESTION FLOWS
    1. classify_question() asks FAST_MODEL to decide category + whether
       the message is actually multiple separate asks bundled together.
       It also detects the special "code + explain" combo via
       has_code_and_explain flag + keyword safety-net.
    2. If has_code_and_explain is True, run_sequential_tasks() handles it:
         - Step 1: CODER_MODEL generates the code (saved to shared HISTORY).
         - Step 2: MAIN_MODEL reads that history and writes the explanation.
    3. Otherwise, if multi-part, break_into_tasks() splits it independently.
    4. get_full_answer() / stream_answer() actually run a model on a task.

MEMORY
    HISTORY is a shared list of past turns, used so follow-up questions
    have context. Fine for a single-user local demo.

SESSION PERSISTENCE
    All messages are stored in PostgreSQL (agent_otg database) via db.py.
    Configure credentials in db_config.json.
"""

import json
import os
import ollama

import db as _db

_CURRENT_SESSION_NAME: str = ""
_LOG_CALLBACK = None


def start_new_session(session_name: str):
    """
    Called at startup to name the current session.
    Creates the session record in PostgreSQL.
    The session_name is a human-readable string (e.g. 'session_2026-09-07_10-30-00').
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


# ─── Models (read from .env via config.py — swap without touching code) ──────────
import config as _cfg
CODER_MODEL = _cfg.CODER_MODEL
MAIN_MODEL  = _cfg.MAIN_MODEL
FAST_MODEL  = _cfg.FAST_MODEL
IMAGE_MODEL = _cfg.IMAGE_MODEL

AVAILABLE_MODELS = {
    "code":    CODER_MODEL,
    "simple":  FAST_MODEL,
    "complex": MAIN_MODEL,
    "image":   IMAGE_MODEL,
}

# ─── Conversation memory ────────────────────────────────────────────────────────
HISTORY   = []
MAX_TURNS = 12


def add_to_history(role: str, content: str):
    HISTORY.append({"role": role, "content": content})
    # Trim to keep only the last MAX_TURNS entries
    if len(HISTORY) > MAX_TURNS:
        del HISTORY[:len(HISTORY) - MAX_TURNS]

    # Persist to PostgreSQL
    if _CURRENT_SESSION_NAME and _db.is_ready():
        ok = _db.save_message(_CURRENT_SESSION_NAME, role, content)
        if not ok:
            print(f"[router] DB save failed for role={role} — message not persisted", flush=True)


def get_history() -> list:
    return HISTORY


def clear_history():
    HISTORY.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Classify — which category, and is it multiple asks?
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are a routing assistant. Read the user's new question
(and the short recent chat history, if any) and decide:

1. category: ONE of
   - "code"    -> writing, fixing, debugging, testing, or explaining code/scripts/SQL/functions
   - "simple"  -> quick factual questions, greetings, small talk, easy one-line math
   - "complex" -> anything needing real reasoning, explanation, essay writing, or multi-step thought
2. is_multi_part: true if the message asks for TWO OR MORE clearly distinct things
   (e.g. "write an essay on X AND give code for Y", "explain A and also list B").
   false if it's really just one request.
3. has_code_and_explain: true if the request combines BOTH a code-writing task AND
   an explanation/walkthrough of that same code. Common signals:
   - "give me code and explain how it works"
   - "write a script and walk me through it"
   - "create X and describe/detail/break down how it works"
   false otherwise.

Reply with ONLY this JSON, nothing else:
{"category": "code" | "simple" | "complex", "is_multi_part": true | false, "has_code_and_explain": true | false, "reason": "one short sentence"}
"""

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
]


def _looks_like_code_and_explain(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _CODE_KEYWORDS) and any(k in q for k in _EXPLAIN_KEYWORDS)


def classify_question(query: str) -> dict:
    recent       = HISTORY[-4:]
    context_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    category         = "complex"
    is_multi_part    = False
    has_code_explain = False
    reason           = ""

    try:
        response = ollama.chat(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user",   "content": f"Recent chat:\n{context_text}\n\nNew question: {query}"},
            ],
            format="json",
            options={"temperature": 0},
        )
        raw = response["message"]["content"] if isinstance(response, dict) else response.message.content
        result           = json.loads(raw)
        category         = result.get("category", "complex")
        is_multi_part    = bool(result.get("is_multi_part", False))
        has_code_explain = bool(result.get("has_code_and_explain", False))
        reason           = result.get("reason", "")

        if not has_code_explain and _looks_like_code_and_explain(query):
            has_code_explain = True

    except Exception:
        # Fallback: keyword-based classification
        q_lower = query.lower()
        if any(k in q_lower for k in ["code", "python", "function", "bug", "error", "script", "sql", "debug", "fix", "class"]):
            category = "code"
        else:
            category = "complex"
        has_code_explain = _looks_like_code_and_explain(query)
        is_multi_part    = has_code_explain or (" and " in q_lower and len(query) > 40)
        reason           = "fallback keyword match (model reply was not valid JSON)"

    # Validate category
    if category not in AVAILABLE_MODELS:
        category = "complex"

    result_data = {
        "category":             category,
        "model":                pick_model(category),
        "is_multi_part":        is_multi_part,
        "has_code_and_explain": has_code_explain,
        "reason":               reason,
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

    # First call — may produce tool_calls
    try:
        response   = ollama.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
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

        # Second call — generate natural language response with tool results in context
        try:
            response = ollama.chat(model=model, messages=messages)
        except Exception as exc:
            answer = f"Tools executed. Error generating follow-up: {exc}"
            add_to_history("user", query)
            add_to_history("assistant", answer)
            return answer

    answer = _get(response, "message", "content") or ""

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

    # Stream first response
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

    # If tools were called, execute them and stream the follow-up
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
        # Stream follow-up response (no tools= here — just natural language)
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

    # Step 1: Code
    code_prompt = (
        f"The user asked: \"{query}\"\n\n"
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

    # Step 2: Explain (MAIN_MODEL sees the code already in shared HISTORY)
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

        # Model may wrap the list in {"tasks": [...]}
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

    # Fallback: treat as a single task
    info = classify_question(query)
    return [{"label": "Full request", "task": query, "category": info["category"], "model": info["model"]}]