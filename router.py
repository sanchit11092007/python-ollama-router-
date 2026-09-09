"""
router.py - Smart multi-model router for Agent OTG

MODELS
    CODER_MODEL : code questions — qwen2.5-coder  ← ALWAYS used for code
    MAIN_MODEL  : complex reasoning, essays, analysis — qwen2.5:7b
    FAST_MODEL  : quick answers, greetings, simple math — qwen2.5:7b
    IMAGE_MODEL : vision analysis — qwen2.5vl:7b


CLASSIFICATION — 5 categories:
    "code"       → CODER_MODEL  (hard-enforced — cannot be overridden)
    "simple"     → FAST_MODEL
    "complex"    → MAIN_MODEL
    "rag_search" → FAST_MODEL (RAG does the heavy lifting)
    "agent_task" → handled by LangGraph agent pipeline

CODE ROUTING GUARANTEE
    Coding questions are routed to CODER_MODEL at THREE enforcement layers:
      1. _hard_code_check()  — fast keyword pre-screen before any LLM call
      2. LLM classifier      — prompted with explicit code-first rules
      3. Post-LLM safety-net — re-checks if LLM returns non-code for a
                               query that contains code keywords
    This makes it impossible for model ambiguity to mis-route a coding
    question to FAST_MODEL or MAIN_MODEL.

SESSION PERSISTENCE
    All messages are stored in SQLite (agent_otg.sqlite3) via db.py.
"""

import json
import os
import re
import ollama
from functools import lru_cache
from langgraph_agent import detect_file_intent
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


@lru_cache(maxsize=1)
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
MAX_TURNS = _cfg.MEMORY_MAX_TURNS


def is_complex_command(query: str) -> bool:
    """True only for the explicit `/complex` command (not normal prompts)."""
    return bool(re.match(r"^\s*/complex(?:\s|$)", str(query or ""), re.I))


def strip_complex_command(query: str) -> str:
    """Remove the terminal command before handing the actual request to 14B."""
    return re.sub(r"^\s*/complex(?:\s+)?", "", str(query or ""), count=1, flags=re.I).strip()


def uses_direct_14b_mode(query: str = "") -> bool:
    """14B bypasses routing only when the user explicitly requests `/complex`."""
    return bool(_cfg.DIRECT_14B_MODE and is_complex_command(query))


def add_to_history(role: str, content: str):
    # Retain concise conversation memory.  The stable system pre-prompt is
    # supplied separately, so duplicated old text only costs context/RAM.
    text = str(content or "")[-_cfg.MEMORY_MAX_CHARS:]
    HISTORY.append({"role": role, "content": text})
    # 1 turn = 1 user prompt + 1 assistant response (2 messages).
    # Keep at least 20 messages (10 full conversational turns) so multi-turn context is not prematurely pruned.
    max_messages = max(MAX_TURNS * 2, 20)
    if len(HISTORY) > max_messages:
        del HISTORY[:len(HISTORY) - max_messages]
    if _CURRENT_SESSION_NAME and _db.is_ready():
        ok = _db.save_message(_CURRENT_SESSION_NAME, role, content)
        if not ok:
            print(f"[router] DB save failed for role={role}", flush=True)


def get_history() -> list:
    return HISTORY


def clear_history():
    global _CURRENT_SESSION_NAME
    HISTORY.clear()
    from datetime import datetime
    new_sess = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_new_session(new_sess)


# ══════════════════════════════════════════════════════════════════════════════
# SMART CLASSIFIER — 5 categories with confidence-based routing
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are an intelligent routing assistant for Agent OTG.
Read the user's message and classify it into EXACTLY one of 5 categories.

CATEGORIES:
- "code"       → ANY question involving programming, coding, scripting, or software development.
                 This includes: writing code, debugging, fixing errors/bugs, explaining code,
                 algorithms, data structures, functions, classes, APIs, SQL queries, regex,
                 shell/bash scripts, web dev (HTML/CSS/JS/React), backend (Flask/FastAPI/Django),
                 system design involving code, performance of code, code reviews, unit tests.
                 Trigger words: code, script, function, program, bug, error, debug, fix,
                 implement, build, class, method, algorithm, API, SQL, query, Python, JavaScript,
                 TypeScript, Java, C++, Rust, Go, React, HTML, CSS, Flask, FastAPI, Django,
                 loop, recursion, array, list, dictionary, compile, runtime, syntax, library,
                 framework, package, import, module, git, docker, database schema, ORM.
                 IMPORTANT: If the topic involves writing or explaining ANY code at all, use "code".
- "simple"     → Greetings, small talk, quick factual one-liners, easy math (< 8 words usually)
                 Examples: "hello", "what time is it", "what is 2+2", "thank you"
- "complex"    → Non-code essays, analysis, research, explanations of concepts (NOT code),
                 multi-step reasoning, general knowledge questions requiring depth.
                 Use this ONLY when there is clearly NO programming/coding content.
- "rag_search" → User wants to search, query, or ask about UPLOADED documents / knowledge base
                 Signals: "what does the document say", "from my files", "search the KB",
                 "what does the report say", "according to the uploaded", "find in docs",
                 "what does it say about", "from the data", "in the file"
- "agent_task" → User wants to CREATE or GENERATE a FILE (PDF, Word, Excel, CSV, PowerPoint)
                 or needs multi-tool autonomous work
                 Signals: "create a PDF", "generate a Word doc", "make a spreadsheet",
                 "write a report and save it", "generate a file", "create an Excel sheet"

RULES (in strict priority order):
1. "code" is the HIGHEST priority category. If there is ANY programming content in the request,
   classify it as "code" — even if it also asks for explanation. Never downgrade a coding question
   to "complex" or "simple".
2. "agent_task" when user explicitly wants a FILE to be created/saved/exported (no code asked)
3. "rag_search" when user references existing documents/files they've uploaded
4. "simple" ONLY for greetings, trivial math, very short factual non-technical questions
5. "complex" for everything else with no coding content
6. "is_multi_part" = true ONLY if message has 2+ clearly distinct, unrelated requests
7. "has_code_and_explain" = true when BOTH writing code AND explaining it is requested

Reply with ONLY this JSON — nothing else:
{"category": "code"|"simple"|"complex"|"rag_search"|"agent_task", "is_multi_part": true|false, "has_code_and_explain": true|false, "reason": "one short sentence", "confidence": 0.0-1.0}
"""

# ── Keyword safety-nets ───────────────────────────────────────────────────────
_EXPLAIN_KEYWORDS = [
    "explain", "walk me through", "walk through", "how it works",
    "step by step", "step-by-step", "break it down", "describe",
    "elaborate", "detail", "breakdown", "walkthrough",
]
# ── Expanded keyword sets ────────────────────────────────────────────────────
_CODE_KEYWORDS = [
    # Generic programming intent
    "code", "script", "function", "program", "debug", "fix", "bug",
    "implement", "build", "test", "class", "method", "algorithm",
    "compile", "runtime", "syntax", "library", "framework", "package",
    "import", "module", "loop", "recursion", "array", "list",
    "dictionary", "variable", "exception", "error handling", "unit test",
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "rust",
    "go", "ruby", "kotlin", "swift", "scala", "r ", "matlab",
    "bash", "shell", "powershell", "perl", "php",
    # Web / frameworks
    "react", "vue", "angular", "html", "css", "sass", "scss",
    "flask", "fastapi", "django", "express", "spring", "rails",
    "next.js", "nuxt", "svelte", "node", "nodejs",
    # Data / DB / DevOps
    "sql", "query", "database", "schema", "orm", "migration",
    "api", "rest", "graphql", "endpoint", "json schema",
    "docker", "kubernetes", "ci/cd", "git", "github actions",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit",
    # System design with code context
    "data structure", "design pattern", "refactor", "optimize code",
    "big-o", "complexity", "pseudocode", "write me a",
]
_CODE_KEYWORD_PHRASES = [
    # Phrases that are unambiguous coding requests (formal, casual, or conversational)
    "write a function", "write a script", "write a program", "write the code",
    "write code", "give me the code", "show me the code", "show me how to code",
    "how do i code", "how to code", "how to write", "how to implement",
    "how to build", "how to create", "how to fix", "how to debug",
    "give me a function", "create a function", "create a class",
    "implement a", "implement the", "build a", "build the",
    "python code", "javascript code", "java code",
    "code me", "code for", "script for", "script to", "code to", "program to",
    "write python", "write javascript", "write java", "write cpp", "write sql",
    "write bash", "write shell", "write html", "write react", "write api",
    "how to solve in python", "how to write in python", "how to code in",
    "bug in my code", "fix my code", "debug my code", "error in my script",
    "code a website", "code an app", "create an api", "build a backend",
]
_SIMPLE_KEYWORDS = [
    "hello", "hi ", "hey ", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "bye", "goodbye", "how are you",
    "what is your name", "who are you", "what time", "what day",
]
_FILE_CREATION_KEYWORDS = [
    # PDF – explicit & implicit phrasing
    "create a pdf", "generate a pdf", "make a pdf", "write a pdf", "build a pdf",
    "pdf on", "pdf about", "pdf of", "convert to pdf", "export to pdf", "save as pdf", "save to pdf",
    "download pdf", "give me a pdf", "make pdf", "create pdf", "generate pdf", "draft a pdf",
    "pdf on the", "pdf about the", "pdf of the", "pdf regarding", "pdf file",
    # DOC / Word – explicit & implicit phrasing
    "create a word", "generate a word", "make a word", "write a word", "give me a word", "draft a word", "prepare a word",
    "create a doc", "generate a doc", "make a doc", "write a doc", "draft a doc", "prepare a doc",
    "create a docx", "generate a docx", "make a docx", "write a docx", "draft a docx", "prepare a docx",
    "docx of", "doc on", "doc about", "doc for", "docx on", "docx about", "docx for",
    "word document", "word doc", "word file", "doc file", "docx file", "ms word", "microsoft word",
    "make a word file", "create word file", "generate word file", "write word file", "draft word file",
    "save as word", "save to word", "convert to word", "export to word", "in word format", "word report",
    "document on", "document about", "document of", "document for",
    # Excel / CSV / Spreadsheet – explicit & implicit
    "create an excel", "make an excel", "generate an excel", "excel sheet", "excel file",
    "generate a spreadsheet", "create a spreadsheet", "make a spreadsheet", "spreadsheet file",
    "spreadsheet on", "spreadsheet about", "export to excel", "save as excel",
    "create a csv", "make a csv", "generate a csv", "csv file", "csv of",
    "csv on", "csv about", "export to csv", "save as csv",
    # Report / File / Presentation – explicit & implicit
    "create a report", "generate a report", "make a report", "write a report", "draft a report",
    "report on", "report about", "report of", "report regarding",
    "create a presentation", "make a powerpoint", "generate a pptx", "powerpoint on",
    "presentation on", "presentation about", "ppt on", "ppt about", "pptx file", "powerpoint file",
    # JSON – explicit & implicit
    "create a json", "generate a json file", "generate a json",
    "json on", "json about", "json file",
    # Generic file creation
    "write a report and save", "create and save", "make and save",
    "make a file", "generate a file", "export a file", "save as file", "save to file",
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
    return _hard_code_check(query) and any(k in q for k in _EXPLAIN_KEYWORDS)


def _hard_code_check(query: str) -> bool:
    """
    Layer-1 hard check: returns True if the query is DEFINITELY about code/programming.
    This is checked BEFORE the LLM classifier and cannot be overridden by LLM output.
    Checks both single keywords and multi-word phrases for higher accuracy.
    """
    q = query.lower().strip()
    # Multi-word phrases (most specific — check first)
    if any(phrase in q for phrase in _CODE_KEYWORD_PHRASES):
        return True
    # Single keywords (broad)
    if any(k in q for k in _CODE_KEYWORDS):
        return True
    return False


def _quick_classify(query: str) -> str | None:
    """
    Fast keyword‑based pre‑screening for obvious cases.
    Returns a category string or None to fall through to LLM classification.
    IMPORTANT: _hard_code_check is always tried first via classify_question()
    before this function is called.
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # Simple: short greetings / small talk matching simple keywords
    if word_count <= 6 and any(k in q for k in _SIMPLE_KEYWORDS):
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

    Enforcement order:
      1. File intent check  — explicit document generation requests always route to agent_task.
      2. _hard_code_check() — deterministic keyword check for non-file programming questions.
      3. _quick_classify()  — fast keyword routing for simple and RAG cases.
      4. LLM classification — for ambiguous queries.
      5. Post-LLM safety-net — safety checks on LLM output.
    """
    # `/complex` explicitly selects Qwen 14B: no classifier call, no task
    # splitting, and no model routing.  All ordinary prompts use the router.
    if uses_direct_14b_mode(query):
        result_data = {
            "category": "complex", "model": _cfg.COMPLEX_MODEL,
            "is_multi_part": False, "has_code_and_explain": False,
            "reason": "Explicit /complex command — Qwen 14B direct mode", "confidence": 1.0,
            "routing_method": "direct_14b",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    recent = HISTORY[-4:]
    context_text = "\n".join(f"{m['role']}: {m['content'][:120]}" for m in recent)

    # ── Layer 0: File creation intent check — requests for document artifacts ──
    file_intent = detect_file_intent(query)
    if file_intent.get("file_format") or any(k in query.lower() for k in _FILE_CREATION_KEYWORDS):
        result_data = {
            "category":             "agent_task",
            "model":                pick_model("agent_task"),
            "is_multi_part":        False,
            "has_code_and_explain": False,
            "reason":               "Explicit file creation request → routed to agent_task",
            "confidence":           1.0,
            "routing_method":       "file_intent_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 1: HARD code check — runs for pure code requests without file intent ────
    if _hard_code_check(query):
        has_code_explain = _looks_like_code_and_explain(query)
        result_data = {
            "category":             "code",
            "model":                pick_model("code"),
            "is_multi_part":        has_code_explain,   # code+explain == 2 steps
            "has_code_and_explain": has_code_explain,
            "reason":               "Hard code keyword match — routed to CODER_MODEL",
            "confidence":           1.0,
            "routing_method":       "hard_code_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 2: fast keyword pre-screen for non-code obvious cases ───────────
    quick = _quick_classify(query)
    if quick:
        result_data = {
            "category":             quick,
            "model":                pick_model(quick),
            "is_multi_part":        False,
            "has_code_and_explain": False,
            "reason":               f"Fast keyword routing → {quick}",
            "confidence":           0.95,
            "routing_method":       "keyword",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 3: LLM classification ───────────────────────────────────────────
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
        parsed           = json.loads(raw)
        category         = parsed.get("category", "complex")
        is_multi_part    = bool(parsed.get("is_multi_part", False))
        has_code_explain = bool(parsed.get("has_code_and_explain", False))
        reason           = parsed.get("reason", "")
        confidence       = float(parsed.get("confidence", 0.7))

        # ── Layer 4: Post-LLM safety-net ─────────────────────────────────────
        # If the LLM says "simple" or "complex" but the query has code keywords,
        # override it. The hard check already handled strong signals, so this
        # catches edge cases where the LLM was still uncertain.
        if category in ("simple", "complex") and _hard_code_check(query):
            category = "code"
            reason   = "Post-LLM override: code keywords detected despite non-code LLM output"
            confidence = 0.9
        # Post-LLM safety-net: detect file intent and override to agent_task
        file_intent = detect_file_intent(query)
        if file_intent.get("file_format") and category != "agent_task":
            category = "agent_task"
            reason = "Post-LLM override: file intent detected"
            confidence = 0.95
        if not has_code_explain and _looks_like_code_and_explain(query):
            has_code_explain = True

    except Exception:
        # ── Layer 5: keyword fallback (LLM call failed) ───────────────────────
        q_lower = query.lower()
        if any(k in q_lower for k in _FILE_CREATION_KEYWORDS):
            category = "agent_task"
        elif any(k in q_lower for k in _RAG_KEYWORDS):
            category = "rag_search"
        elif _hard_code_check(query):   # re-use the same hard check
            category = "code"
        elif len(query.split()) <= 8:
            category = "simple"
        else:
            category = "complex"
        has_code_explain = _looks_like_code_and_explain(query)
        is_multi_part    = has_code_explain or (" and " in query.lower() and len(query) > 40)
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
    return [{"role": "system", "content": sys_prompt}] + HISTORY + [{"role": "user", "content": strip_complex_command(query)}]


def get_full_answer(model: str, query: str) -> str:
    from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
    _log_event("model_start", {"model": model, "query": query})
    messages = _build_messages(query)

    try:
        # Tool schemas are large and slow local inference considerably.  The
        # terminal sends explicit document/tool requests to the deterministic
        # agent path, so ordinary answers do not need them in their pre-prompt.
        response = ollama.chat(model=model, messages=messages,
                               options={"num_ctx": _cfg.OLLAMA_NUM_CTX,
                                        "num_predict": _cfg.OLLAMA_NUM_PREDICT})
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
        for chunk in ollama.chat(model=model, messages=messages, stream=True,
                                 options={"num_ctx": _cfg.OLLAMA_NUM_CTX,
                                          "num_predict": _cfg.OLLAMA_NUM_PREDICT}):
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
