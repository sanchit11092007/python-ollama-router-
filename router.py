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
    have context. Fine for a single-user hackathon demo.
"""

import json
import ollama

# ─── Models (must match `ollama list` on your machine) ─────────────────
CODER_MODEL = "qwen2.5-coder:latest"
MAIN_MODEL  = "qwen2.5:14b"
FAST_MODEL  = "qwen2.5:7b"
IMAGE_MODEL = "qwen2.5vl:7b"   # vision model - handles image + text queries

AVAILABLE_MODELS = {
    "code":    CODER_MODEL,
    "simple":  FAST_MODEL,
    "complex": MAIN_MODEL,
    "image":   IMAGE_MODEL,
}

# ─── Conversation memory ────────────────────────────────────────────────
HISTORY   = []
MAX_TURNS = 12


def add_to_history(role: str, content: str):
    HISTORY.append({"role": role, "content": content})
    del HISTORY[:-MAX_TURNS]


def get_history() -> list:
    return HISTORY


def clear_history():
    HISTORY.clear()


# ══════════════════════════════════════════════════════════════════
# STEP 1: Classify - which category, and is it actually multiple asks?
# ══════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are a routing assistant. Read the user's new question
(and the short recent chat history, if any) and decide:

1. category: ONE of
   - "code"    -> writing, fixing, explaining, or debugging code/scripts
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

# Keywords that signal "please also explain the code"
_EXPLAIN_KEYWORDS = [
    "explain", "walk me through", "walk through", "how it works",
    "step by step", "step-by-step", "break it down", "describe",
    "elaborate", "detail", "breakdown",
]

# Keywords that signal "please write code"
_CODE_KEYWORDS = [
    "code", "script", "function", "program", "write", "build",
    "implement", "create", "generate",
]


def _looks_like_code_and_explain(query: str) -> bool:
    """
    Keyword safety-net: returns True when the query wants BOTH code AND
    an explanation of that code. Used as a fallback / extra check.
    """
    q = query.lower()
    has_code    = any(k in q for k in _CODE_KEYWORDS)
    has_explain = any(k in q for k in _EXPLAIN_KEYWORDS)
    return has_code and has_explain


def classify_question(query: str) -> dict:
    recent       = HISTORY[-4:]
    context_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

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
        result           = json.loads(response["message"]["content"])
        category         = result.get("category", "complex")
        is_multi_part    = bool(result.get("is_multi_part", False))
        has_code_explain = bool(result.get("has_code_and_explain", False))
        reason           = result.get("reason", "")

        # Extra safety net: even if the model missed it, keyword check catches it
        if not has_code_explain and _looks_like_code_and_explain(query):
            has_code_explain = True
            is_multi_part    = True

    except Exception:
        category         = "code" if any(
            k in query.lower() for k in ["code", "python", "function", "bug", "error", "script", "sql"]
        ) else "complex"
        has_code_explain = _looks_like_code_and_explain(query)
        is_multi_part    = has_code_explain or " and " in query.lower()
        reason           = "fallback keyword match (model reply was not valid JSON)"

    return {
        "category":             category,
        "model":                pick_model(category),
        "is_multi_part":        is_multi_part,
        "has_code_and_explain": has_code_explain,
        "reason":               reason,
    }


def pick_model(category: str) -> str:
    return AVAILABLE_MODELS.get(category, MAIN_MODEL)


# ══════════════════════════════════════════════════════════════════
# STEP 2: Run a model on a single task (with memory)
# ══════════════════════════════════════════════════════════════════

def _build_messages(query: str) -> list:
    return HISTORY + [{"role": "user", "content": query}]


def get_full_answer(model: str, query: str) -> str:
    messages = _build_messages(query)
    response = ollama.chat(model=model, messages=messages)
    answer   = response["message"]["content"]

    add_to_history("user",      query)
    add_to_history("assistant", answer)
    return answer


def stream_answer(model: str, query: str):
    messages    = _build_messages(query)
    full_answer = ""

    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        token        = chunk["message"]["content"]
        full_answer += token
        yield token

    add_to_history("user",      query)
    add_to_history("assistant", full_answer)


# ══════════════════════════════════════════════════════════════════
# IMAGE: Send an image + question to the vision model
#
#  image_b64 is a raw base64 string (no data-URI prefix).
#  Both functions mirror get_full_answer / stream_answer in style.
# ══════════════════════════════════════════════════════════════════

def ask_image(image_b64: str, query: str) -> str:
    """
    Send an image + question to IMAGE_MODEL, return the full answer.
    image_b64 must be a plain base64-encoded string (PNG / JPEG).
    """
    response = ollama.chat(
        model=IMAGE_MODEL,
        messages=[{
            "role":    "user",
            "content": query,
            "images":  [image_b64],   # ollama accepts raw base64 here
        }],
    )
    answer = response["message"]["content"]
    # Save to shared history so follow-up text questions have context
    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", answer)
    return answer


def stream_image_answer(image_b64: str, query: str):
    """
    Same as ask_image but streams tokens one by one.
    image_b64 must be a plain base64-encoded string (PNG / JPEG).
    """
    full_answer = ""
    for chunk in ollama.chat(
        model=IMAGE_MODEL,
        messages=[{
            "role":    "user",
            "content": query,
            "images":  [image_b64],
        }],
        stream=True,
    ):
        token        = chunk["message"]["content"]
        full_answer += token
        yield token

    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", full_answer)


# ══════════════════════════════════════════════════════════════════
# STEP 3a: Sequential pipeline for "code + explain" requests
#
#  Step 1 -> CODER_MODEL writes the code, saved to shared HISTORY.
#  Step 2 -> MAIN_MODEL reads the history and writes the explanation.
#            It naturally "sees" the code without us repeating it.
# ══════════════════════════════════════════════════════════════════

def run_sequential_tasks(query: str) -> list:
    """
    Handles "give me code AND explain how it works" in two ordered steps.

    Returns a list of result dicts, one per step:
      - step        : step number (1 or 2)
      - label       : short human-readable title
      - model_used  : which model ran this step
      - category    : "code" or "complex"
      - answer      : the model's response text
    """
    results = []

    # ── Step 1: Coder model writes the code ────────────────────────
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

    # ── Step 2: Main model explains the code ───────────────────────
    # HISTORY already contains the code from Step 1, so MAIN_MODEL
    # can refer to it directly — no need to copy-paste the code here.
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


# ══════════════════════════════════════════════════════════════════
# STEP 3b: Split a multi-part question, each part -> its own model
# ══════════════════════════════════════════════════════════════════

def break_into_tasks(query: str) -> list:
    """
    "Write an essay on Programming and give me 10 code snippets"
    -> [ {label: "Essay on Programming", task: "...", model: MAIN_MODEL},
         {label: "10 code snippets",     task: "...", model: CODER_MODEL} ]
    """
    prompt = f"""The user's message contains MULTIPLE distinct requests bundled together.
Split it into separate, self-contained tasks - do NOT merge them back into one.
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
        raw = json.loads(response["message"]["content"])

        # Some models wrap the list in a dict like {"tasks": [...]}. Handle both.
        raw_tasks = raw.get("tasks", raw) if isinstance(raw, dict) else raw

        tasks = [
            {
                "label":    t["label"],
                "task":     t["task"],
                "category": t.get("category", "complex"),
                "model":    pick_model(t.get("category", "complex")),
            }
            for t in raw_tasks
        ]
        if len(tasks) >= 2:
            return tasks
    except Exception:
        pass

    # Fallback: couldn't split cleanly -> treat as one task
    info = classify_question(query)
    return [{"label": "Full request", "task": query, "category": info["category"], "model": info["model"]}]