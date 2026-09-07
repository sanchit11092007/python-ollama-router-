"""
main.py - The FastAPI Server for Agent OTG

Endpoints:
    GET  /            -> Status page
    GET  /health      -> Check if Ollama is running
    POST /ask         -> Ask a question, auto-detects single vs multi-part vs sequential
    POST /ask/stream  -> Same, but streamed with visible stages and token chunks
    POST /ask/complex -> Force-split a question into sub-tasks
    POST /ask/image   -> Ask a question about an image (base64)
    POST /ask/agent   -> LangGraph agent: structured tool-calling graph
    POST /reset       -> Clear conversation memory

Run with:
    py -m uvicorn main:app --reload
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
import datetime
import importlib.util
from urllib.parse import quote
from contextlib import asynccontextmanager

from offline_guard import enable_offline_mode
enable_offline_mode()

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
import router
from router import (
    classify_question,
    stream_answer,
    get_full_answer,
    break_into_tasks,
    run_sequential_tasks,
    clear_history,
    ask_image,
    get_history,
    AVAILABLE_MODELS,
    CODER_MODEL,
    MAIN_MODEL,
)

# ══════════════════════════════════════════════════════════════════════════════
# SERVER LOGGING SYSTEM
# Provides clear, emoji-rich, structured console logging on the Uvicorn server
# ══════════════════════════════════════════════════════════════════════════════

def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_print(msg: str):
    try:
        print(msg, flush=True)
    except (UnicodeEncodeError, Exception):
        try:
            print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"), flush=True)
        except Exception:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

def log_user(msg: str):
    _safe_print(f"\033[94m[{_now_str()}] 📥 [USER ACTION] {msg}\033[0m")

def log_router(msg: str):
    _safe_print(f"\033[95m[{_now_str()}] 🧠 [ROUTER] {msg}\033[0m")

def log_model(msg: str):
    _safe_print(f"\033[96m[{_now_str()}] 🤖 [MODEL] {msg}\033[0m")

def log_tool(msg: str):
    _safe_print(f"\033[93m[{_now_str()}] 🛠️ [TOOL] {msg}\033[0m")

def log_response(msg: str):
    _safe_print(f"\033[92m[{_now_str()}] 📤 [RESPONSE] {msg}\033[0m")

def log_stream(msg: str):
    _safe_print(f"\033[90m[{_now_str()}] 🌊 [STREAM] {msg}\033[0m")

def log_system(msg: str):
    _safe_print(f"\033[97m[{_now_str()}] ⚙️ [SYSTEM] {msg}\033[0m")

def log_err(msg: str):
    _safe_print(f"\033[91m[{_now_str()}] ❌ [ERROR] {msg}\033[0m")


# Hook router.py's internal events into our server logger
def _router_event_listener(event_type: str, data: dict):
    if event_type == "classify":
        res = data.get("result", {})
        log_router(
            f'Classified query="{data.get("query", "")[:60]}" -> '
            f'category={res.get("category")} | model={res.get("model")} | '
            f'is_multi={res.get("is_multi_part")} | has_code_explain={res.get("has_code_and_explain")}'
        )
    elif event_type == "model_start":
        log_model(f'Invoking model={data.get("model")} | query="{data.get("query", "")[:70]}"')
    elif event_type == "model_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_model(f'Model {data.get("model")} completed answer -> "{ans_preview}..."')
    elif event_type == "stream_start":
        log_stream(f'Streaming started for model={data.get("model")} | query="{data.get("query", "")[:70]}"')
    elif event_type == "stream_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_stream(f'Streaming finished for model={data.get("model")} | final="{ans_preview}..."')
    elif event_type == "tool_call":
        log_tool(f'Tool "{data.get("tool")}" called with args={data.get("args")} -> result={data.get("result")[:120]}')
    elif event_type == "image_start":
        log_model(f'Vision model={data.get("model")} analyzing image (b64 size: {data.get("image_size_b64", 0)} chars) | query="{data.get("query", "")}"')
    elif event_type == "image_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_model(f'Vision model completed -> "{ans_preview}..."')

router.set_log_callback(_router_event_listener)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN & APP INIT
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Initialise local persistence ───────────────────────────────────────
    db_ok = db.init_db()
    if db_ok:
        log_system("SQLite session store: connected OK")
    else:
        log_err(f"SQLite session store: UNAVAILABLE — {db.get_error()}")
        log_system("Sessions will not be persisted.")

    # ── Session name ────────────────────────────────────────────────────
    session_name = datetime.datetime.now().strftime("server_session_%Y-%m-%d_%H-%M-%S")
    router.start_new_session(session_name)
    log_system("Agent OTG FastAPI Server Started")
    log_system(f"Active Session: {session_name}")
    log_system(f"Configured Models: {AVAILABLE_MODELS}")
    yield
    log_system("Agent OTG FastAPI Server Stopping...")

app = FastAPI(
    title="Agent OTG",
    description="Routes questions to the best local AI model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    # Log incoming request summary for non-health endpoints to avoid noise
    if path != "/health":
        log_user(f"Incoming {method} {path} from {client_host}")

    try:
        response = await call_next(request)
        elapsed = round(time.time() - start_time, 3)
        if path != "/health":
            log_response(f"Completed {method} {path} with HTTP {response.status_code} in {elapsed}s")
        return response
    except Exception as exc:
        elapsed = round(time.time() - start_time, 3)
        log_err(f"Failed {method} {path} with exception: {exc} (after {elapsed}s)")
        raise


class Question(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")


class ImageQuestion(BaseModel):
    query: str  = Field(..., min_length=1, description="The question about the image")
    image_b64: str = Field(
        ...,
        description=(
            "Raw base64-encoded image (PNG or JPEG). "
            "Do NOT include a data-URI prefix like 'data:image/png;base64,'. "
            "Just the plain base64 string."
        ),
    )


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1, max_length=50)
    replace_existing: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    import config as _cfg
    log_system("Client requested server status page /")
    return {
        "agent":       _cfg.APP_NAME,
        "version":     _cfg.APP_VERSION,
        "made_by":     _cfg.APP_TEAM,
        "description": "100% On-Premise • Air-Gapped • No cloud • No tracking",
        "status":      "online",
        "models":      AVAILABLE_MODELS,
        "endpoints": [
            "/ask", "/ask/stream", "/ask/complex",
            "/ask/image", "/ask/agent",
            "/reset", "/health", "/tools",
            "/sessions", "/knowledge-base/ingest", "/files/{filename}", "/docs",
        ],
    }


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        status = "connected" if r.status_code == 200 else "error"
        return {"ollama": status}
    except Exception as exc:
        log_err(f"Health check failed to reach Ollama: {exc}")
        return {"ollama": "disconnected"}


@app.get("/capabilities")
def capabilities():
    """Report available offline runtimes without contacting any cloud service."""
    optional = {name: importlib.util.find_spec(name) is not None for name in ("pytesseract", "pyttsx3", "speech_recognition")}
    return {
        "offline": True,
        "artifact_formats": ["pdf", "docx", "txt", "md", "pptx", "xlsx", "csv", "json"],
        "optional": optional,
    }


@app.post("/knowledge-base/ingest")
def ingest_knowledge_base(request: IngestRequest):
    try:
        from rag.pipeline import ingest_paths
        return ingest_paths(request.paths, replace_existing=request.replace_existing)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Local file not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {exc}")


@app.post("/reset")
def reset():
    log_user("Requested conversation memory reset")
    clear_history()
    log_system("Conversation history cleared from memory")
    return {"status": "conversation memory cleared"}


@app.post("/ask")
def ask(q: Question):
    try:
        start  = time.time()
        log_user(f'Query: "{q.query}"')
        info   = classify_question(q.query)
        stages = [{"stage": "understanding", "detail": f'category = "{info["category"]}"'}]

        # ── Path A: Code + Explain (sequential, ordered pipeline) ──────
        if info["has_code_and_explain"]:
            log_router(f'Detected Code + Explain combo request. Executing 2-step sequential pipeline...')
            stages.append({
                "stage":  "planning",
                "detail": "Code + explain request — running sequential pipeline...",
            })
            results = run_sequential_tasks(q.query)

            for r in results:
                stages.append({
                    "stage":  "working",
                    "detail": f"Step {r['step']}: {r['label']} → {r['model_used']}",
                })

            elapsed = round(time.time() - start, 2)
            log_response(f"Sequential pipeline completed in {elapsed}s across {len(results)} steps")
            return {
                "type":        "sequential",
                "description": (
                    "Code was generated first by the coder model, then passed "
                    "to the main model via shared history for the explanation."
                ),
                "stages":       stages,
                "steps_run":    len(results),
                "time_seconds": elapsed,
                "results": [
                    {
                        "step":       r["step"],
                        "label":      r["label"],
                        "model_used": r["model_used"],
                        "category":   r["category"],
                        "answer":     r["answer"],
                    }
                    for r in results
                ],
            }

        # ── Path B: Generic multi-part (independent sub-tasks) ─────────
        if info["is_multi_part"]:
            log_router("Detected multi-part question. Splitting into independent sub-tasks...")
            stages.append({"stage": "planning", "detail": "Multi-part request detected, splitting..."})
            tasks   = break_into_tasks(q.query)
            results = []
            for i, task in enumerate(tasks):
                stages.append({
                    "stage":  "working",
                    "detail": f"Sub-task {i+1}: {task['label']} → {task['model']}",
                })
                log_model(f"Running sub-task {i+1}/{len(tasks)}: '{task['label']}' on model {task['model']}")
                answer = get_full_answer(task["model"], task["task"])
                results.append({
                    "task_number": i + 1,
                    "label":       task["label"],
                    "model_used":  task["model"],
                    "category":    task["category"],
                    "answer":      answer,
                })

            elapsed = round(time.time() - start, 2)
            log_response(f"Multi-part execution completed in {elapsed}s across {len(tasks)} sub-tasks")
            return {
                "type":         "multi",
                "stages":       stages,
                "sub_tasks":    len(tasks),
                "time_seconds": elapsed,
                "results":      results,
            }

        # ── Path C: Single model execution ────────────────────────────
        stages.append({"stage": "routing", "detail": f'{info["model"]} - {info["reason"]}'})
        log_router(f'Routing to single model {info["model"]} (Category: {info["category"]})')
        answer = get_full_answer(info["model"], q.query)
        stages.append({"stage": "done", "detail": "answer generated"})

        elapsed = round(time.time() - start, 2)
        log_response(f'Single task completed in {elapsed}s by {info["model"]}')

        return {
            "type": "single",
            "stages": stages,
            "model_used": info["model"],
            "category": info["category"],
            "reason": info["reason"],
            "time_seconds": elapsed,
            "answer": answer,
        }

    except Exception as e:
        log_err(f"Exception in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(q: Question):
    log_user(f'Stream Request: "{q.query}"')

    def generate():
        start = time.time()

        try:
            yield json.dumps({"type": "stage", "label": "understanding", "detail": "Reading your question..."}) + "\n"
            info = classify_question(q.query)

            # ── Path A: Code + Explain sequential streaming ───────────
            if info["has_code_and_explain"]:
                log_router("Stream request: executing Code + Explain sequential pipeline")
                yield json.dumps({
                    "type": "stage",
                    "label": "planning",
                    "detail": "Code + explain combo detected — preparing 2-stage sequential generation...",
                }) + "\n"

                # Step 1: Code generation
                yield json.dumps({
                    "type": "stage",
                    "label": "working",
                    "detail": f"Step 1/2: Generating code with {CODER_MODEL}...",
                }) + "\n"
                code_prompt = (
                    f"The user asked: \"{q.query}\"\n\n"
                    "Your job for this step: write ONLY the code. "
                    "Do not explain it yet — just provide clean, well-commented code."
                )
                for token in stream_answer(CODER_MODEL, code_prompt):
                    yield json.dumps({"type": "token", "step": 1, "content": token}) + "\n"
                yield json.dumps({"type": "step_done", "step": 1}) + "\n"

                # Step 2: Explanation
                yield json.dumps({
                    "type": "stage",
                    "label": "working",
                    "detail": f"Step 2/2: Writing explanation with {MAIN_MODEL}...",
                }) + "\n"
                explain_prompt = (
                    "Now explain the code you just wrote above, step by step. "
                    "Be clear and beginner-friendly. Cover what each part does and why."
                )
                for token in stream_answer(MAIN_MODEL, explain_prompt):
                    yield json.dumps({"type": "token", "step": 2, "content": token}) + "\n"
                yield json.dumps({"type": "step_done", "step": 2}) + "\n"

                elapsed = round(time.time() - start, 2)
                log_stream(f"Sequential streaming finished in {elapsed}s")
                yield json.dumps({"type": "done", "time_seconds": elapsed}) + "\n"

            # ── Path B: Generic multi-part streaming ──────────────────
            elif info["is_multi_part"]:
                log_router("Stream request: breaking down multi-part sub-tasks")
                yield json.dumps({"type": "stage", "label": "planning", "detail": "Multi-part request detected, breaking it down..."}) + "\n"
                tasks = break_into_tasks(q.query)
                yield json.dumps({
                    "type": "plan",
                    "sub_tasks": [{"label": t["label"], "model": t["model"], "category": t["category"]} for t in tasks],
                }) + "\n"

                for i, task in enumerate(tasks, 1):
                    yield json.dumps({
                        "type": "stage",
                        "label": "working",
                        "detail": f"Sub-task {i}/{len(tasks)}: {task['label']} -> {task['model']}",
                    }) + "\n"
                    for token in stream_answer(task["model"], task["task"]):
                        yield json.dumps({"type": "token", "task_number": i, "content": token}) + "\n"
                    yield json.dumps({"type": "task_done", "task_number": i}) + "\n"

                elapsed = round(time.time() - start, 2)
                log_stream(f"Multi-part streaming finished in {elapsed}s")
                yield json.dumps({"type": "done", "time_seconds": elapsed}) + "\n"

            # ── Path C: Single model streaming ────────────────────────
            else:
                model = info["model"]
                log_router(f"Stream request: routing to {model}")
                yield json.dumps({
                    "type": "stage",
                    "label": "routing",
                    "detail": f"Routing to {model} ({info['reason']})",
                    "category": info["category"],
                }) + "\n"
                yield json.dumps({"type": "stage", "label": "generating", "detail": "Writing the answer..."}) + "\n"

                token_count = 0
                for token in stream_answer(model, q.query):
                    token_count += 1
                    yield json.dumps({"type": "token", "content": token}) + "\n"

                elapsed = round(time.time() - start, 2)
                log_stream(f"Single model streaming finished in {elapsed}s with ~{token_count} tokens")
                yield json.dumps({
                    "type": "done",
                    "model_used": model,
                    "token_count": token_count,
                    "time_seconds": elapsed,
                }) + "\n"

        except Exception as e:
            log_err(f"Exception during streaming: {e}")
            yield json.dumps({
                "type": "error",
                "detail": str(e),
                "time_seconds": round(time.time() - start, 2),
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ask/complex")
def ask_complex(q: Question):
    try:
        start = time.time()
        log_user(f'Force-Split Complex Request: "{q.query}"')
        tasks = break_into_tasks(q.query)
        log_router(f"Split into {len(tasks)} sub-tasks")

        results = []
        for i, task in enumerate(tasks):
            log_model(f"Running sub-task {i+1}: '{task['label']}' -> {task['model']}")
            answer = get_full_answer(task["model"], task["task"])
            results.append({
                "task_number": i + 1,
                "label": task["label"],
                "model": task["model"],
                "category": task["category"],
                "answer": answer,
            })

        elapsed = round(time.time() - start, 2)
        log_response(f"Complex tasks finished in {elapsed}s")
        return {
            "sub_tasks": len(tasks),
            "time_seconds": elapsed,
            "results": results,
        }
    except Exception as e:
        log_err(f"Exception in /ask/complex: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/image")
def ask_image_endpoint(q: ImageQuestion):
    try:
        start  = time.time()
        log_user(f'Image Query: "{q.query}" (b64 length: {len(q.image_b64)})')
        answer = ask_image(q.image_b64, q.query)
        elapsed = round(time.time() - start, 2)
        log_response(f"Image analysis completed in {elapsed}s by qwen2.5vl:7b")
        return {
            "type":         "image",
            "model_used":   "qwen2.5vl:7b",
            "time_seconds": elapsed,
            "answer":       answer,
        }
    except Exception as e:
        log_err(f"Exception in /ask/image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/agent")
def ask_agent(q: Question):
    """
    LangGraph agent endpoint.
    Runs the question through a structured multi-turn graph:
      1. classify  — decides if a tool is needed and which one
      2. tools     — executes the real tool from tools.py (if needed)
      3. classify  — loops back to check if more tools needed (up to 5x)
      4. respond   — generates the final answer with full context
    """
    try:
        start = time.time()
        log_user(f'LangGraph Agent Request: "{q.query}"')

        try:
            from langgraph_agent import run_agent
        except ImportError as exc:
            log_err(f"LangGraph not installed: {exc}")
            raise HTTPException(
                status_code=500,
                detail="LangGraph packages not installed. Run: py -m pip install langgraph langchain-ollama"
            )

        history = get_history()
        log_model("Running supervised LangGraph workflow...")
        result = run_agent(q.query, history=history)

        needs_tool = result.get("needs_tool", False)
        tool_log   = result.get("tool_log", [])
        answer     = result.get("final_answer", "").strip()
        artifact   = result.get("artifact")
        if artifact:
            artifact = {**artifact, "download_url": f"/files/{quote(artifact['filename'])}"}

        if needs_tool and tool_log:
            for t in tool_log:
                log_tool(f"Agent tool: {t.get('tool')} -> {str(t.get('result', ''))[:120]}")
        else:
            log_model("Agent decided: direct response (no tool needed)")

        if not answer:
            answer = "Agent completed. Check generated_files/ for any output files."

        elapsed = round(time.time() - start, 2)
        log_response(f"LangGraph agent completed in {elapsed}s | tools_called={len(tool_log)}")

        return {
            "type":         "agent",
            "model_used":   "qwen2.5:14b (LangGraph)",
            "time_seconds": elapsed,
            "needs_tool":   needs_tool,
            "tool_log":     tool_log,
            "answer":       answer,
            "progress":     result.get("events", []),
            "artifact":     artifact,
            "errors":       result.get("errors", []),
            "run_id":       result.get("run_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log_err(f"Exception in /ask/agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
def list_tools():
    """List all tools available to the LangGraph agent."""
    from tools import TOOL_SCHEMAS
    return {
        "tools": [
            {
                "name":        t["function"]["name"],
                "description": t["function"]["description"],
                "required":    t["function"]["parameters"].get("required", []),
            }
            for t in TOOL_SCHEMAS
        ]
    }


@app.get("/files/{filename}")
def download_generated_file(filename: str):
    """Download a validated artifact created by the agent."""
    try:
        from artifacts import resolve_artifact
        path = resolve_artifact(filename)
        return FileResponse(path, filename=path.name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Generated file not found")


@app.get("/sessions")
def list_sessions():
    """List all locally persisted chat sessions."""
    if not db.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Local session store unavailable: {db.get_error()}"
        )
    sessions = db.get_all_sessions(limit=50)
    return {
        "total":    db.get_session_count(),
        "sessions": sessions,
    }


@app.get("/sessions/{session_name}")
def get_session_messages(session_name: str):
    """Retrieve all messages for a locally persisted session."""
    if not db.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Local session store unavailable: {db.get_error()}"
        )
    messages = db.get_session_messages(session_name, limit=500)
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_name}' not found or has no messages."
        )
    return {
        "session_name": session_name,
        "message_count": len(messages),
        "messages": messages,
    }
