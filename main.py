"""
main.py - The FastAPI Server

Endpoints:
    GET  /            -> Status page
    GET  /health      -> Check if Ollama is running
    POST /ask         -> Ask a question, auto-detects single vs multi-part
    POST /ask/stream  -> Same, but streamed with visible stages
    POST /ask/complex -> Force-split a question into sub-tasks
    POST /reset       -> Clear conversation memory

Run with:
    uvicorn main:app --reload
"""

import json
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from router import (
    classify_question,
    stream_answer,
    get_full_answer,
    break_into_tasks,
    run_sequential_tasks,
    clear_history,
    AVAILABLE_MODELS,
)

app = FastAPI(
    title="Local AI Router",
    description="Routes questions to the best local AI model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")


@app.get("/")
def home():
    return {
        "status": "online",
        "models": AVAILABLE_MODELS,
        "endpoints": ["/ask", "/ask/stream", "/ask/complex", "/reset", "/health", "/docs"],
    }


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return {"ollama": "connected" if r.status_code == 200 else "error"}
    except Exception:
        return {"ollama": "disconnected"}


@app.post("/reset")
def reset():
    clear_history()
    return {"status": "conversation memory cleared"}


@app.post("/ask")
def ask(q: Question):
    try:
        start  = time.time()
        info   = classify_question(q.query)
        stages = [{"stage": "understanding", "detail": f'category = "{info["category"]}"'}]

        # ── Path A: Code + Explain (sequential, ordered pipeline) ──────
        # Step 1: CODER_MODEL generates the code.
        # Step 2: MAIN_MODEL reads history and writes the explanation.
        if info["has_code_and_explain"]:
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

            return {
                # "type" tells the caller exactly which path was taken
                "type":        "sequential",
                "description": (
                    "Code was generated first by the coder model, then passed "
                    "to the main model via shared history for the explanation."
                ),
                "stages":       stages,
                "steps_run":    len(results),
                "time_seconds": round(time.time() - start, 2),
                # Every result item clearly names which model handled it
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
            stages.append({"stage": "planning", "detail": "Multi-part request detected, splitting..."})
            tasks   = break_into_tasks(q.query)
            results = []
            for i, task in enumerate(tasks):
                stages.append({
                    "stage":  "working",
                    "detail": f"Sub-task {i+1}: {task['label']} → {task['model']}",
                })
                answer = get_full_answer(task["model"], task["task"])
                results.append({
                    "task_number": i + 1,
                    "label":       task["label"],
                    "model_used":  task["model"],
                    "category":    task["category"],
                    "answer":      answer,
                })
            return {
                "type":         "multi",
                "stages":       stages,
                "sub_tasks":    len(tasks),
                "time_seconds": round(time.time() - start, 2),
                "results":      results,
            }

        stages.append({"stage": "routing", "detail": f'{info["model"]} - {info["reason"]}'})
        answer = get_full_answer(info["model"], q.query)
        stages.append({"stage": "done", "detail": "answer generated"})

        return {
            "type": "single",
            "stages": stages,
            "model_used": info["model"],
            "category": info["category"],
            "reason": info["reason"],
            "time_seconds": round(time.time() - start, 2),
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(q: Question):
    def generate():
        start = time.time()

        yield json.dumps({"type": "stage", "label": "understanding", "detail": "Reading your question..."}) + "\n"
        info = classify_question(q.query)

        if info["is_multi_part"]:
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

            yield json.dumps({"type": "done", "time_seconds": round(time.time() - start, 2)}) + "\n"

        else:
            model = info["model"]
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

            yield json.dumps({
                "type": "done",
                "model_used": model,
                "token_count": token_count,
                "time_seconds": round(time.time() - start, 2),
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ask/complex")
def ask_complex(q: Question):
    try:
        start = time.time()
        tasks = break_into_tasks(q.query)

        results = []
        for i, task in enumerate(tasks):
            answer = get_full_answer(task["model"], task["task"])
            results.append({
                "task_number": i + 1,
                "label": task["label"],
                "model": task["model"],
                "category": task["category"],
                "answer": answer,
            })

        return {
            "sub_tasks": len(tasks),
            "time_seconds": round(time.time() - start, 2),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))