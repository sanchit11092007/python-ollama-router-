"""
ask.py - The Interactive Terminal Client for Agent OTG
Developed by Team DWE

Features:
    - Smart Intent Detection: automatically routes to RAG, Agent, Vision, or Chat
      WITHOUT requiring /rag, /agent, /image prefixes
    - Advanced 5-Category Routing: code, simple, complex, rag_search, agent_task
    - Premium Terminal UI with rich formatting and animations
    - Document Ingestion: PDF/DOCX/CSV/Excel/JSON into RAG knowledge base
    - LangGraph Autonomous Agent: multi-step structured agent
    - All generated files saved to ~/Downloads/AgentOTG/
"""

import sys
import time
import base64
import os
import threading
import datetime
import json
import re
import traceback
import requests as _requests

from offline_guard import enable_offline_mode, allow_external
enable_offline_mode()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import db
import router
from router import (
    classify_question,
    stream_answer,
    stream_image_answer,
    break_into_tasks,
    run_sequential_tasks,
    clear_history,
    get_history,
    CODER_MODEL,
    MAIN_MODEL,
    FAST_MODEL,
    IMAGE_MODEL,
    AVAILABLE_MODELS,
)
import config as _cfg

# ── Output directory info ─────────────────────────────────────────────────────
from artifacts import OUTPUT_DIR as _OUTPUT_DIR

# ── Status phrases for spinner ────────────────────────────────────────────────
THINKING_PHRASES = [
    "🧠 Analyzing intent...",
    "⚡ Routing to optimal model...",
    "💭 Composing precise response...",
    "✨ Synthesizing output...",
    "🔍 Processing context...",
]

# ── Max characters to show per message in history view ───────────────────────
_HISTORY_MSG_TRUNCATE = 200
_HISTORY_MAX_MSGS     = 20

# ── Smart dispatch intent keywords ───────────────────────────────────────────
# Used to auto-detect RAG / agent intent WITHOUT requiring /rag or /agent prefix
_RAG_AUTO_TRIGGERS = [
    "what does the document", "what does my document", "what does the file",
    "from the uploaded", "in my files", "search the knowledge base",
    "from the knowledge base", "what does the report say", "search my docs",
    "what does the uploaded document", "what does the uploaded file",
    "what does it say", "according to the document", "according to the report",
    "from the data", "in the pdf", "in the doc",
    "find in the document", "look up in", "from the uploaded file",
    "what the document says", "search my knowledge",
    "what does the report", "what are the findings", "extract from",
    "search my documents", "query the knowledge base",
    "what does my report", "what does my file",
]

_AGENT_AUTO_TRIGGERS = [
    "create a pdf", "generate a pdf", "make a pdf", "write a pdf",
    "create a word", "generate a word", "make a word doc",
    "create a doc", "generate a doc",
    "create an excel", "make an excel", "generate an excel",
    "generate a spreadsheet", "create a spreadsheet", "make a spreadsheet",
    "create a csv", "make a csv", "generate a csv",
    "create a report and save", "generate a report", "save as pdf",
    "export to pdf", "create a presentation", "make a powerpoint",
    "generate a pptx", "create a json file", "generate a json",
    "write a report and", "create a file", "generate a file",
    "make a file", "write and save", "create and save",
    "make and save", "export a file",
]

_IMAGE_AUTO_TRIGGERS = [
    "analyze this image", "what is in this image", "describe this image",
    "what does this picture show", "look at this image",
    "analyze the image", "what is shown in",
]


def _clean_path(path: str) -> str:
    """Clean surrounding quotes and whitespace from paths."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


# ══════════════════════════════════════════════════════════════════════════════
# SPINNER & STREAMING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def show_spinner(messages, stop_flag, start_time):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    msg_index = 0
    last_switch = start_time
    while not stop_flag[0]:
        now = time.time()
        elapsed = now - start_time
        if len(messages) > 1 and now - last_switch > 2.2:
            msg_index = (msg_index + 1) % len(messages)
            last_switch = now
        sys.stdout.write(
            f"\r  \033[96m{frames[i % len(frames)]}\033[0m "
            f"\033[93m{messages[msg_index]}\033[0m "
            f"\033[90m({elapsed:.1f}s)\033[0m "
        )
        sys.stdout.flush()
        i += 1
        time.sleep(0.08)
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()


def stage(badge: str, text: str, style="bold cyan"):
    console.print(f"  [{style}]{badge}[/{style}] {text}")


def run_with_spinner(messages, work_fn):
    """Runs work_fn() while an animated spinner displays."""
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(messages, stop, start_time), daemon=True)
    spinner.start()
    try:
        result = work_fn()
    finally:
        stop[0] = True
        spinner.join()
    return result


def stream_with_spinner(model, query):
    """Shows a spinner until the first token arrives, then streams tokens live."""
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time), daemon=True)
    spinner.start()

    full_answer = ""
    spinner_stopped = False
    try:
        for token in stream_answer(model, query):
            if not spinner_stopped:
                stop[0] = True
                spinner.join()
                spinner_stopped = True

            if token.startswith("\n\n[System: Calling tool"):
                sys.stdout.write("\n")
                console.print(f"  [bold yellow]{token.strip()}[/bold yellow]")
                sys.stdout.write("\n")
            else:
                full_answer += token
                sys.stdout.write(token)
            sys.stdout.flush()
    finally:
        if not spinner_stopped:
            stop[0] = True
            spinner.join()

    print()
    return len(full_answer.split())


# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM UI — Welcome Banner & Help
# ══════════════════════════════════════════════════════════════════════════════

def _check_ollama_models() -> dict:
    """Check which configured models are available in Ollama."""
    available = {}
    try:
        import ollama as _ollama
        with allow_external():
            pulled = {m["name"] for m in _ollama.list()["models"]}
        pulled_base = {n.split(":")[0] for n in pulled} | pulled
        for role, model in AVAILABLE_MODELS.items():
            base = model.split(":")[0]
            available[model] = (model in pulled) or (base in pulled_base)
    except Exception:
        for model in AVAILABLE_MODELS.values():
            available[model] = True
    return available


def print_welcome_banner():
    console.print()

    # ── DWE Team Credits Panel ─────────────────────────────────────────────
    credits_text = (
        f"[bold white]{_cfg.APP_NAME}[/bold white]  [dim]v{_cfg.APP_VERSION}[/dim]\n\n"
        f"[dim]Crafted with ❤️  by the [/dim][bold bright_magenta]{_cfg.APP_TEAM}[/bold bright_magenta]\n"
        "[dim]Sanchit  •  DWE Member 2  •  DWE Member 3[/dim]\n\n"
        "[dim]🔒 100% On-Premise  •  Air-Gapped  •  No Cloud  •  No Tracking[/dim]\n"
        "[dim]Your data never leaves this machine.[/dim]"
    )
    console.print(Panel(
        credits_text,
        box=box.DOUBLE,
        border_style="bright_magenta",
        padding=(1, 4),
        title="[bold bright_magenta]★  AGENT OTG  ★[/bold bright_magenta]",
        subtitle="[dim]Developed by DWE Team[/dim]",
    ))

    # ── Main system banner ─────────────────────────────────────────────────
    banner_text = (
        "[bold cyan]🤖 AGENT OTG — Local Autonomous Multi-Model AI[/bold cyan]\n"
        "[dim]Type anything naturally — no commands needed. The system detects your intent automatically.[/dim]\n\n"
        f"[dim]📁 Output files → [yellow]{_OUTPUT_DIR}[/yellow][/dim]"
    )
    console.print(Panel(banner_text, box=box.ROUNDED, border_style="cyan", padding=(1, 3)))

    # ── Model roster ───────────────────────────────────────────────────────
    model_table = Table(
        title="🤖 Active Model Roster",
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        show_header=True,
        padding=(0, 1),
    )
    model_table.add_column("Role",    style="bold green",  no_wrap=True)
    model_table.add_column("Model",   style="bold yellow", no_wrap=True)
    model_table.add_column("Handles", style="dim white")
    model_table.add_row("⚡ Fast",    FAST_MODEL,    "Greetings, quick Q&A, routing decisions, RAG formatting")
    model_table.add_row("💻 Coder",   CODER_MODEL,   "Code generation, debugging, SQL, APIs, scripts")
    model_table.add_row("🧠 Main",    MAIN_MODEL,    "General reasoning, essays, analysis, agent tasks")
    model_table.add_row("🧠 Deep",    _cfg.COMPLEX_MODEL, "Direct answer only when prefixed with /complex")
    model_table.add_row("👁️  Vision",  IMAGE_MODEL,   "Image analysis, diagrams, screenshots, OCR")
    console.print(model_table)

    # ── Capability guide ───────────────────────────────────────────────────
    cap_table = Table(
        title="🚀 Capabilities — Type Naturally (No Commands Required)",
        box=box.SIMPLE_HEAVY,
        border_style="bright_blue",
        show_header=True,
        padding=(0, 1),
    )
    cap_table.add_column("Capability",          style="bold green",   no_wrap=True)
    cap_table.add_column("Just Type…",          style="yellow")
    cap_table.add_column("Or Use Command",       style="dim cyan",     no_wrap=True)
    cap_table.add_column("Routed To",           style="magenta",      no_wrap=True)

    cap_table.add_row(
        "💻 Code Generation",
        "Write a FastAPI middleware for rate limiting",
        "—", CODER_MODEL,
    )
    cap_table.add_row(
        "🔄 Code + Explain",
        "Write a binary search function and explain it",
        "—", f"{CODER_MODEL} → {MAIN_MODEL}",
    )
    cap_table.add_row(
        "🧠 Deep Reasoning",
        "Compare PostgreSQL vs MongoDB for high-write loads",
        "—", MAIN_MODEL,
    )
    cap_table.add_row(
        "⚡ Quick Chat",
        "Hello / What is 25 × 4?",
        "—", FAST_MODEL,
    )
    cap_table.add_row(
        "🔍 RAG Search",
        "What does my report say about emissions?",
        "/rag <question>", FAST_MODEL,
    )
    cap_table.add_row(
        "📂 Ingest File",
        "Paste a file path (auto-detected)",
        "/doc <file path>", "Vector Store",
    )
    cap_table.add_row(
        "📄 Create PDF",
        "Create a PDF report on AI trends",
        "/agent <task>", "LangGraph + " + MAIN_MODEL,
    )
    cap_table.add_row(
        "📝 Create Word",
        "Generate a Word doc with Q3 highlights",
        "/agent <task>", "LangGraph + " + MAIN_MODEL,
    )
    cap_table.add_row(
        "👁️  Vision / Image",
        "/image <path or URL> [question]",
        "/img <path>", IMAGE_MODEL,
    )
    cap_table.add_row("🧠  Deep Direct", "/complex <query>", "/complex <query>", _cfg.COMPLEX_MODEL)

    console.print(cap_table)

    # ── Quick commands ─────────────────────────────────────────────────────
    cmd_table = Table(
        title="⌨️  Quick Commands",
        box=box.SIMPLE,
        border_style="dim",
        show_header=False,
        padding=(0, 2),
    )
    cmd_table.add_column("Command",  style="bold cyan",   no_wrap=True)
    cmd_table.add_column("Action",   style="dim white")
    cmd_table.add_row("/models",  "Show active model configuration")
    cmd_table.add_row("/kb",      "Show RAG knowledge base statistics")
    cmd_table.add_row("/files",   "List all generated files in Downloads/AgentOTG/")
    cmd_table.add_row("history",  "Browse saved chat sessions")
    cmd_table.add_row("reset",    "Clear conversation memory")
    cmd_table.add_row("/cancel",  "Cancel active file attachment")
    cmd_table.add_row("help",     "Show detailed example guide")
    cmd_table.add_row("exit",     "Quit Agent OTG")
    console.print(cmd_table)

    console.print(
        "\n  [dim]💡 [bold cyan]Pro tip:[/bold cyan] Just type naturally — "
        "Agent OTG detects whether to search documents, create files, or answer directly.[/dim]\n"
    )


def print_help_guide():
    console.print()
    console.print(Rule("[bold cyan]Agent OTG — Practical Example Guide[/bold cyan]", style="bright_blue"))

    examples = [
        ("💻 Coding (auto-detected)",
         'Write a Python FastAPI middleware for request timing and explain it.'),
        ("🧠 Reasoning (auto-detected)",
         'Compare PostgreSQL vs MongoDB for high-write telemetry systems.'),
        ("⚡ Quick (auto-detected)",
         'Hello / What is the capital of France?'),
        ("🔍 RAG — auto-detect",
         'What does the report say about the safety guidelines?'),
        ("🔍 RAG — explicit",
         '/rag What are the main findings in the uploaded document?'),
        ("📂 Ingest file (just paste path)",
         r'C:\Users\me\Downloads\report.pdf  (auto-detected as file path)'),
        ("📂 Ingest + query in one step",
         r'/doc "C:\Users\me\Downloads\etp.csv" Analyze this and give 10 key points'),
        ("📄 Create PDF (auto-detected)",
         'Create a PDF report on machine learning trends with 5 key sections.'),
        ("📝 Create Word (auto-detected)",
         'Generate a Word document titled "Q3 Report" with 3 bullet highlights.'),
        ("📊 Create Excel",
         'Create an Excel sheet with columns: Name, Score, Grade, and 5 sample rows.'),
        ("👁️ Vision / Image",
         '/image "C:\\Users\\me\\Pictures\\diagram.png" What architecture is shown here?'),
        ("👁️ Vision (Web URL)",
         '/image https://example.com/logo.png Describe this logo in detail.'),
        ("🛠️ Agent — Chain tasks",
         '/agent Generate a PDF about Python basics, then tell me how many pages it has.'),
        ("🧠 Direct 14B reasoning",
         '/complex Compare three database designs for a high-volume financial ledger.'),
        ("🧹 Reset memory",
         'reset  — clears conversation context for a fresh start'),
        ("🗑️ List output files",
         '/files  — shows all files in ~/Downloads/AgentOTG/'),
    ]

    for cat, ex in examples:
        console.print(f"\n[bold green]{cat}:[/bold green]")
        console.print(f"  [yellow]{ex}[/yellow]")

    console.print()
    console.print(Rule(style="bright_blue"))
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# SMART INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _is_multi_request(query: str) -> bool:
    """True if query contains multiple distinct requests, numbered items, or mixed actions."""
    q = (query or "").strip()
    # Check for numbered items like 1) ... 2) ... or 1. ... 2. ...
    if re.search(r"(?:^|\s)(?:1[\).]|first[,:])\s+.*?(?:\s+(?:2[\).]|second[,:]))\s+", q, re.I | re.DOTALL):
        return True
    # Check for bulleted list items
    if re.search(r"(?:^|\n)\s*[-*•]\s+.*?\n\s*[-*•]\s+", q):
        return True
    # Check for multiple action verbs with distinct objects
    actions = re.findall(r"\b(?:write|create|generate|make|draft|give|explain|summarize|compare|draw)\b", q, re.I)
    if len(actions) >= 2 and any(sep in q.lower() for sep in [";", "\n", " and also ", ", also ", " and then ", " 2)", " 2."]):
        return True
    return _looks_like_mixed_file_request(q)


def _detect_intent(query: str) -> str:
    """
    Detect the primary intent of a natural-language query WITHOUT requiring
    command prefixes. Returns one of:
      'rag'    → search knowledge base
      'agent'  → create a file / autonomous agent task
      'image_gen' → create a local image with SD-Turbo
      'normal' → standard chat/code/reasoning
      'file'   → user pasted a file path (ingest it)
    """
    # If the user has bundled multiple requests together, let the multi-task router decompose them
    if _is_multi_request(query):
        return "normal"

    q = query.lower().strip()

    # RAG search signals
    if any(trigger in q for trigger in _RAG_AUTO_TRIGGERS):
        return "rag"

    # Agent/file creation signals
    if any(trigger in q for trigger in _AGENT_AUTO_TRIGGERS) and not _looks_like_mixed_file_request(query):
        return "agent"

    if re.search(r"\b(?:generate|create|make|draw)\s+(?:an?\s+)?image\b", q):
        return "image_gen"

    return "normal"


def _show_intent_badge(intent: str, query_preview: str = ""):
    """Show a styled badge when auto-routing to a special handler."""
    badges = {
        "rag":   ("🔍", "RAG Search",   "Auto-detected: searching knowledge base...", "bold blue"),
        "agent": ("📄", "File Creator", "Auto-detected: file creation task...",       "bold magenta"),
        "image_gen": ("🖼️", "Image Generator", "Auto-detected: generating a local image...", "bold magenta"),
    }
    if intent in badges:
        icon, label, detail, style = badges[intent]
        console.print(
            Panel(
                f"[{style}]{icon} Auto-Routing → {label}[/{style}]\n[dim]{detail}[/dim]",
                border_style=style.split()[1],
                padding=(0, 2),
            )
        )


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _show_route_card(info: dict):
    """Display a visually rich routing decision card."""
    category   = info.get("category", "complex")
    model      = info.get("model", MAIN_MODEL)
    reason     = info.get("reason", "")
    confidence = info.get("confidence", 0.0)
    method     = info.get("routing_method", "llm")

    emoji_map = {
        "code":       "💻",
        "simple":     "⚡",
        "complex":    "🧠",
        "rag_search": "🔍",
        "agent_task": "📄",
    }
    color_map = {
        "code":       "green",
        "simple":     "yellow",
        "complex":    "cyan",
        "rag_search": "blue",
        "agent_task": "magenta",
    }
    icon  = emoji_map.get(category, "🤖")
    color = color_map.get(category, "cyan")

    conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    method_label = "⚡ keyword" if method == "keyword" else "🧠 LLM"

    stage(
        f"{icon} [Route]",
        f"Category: [{color}]{category}[/{color}] → Model: [bold {color}]{model}[/bold {color}]"
        f"  [dim]| Confidence: {conf_bar} {confidence:.0%} | Via: {method_label}[/dim]",
        style=f"bold {color}",
    )
    if reason:
        stage("💡 [Reason]", reason, style="dim")
    console.print("─" * 65, style="dim")


def handle_single(query, info):
    _show_route_card(info)
    model = info["model"]
    start_time = time.time()
    token_count = stream_with_spinner(model, query)
    elapsed = round(time.time() - start_time, 2)

    console.print("─" * 65, style="dim")
    console.print(
        f"  ✅ [bold green]Done in {elapsed}s[/bold green] | "
        f"Model: [cyan]{model}[/cyan] | "
        f"~[yellow]{token_count}[/yellow] words\n"
    )


def handle_image_generation(query: str):
    """Run the local SD-Turbo generator; never route an image request to Ollama chat."""
    prompt = re.sub(
        r"^\s*(?:generate|create|make|draw)\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
        "", query, flags=re.I,
    ).strip()
    if not prompt:
        console.print("  ❌ [bold red]Please include an image description.[/bold red]\n")
        return
    stage("🖼️ [Image Generation]", "Loading local SD-Turbo model (first use can take longer)...", style="bold magenta")
    from image_gen import generate_image
    result = run_with_spinner(["Loading SD-Turbo pipeline...", "Generating image..."], lambda: generate_image(prompt))
    if result.startswith("❌"):
        console.print(f"  {result}\n", style="bold red")
    else:
        console.print(f"  {result}\n", style="bold green")


def handle_sequential(query):
    """Handles 'Code + Explain' queries in two ordered, chained steps."""
    stage("🔄 [Pipeline]", "Code + Explanation combo detected → 2-Step Pipeline", style="bold magenta")
    console.print("─" * 65, style="dim")

    start_time = time.time()

    stage("💻 [Step 1/2]", f"Generating code with [bold cyan]{CODER_MODEL}[/bold cyan]...", style="bold green")
    code_prompt = (
        f'The user asked: "{query}"\n\n'
        "Your job for this step: write ONLY the code. "
        "Do not explain it yet — just provide clean, well-commented code."
    )
    tokens_code = stream_with_spinner(CODER_MODEL, code_prompt)
    console.print()

    stage("🧠 [Step 2/2]", f"Writing step-by-step walkthrough with [bold cyan]{MAIN_MODEL}[/bold cyan]...", style="bold green")
    explain_prompt = (
        "Now explain the code you just wrote above, step by step. "
        "Be clear and beginner-friendly. Cover what each part does and why."
    )
    tokens_explain = stream_with_spinner(MAIN_MODEL, explain_prompt)

    elapsed = round(time.time() - start_time, 2)
    console.print("─" * 65, style="dim")
    console.print(
        f"  ✅ [bold green]Sequential Pipeline done in {elapsed}s[/bold green] | "
        f"Models: [cyan]{CODER_MODEL}[/cyan] & [cyan]{MAIN_MODEL}[/cyan] | "
        f"~[yellow]{tokens_code + tokens_explain}[/yellow] total words\n"
    )


def handle_multi(query):
    stage("🗂️ [Planning]", "Multi-part request detected → decomposing into sub-tasks...", style="bold yellow")
    tasks = run_with_spinner(["Analyzing and structuring sub-tasks..."], lambda: break_into_tasks(query))

    table = Table(
        title=f"📋 Execution Plan ({len(tasks)} Sub-Tasks)",
        box=box.ROUNDED,
        border_style="yellow",
    )
    table.add_column("#",               justify="right", style="bold cyan")
    table.add_column("Sub-Task Title",  style="bold white")
    table.add_column("Model",           style="bold magenta")
    table.add_column("Category",        style="yellow")

    for i, task in enumerate(tasks, 1):
        table.add_row(str(i), task["label"], task["model"], task["category"])
    console.print(table)
    console.print()

    start_time = time.time()
    for i, task in enumerate(tasks, 1):
        stage(f"⚙️ [{i}/{len(tasks)}]", f"{task['label']} via [bold cyan]{task['model']}[/bold cyan]")
        console.print("─" * 65, style="dim")
        # A file sub-task must use the deterministic artifact workflow even
        # when it originated inside a larger multi-part request.
        if re.search(r"\b(?:generate|create|make|draw)\s+(?:an?\s+)?image\b", task["task"], re.I):
            handle_image_generation(task["task"])
        elif is_file_creation_request(task["task"]):
            handle_agent(task["task"])
        else:
            stream_with_spinner(task["model"], task["task"])
        console.print("─" * 65 + "\n", style="dim")

    elapsed = round(time.time() - start_time, 2)
    console.print(f"  ✅ [bold green]All {len(tasks)} sub-tasks completed in {elapsed}s[/bold green]\n")


def is_file_creation_request(text: str) -> bool:
    """Detects whether user prompt is asking to generate, save, or export a file/document."""
    t = text.lower()
    has_action = bool(re.search(r"\b(generate|generatet|create|make|save|export|write|build|output|download|draft|prepare|provide|deliver|give|craft|produce|convert)\b", t))
    has_file   = bool(re.search(r"\b(pdf|docx|doc|word doc|word document|word file|word report|word|ms word|microsoft word|excel|xlsx|xls|csv|pptx|ppt|powerpoint|spreadsheet|json|text file|txt)\b", t))
    direct_phrase = bool(re.search(r"\b(word file|doc file|docx file|pdf file|excel file|excel sheet|pptx file|save as|export to|convert to|in word|in pdf|in excel)\b", t))
    return (has_action and has_file) or direct_phrase


def _looks_like_mixed_file_request(text: str) -> bool:
    """True when a file request is only one part of a larger request."""
    if not is_file_creation_request(text):
        return False
    actions = re.findall(r"\b(?:write|create|generate|make|draft|give|explain)\b", text, re.I)
    return len(actions) >= 2


def ask_anything(query, force_multi=False):
    """Main dispatcher for standard queries (non-agent, non-RAG)."""
    # A configured Qwen 14B is explicitly a direct-answer model.  Do this
    # before any automatic intent/routing work so terminal latency is one model
    # call and the prompt receives the shared system pre-prompt only once.
    if router.uses_direct_14b_mode(query):
        stage("🧠 [Qwen 14B]", "Direct response mode (routing disabled).", style="bold cyan")
        stream_with_spinner(_cfg.COMPLEX_MODEL, router.strip_complex_command(query))
        return
    if not force_multi and _is_multi_request(query):
        handle_multi(query)
        return

    if not force_multi and is_file_creation_request(query) and not _looks_like_mixed_file_request(query):
        handle_agent(query)
        return

    if not force_multi and _looks_like_mixed_file_request(query):
        handle_multi(query)
        return

    stage("🧠 [Understanding]", "Analyzing question structure and context...", style="bold blue")
    info = run_with_spinner(["Evaluating intent and routing logic..."], lambda: classify_question(query))

    # If classifier detected a RAG or agent category, route accordingly
    category = info.get("category", "complex")
    if category == "rag_search":
        handle_rag(query)
        return
    elif category == "agent_task":
        handle_agent(query)
        return

    if force_multi:
        handle_multi(query)
    elif info.get("has_code_and_explain"):
        handle_sequential(query)
    elif info.get("is_multi_part"):
        handle_multi(query)
    else:
        handle_single(query, info)


def handle_agent(query):
    stage("🛠️ [LangGraph Agent]", "Initializing multi-tool reasoning workflow...", style="bold magenta")
    stage("🤖 [Model]", f"[bold cyan]{MAIN_MODEL}[/bold cyan] with dynamic tool calling & RAG integration", style="cyan")
    console.print("─" * 65, style="dim")

    start_time = time.time()
    try:
        from langgraph_agent import run_agent
    except ImportError as e:
        console.print(f"  ❌ [bold red]LangGraph not available:[/bold red] {e}")
        console.print("  [dim]Install: py -m pip install langgraph langchain-ollama[/dim]")
        return

    result = run_with_spinner(
        ["Analyzing intent...", "Invoking registered tools...", "Checking if more tools needed...", "Synthesizing response..."],
        lambda: run_agent(query, history=get_history()),
    )

    elapsed = round(time.time() - start_time, 2)

    for event in result.get("events", []):
        if event.get("stage") == "timing":
            console.print(f"  ⏱️ [dim]{event.get('detail')}[/dim]")

    # Show tool execution table
    if result.get("needs_tool") and result.get("tool_log"):
        tool_table = Table(
            title="🛠️ Tools Executed by Agent",
            box=box.ROUNDED,
            border_style="green",
        )
        tool_table.add_column("#",      justify="right", style="bold cyan", no_wrap=True)
        tool_table.add_column("Tool",   style="bold yellow")
        tool_table.add_column("Args",   style="dim")
        tool_table.add_column("Result", style="white")
        for i, entry in enumerate(result["tool_log"], 1):
            args_str   = ", ".join(f"{k}={repr(str(v))[:35]}" for k, v in entry.get("args", {}).items())
            result_str = str(entry.get("result", ""))[:150]
            tool_table.add_row(str(i), entry["tool"], args_str, result_str)
        console.print(tool_table)
        console.print()

    # Render final answer
    final_answer = result.get("final_answer", "").strip()
    if not final_answer:
        final_answer = "Agent completed the task. Check your Downloads/AgentOTG/ folder for generated files."

    console.print(Markdown(final_answer))
    console.print("─" * 65, style="dim")

    tool_count = len(result.get("tool_log", []))
    tool_info  = f" | [green]{tool_count} tool(s) invoked[/green]" if tool_count else ""
    errors     = result.get("errors", [])
    err_info   = f" | [red]{len(errors)} error(s)[/red]" if errors else ""

    console.print(
        f"  ✅ [bold green]Agent done in {elapsed}s[/bold green]"
        f"{tool_info}{err_info} | "
        f"[cyan]{MAIN_MODEL}[/cyan] | "
        f"[dim]Files → {_OUTPUT_DIR}[/dim]\n"
    )


def parse_image_command(rest: str):
    r"""
    Safely parse image command argument to handle paths with spaces and quotes.
    Returns (source, question).
    """
    rest = rest.strip()
    if not rest:
        return "", ""

    if rest.startswith('"'):
        end_idx = rest.find('"', 1)
        if end_idx != -1:
            return rest[1:end_idx], rest[end_idx + 1:].strip()
    elif rest.startswith("'"):
        end_idx = rest.find("'", 1)
        if end_idx != -1:
            return rest[1:end_idx], rest[end_idx + 1:].strip()

    if rest.startswith("http://") or rest.startswith("https://"):
        parts = rest.split(" ", 1)
        return parts[0], parts[1].strip() if len(parts) > 1 else ""

    tokens = rest.split(" ")
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        norm = os.path.normpath(candidate)
        if os.path.isfile(norm):
            return norm, " ".join(tokens[end:]).strip()

    parts = rest.split(" ", 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def handle_image(source, question):
    source = _clean_path(source).strip("<>")
    if not source:
        console.print("  ❌ [bold red]Error:[/bold red] Missing image path or URL.\n")
        return

    is_url = source.startswith("http://") or source.startswith("https://")

    if is_url:
        stage("🌐 [Vision]", "Fetching remote image from URL...", style="bold cyan")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with allow_external():
                resp = _requests.get(source, headers=headers, timeout=15)
            resp.raise_for_status()
            image_b64 = base64.b64encode(resp.content).decode()
            label = source.split("/")[-1].split("?")[0] or "remote_image"
        except Exception as e:
            console.print(f"  ❌ [bold red]Could not fetch URL:[/bold red] {e}\n")
            return
    else:
        if not os.path.isfile(source):
            console.print(f"  ❌ [bold red]File not found:[/bold red] {source}\n")
            return
        ext = os.path.splitext(source)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            console.print(f"  ❌ [bold red]Unsupported format:[/bold red] '{ext}'. Use PNG, JPEG, WEBP, or GIF.\n")
            return
        with open(source, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        label = os.path.basename(source)

    if not question:
        question = "Describe this image in detail. Extract all visible text, data, and key information."

    stage("👁️ [Vision Analysis]", f'Image: [bold yellow]{label}[/bold yellow] | Prompt: "[cyan]{question[:60]}[/cyan]"')
    stage("🤖 [Model]", f"[bold cyan]{IMAGE_MODEL}[/bold cyan]")
    console.print("─" * 65, style="dim")

    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time), daemon=True)
    spinner.start()

    full_answer = ""
    token_count = 0
    spinner_done = False
    try:
        for token in stream_image_answer(image_b64, question):
            if not spinner_done:
                stop[0] = True
                spinner.join()
                spinner_done = True
            full_answer += token
            token_count += len(token.split())
            sys.stdout.write(token)
            sys.stdout.flush()
    finally:
        if not spinner_done:
            stop[0] = True
            spinner.join()

    print()
    elapsed = round(time.time() - start_time, 2)
    console.print("─" * 65, style="dim")
    console.print(f"  ✅ [bold green]Done in {elapsed}s[/bold green] | Model: [cyan]{IMAGE_MODEL}[/cyan] | ~[yellow]{token_count}[/yellow] words\n")


def handle_list_files():
    """Show all files in ~/Downloads/AgentOTG/."""
    from tools import list_generated_files
    result = list_generated_files()
    console.print(Panel(result, title="🗂️ Generated Files", border_style="green", padding=(0, 2)))
    console.print()


def is_file_path_arg(text: str) -> tuple[bool, str, str]:
    """Checks if text is or begins with a file path. Returns (is_file, filepath, user_prompt)."""
    filepath, prompt = parse_doc_command(text)
    if not filepath:
        return False, "", ""
    ext = os.path.splitext(filepath)[1].lower()
    valid_exts = (".csv", ".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".json")
    if os.path.isfile(filepath) or ext in valid_exts:
        return True, filepath, prompt
    return False, "", ""


def handle_rag(query: str):
    """Directly search the RAG knowledge base with query expansion."""
    if not query.strip():
        console.print("  [dim]Usage: /rag <your question>[/dim]\n")
        return

    is_file, filepath, prompt = is_file_path_arg(query)
    if is_file:
        console.print(f"  💡 [cyan]File path detected — ingesting into knowledge base first...[/cyan]")
        doc_path, user_p = handle_doc(query)
        if user_p:
            stage("🔍 [RAG Search]", f'Querying: "[cyan]{user_p[:70]}[/cyan]"', style="bold blue")
            console.print("─" * 65, style="dim")
            try:
                from rag.pipeline import answer as _rag_answer
                result = run_with_spinner(
                    ["Expanding query variants...", "Retrieving relevant chunks...", "Re-ranking results...", "Synthesizing answer..."],
                    lambda: _rag_answer(user_p),
                )
                console.print(Markdown(result.get("answer", "No answer found.")))
                _show_rag_sources(result)
            except Exception as exc:
                console.print(f"  ❌ [bold red]RAG search failed:[/bold red] {exc}\n")
        return

    stage("🔍 [RAG Search]", f'Query: "[cyan]{query[:70]}[/cyan]"', style="bold blue")
    console.print("─" * 65, style="dim")
    start = time.time()
    try:
        from rag.pipeline import answer as _rag_answer
        result = run_with_spinner(
            ["Expanding query variants...", "Retrieving relevant chunks...", "Re-ranking results...", "Synthesizing answer..."],
            lambda: _rag_answer(query),
        )
        ans      = result.get("answer", "No answer found.")
        sources  = result.get("sources", [])
        strategy = result.get("retrieval_strategy", "similarity")

        console.print(Markdown(ans))
        _show_rag_sources(result)

        elapsed = round(time.time() - start, 2)
        console.print("─" * 65, style="dim")
        console.print(
            f"  ✅ [bold green]RAG done in {elapsed}s[/bold green] "
            f"| Strategy: [cyan]{strategy}[/cyan] "
            f"| [yellow]{len(sources)}[/yellow] source(s)\n"
        )
    except Exception as exc:
        console.print(f"  ❌ [bold red]RAG search failed:[/bold red] {exc}\n")
        console.print("  [dim]Make sure documents are ingested with: /doc <file path>[/dim]\n")


def _show_rag_sources(result: dict):
    """Display RAG sources in a clean deduplicated format."""
    sources = result.get("sources", [])
    if not sources:
        return

    seen = set()
    unique_sources = []
    for s in sources:
        src  = s.get("source", "unknown")
        page = s.get("page")
        sheet = s.get("sheet")
        key = (src, page, sheet)
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    if unique_sources:
        console.print("\n  [dim]📚 Sources cited:[/dim]")
        for s in unique_sources[:6]:
            src   = s.get("source", "unknown")
            page  = f", page {s['page']}" if s.get("page") else ""
            sheet = f", sheet '{s['sheet']}'" if s.get("sheet") else ""
            console.print(f"  [dim]  • {src}{page}{sheet}[/dim]")


def parse_doc_command(rest: str) -> tuple[str, str]:
    r"""
    Robustly extract a file path and optional prompt from user input.
    Handles all Windows path edge cases:
      - Quoted paths:       "D:\Users\me\file.pdf" query here
      - Unquoted paths:     D:\Users\me\file.pdf query here
      - Paths with spaces:  D:\My Documents\file.csv
      - Forward slashes:    D:/Users/me/file.pdf
      - Dropped paths:      file.csv (just the filename)
      - Path with prompt:   report.pdf tell me the key findings
    Returns (normalized_filepath, user_prompt).
    """
    rest = rest.strip().strip("<>")   # strip drag-drop angle brackets
    if not rest:
        return "", ""

    # ── 1. Quoted path ─────────────────────────────────────────────────────
    for q in ('"', "'"):
        if rest.startswith(q):
            end_idx = rest.find(q, 1)
            if end_idx != -1:
                raw_path = rest[1:end_idx].strip()
                prompt   = rest[end_idx + 1:].strip()
                return os.path.normpath(raw_path), prompt

    # ── 2. Looks like a Windows absolute path (D:\... or D:/...) ───────────
    # Try consuming as many tokens as needed to form an existing file path.
    tokens = rest.split()

    # First check if the whole string (minus trailing prompt words) is a path
    # Strategy: try to greedily find the longest prefix that is an existing file.
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        # Normalise forward slashes too
        norm = os.path.normpath(candidate.replace("/", os.sep))
        if os.path.isfile(norm):
            prompt = " ".join(tokens[end:]).strip()
            return norm, prompt

    # ── 3. No existing file found — try heuristic: does the first token look
    #       like a path? (has extension, has path separator, or has drive letter)
    first = tokens[0]
    has_drive     = len(first) >= 2 and first[1] == ":"           # C: D: etc.
    has_sep       = ("/" in first or "\\" in first)
    has_extension = "." in os.path.basename(first)

    if has_drive or has_sep or has_extension:
        # Try consuming tokens until the path has a known extension
        supported_exts = {".pdf", ".docx", ".xlsx", ".xls", ".csv",
                          ".txt", ".md", ".json", ".png", ".jpg", ".jpeg", ".webp"}
        # Walk forward token by token — stop when we hit a word after a valid ext
        best_path = ""
        best_end  = 0
        for end in range(1, len(tokens) + 1):
            candidate = " ".join(tokens[:end])
            ext = os.path.splitext(candidate)[1].lower()
            if ext in supported_exts:
                best_path = candidate
                best_end  = end
        if best_path:
            norm   = os.path.normpath(best_path.replace("/", os.sep))
            prompt = " ".join(tokens[best_end:]).strip()
            return norm, prompt

        # Fall back — everything is the path
        norm = os.path.normpath(first.replace("/", os.sep))
        prompt = " ".join(tokens[1:]).strip()
        return norm, prompt

    # ── 4. No path-like pattern found ─────────────────────────────────────
    return "", rest


def handle_doc(raw_arg: str):
    """Ingest a file into the RAG vector knowledge base with pre-validation and verbose logging."""
    filepath, prompt = parse_doc_command(raw_arg)

    if not filepath:
        console.print("  ❌ [bold red]Ingestion failed:[/bold red] No file path provided.\n")
        return None, None

    try:
        from rag.loaders import validate_file_pre_ingestion
        validate_file_pre_ingestion(filepath)
    except Exception as val_err:
        err_msg = str(val_err)
        console.print(Panel(
            f"[bold red]❌ Ingestion Failed[/bold red]\n\n{err_msg}",
            border_style="red", padding=(0, 2),
        ))
        return None, None

    stage("📂 [RAG Ingest]", f"Indexing [bold yellow]{os.path.basename(filepath)}[/bold yellow] into knowledge base...", style="bold green")
    console.print("─" * 65, style="dim")
    start = time.time()
    try:
        from rag.pipeline import ingest_path as _ingest
        result = run_with_spinner(
            ["Validating file...", "Loading & parsing document...", "Splitting into chunks...", "Embedding and indexing..."],
            lambda: _ingest(filepath, replace_existing=True),
        )
        elapsed = round(time.time() - start, 2)

        loader_name = result.get("loader", "DocumentLoader")
        rows_loaded = result.get("rows_loaded", result.get("documents_loaded", 0))
        chunks_gen  = result.get("chunks_generated", result.get("chunks_indexed", 0))
        embed_count = result.get("embeddings_created", chunks_gen)

        # Rich ingestion log panel
        log_content = (
            f"[bold green]✅ Ingestion Successful[/bold green]  ({elapsed}s)\n\n"
            f"  [bold]File:[/bold]        {filepath}\n"
            f"  [bold]Loader:[/bold]      {loader_name}\n"
            f"  [bold]Rows/Docs:[/bold]   {rows_loaded}\n"
            f"  [bold]Chunks:[/bold]      {chunks_gen}\n"
            f"  [bold]Embeddings:[/bold]  {embed_count}\n"
            f"  [bold]Vector DB:[/bold]   SUCCESS\n\n"
            f"  [dim]You can now query with: What does the document say about ...[/dim]"
        )
        console.print(Panel(log_content, title="[bold green]📂 Document Ingestion Log[/bold green]", border_style="green", padding=(0, 2)))
        console.print("─" * 65, style="dim")

        return filepath, prompt
    except Exception as exc:
        err_str = str(exc)
        if hasattr(exc, "__cause__") and exc.__cause__:
            err_str = str(exc.__cause__)
        if err_str.startswith("RuntimeError:"):
            err_str = err_str.replace("RuntimeError:", "").strip()
        console.print(Panel(
            f"[bold red]❌ Ingestion Failed[/bold red]\n\n{err_str}\n\n"
            f"[dim]{traceback.format_exc().strip()[:400]}[/dim]",
            border_style="red", padding=(0, 2),
        ))
        return None, None


def handle_kb_stats():
    """Show RAG knowledge base statistics."""
    try:
        from rag.vectorstore import get_vector_store
        store = get_vector_store()
        count = store._collection.count()
        console.print(Panel(
            f"[bold cyan]📊 Knowledge Base Statistics[/bold cyan]\n\n"
            f"  Chunks indexed : [bold yellow]{count:,}[/bold yellow]\n"
            f"  Collection     : [dim]{store._collection.name}[/dim]\n\n"
            f"  [dim]Ingest more: /doc <file>  •  Query: ask naturally or /rag <question>[/dim]",
            title="📊 RAG Knowledge Base",
            border_style="cyan",
            padding=(0, 2),
        ))
    except Exception as exc:
        console.print(f"  ❌ [bold red]Could not read KB stats:[/bold red] {exc}\n")
    print()


def handle_show_models():
    """Display currently configured models."""
    table = Table(
        title="🤖 Active Model Configuration  (edit .env to change)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Role",         style="bold green",  no_wrap=True)
    table.add_column("Model Name",   style="bold yellow")
    table.add_column("Env Variable", style="dim")
    table.add_column("Handles",      style="dim white")
    table.add_row("Code",      _cfg.CODER_MODEL,         "CODER_MODEL",         "Programming, debugging, SQL")
    table.add_row("Main/Agent", _cfg.MAIN_MODEL,         "MAIN_MODEL",          "Complex reasoning, agent tasks")
    table.add_row("Fast",      _cfg.FAST_MODEL,          "FAST_MODEL",          "Quick Q&A, routing, RAG formatting")
    table.add_row("Vision",    _cfg.IMAGE_MODEL,         "IMAGE_MODEL",         "Image analysis, diagrams")
    table.add_row("Embedding", _cfg.RAG_EMBEDDING_MODEL, "RAG_EMBEDDING_MODEL", "Document vectorization")
    table.add_row("RAG LLM",   _cfg.RAG_LLM_MODEL,      "RAG_LLM_MODEL",       "Knowledge base answering")
    console.print(table)
    console.print(f"\n  [dim]Output directory: [yellow]{_OUTPUT_DIR}[/yellow][/dim]")
    console.print("  [dim]Restart ask.py after editing .env for changes to take effect.[/dim]\n")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def handle_history():
    """Reads sessions and messages from the database and displays them."""
    if not db.is_ready():
        console.print(Panel(
            f"[bold red]Database not available.[/bold red]\n[dim]{db.get_error()}[/dim]",
            title="Database Unavailable",
            border_style="red",
        ))
        return

    sessions = db.get_all_sessions(limit=15)
    if not sessions:
        console.print("  [dim]No sessions found in the database yet.[/dim]\n")
        return

    hist_table = Table(
        title=f"📜 Saved Chat Sessions ({db.get_session_count()} total)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    hist_table.add_column("#",              justify="right", style="bold cyan",  no_wrap=True)
    hist_table.add_column("Session Name",                    style="yellow")
    hist_table.add_column("Started",                         style="dim")
    hist_table.add_column("Last Activity",                   style="dim")
    hist_table.add_column("Messages",       justify="right", style="green")

    for i, s in enumerate(sessions, 1):
        hist_table.add_row(
            str(i),
            s["session_name"],
            s["created_at"],
            s["last_activity"],
            str(s["message_count"]),
        )
    console.print(hist_table)

    try:
        choice = input("  Pick a session # to preview (or Enter to cancel): ").strip()
        if not choice or not choice.isdigit():
            console.print()
            return

        idx = int(choice)
        if not (1 <= idx <= len(sessions)):
            console.print("  [dim]Invalid selection.[/dim]\n")
            return

        selected     = sessions[idx - 1]
        session_name = selected["session_name"]
        messages     = db.get_session_messages(session_name, limit=_HISTORY_MAX_MSGS)

        console.print(f"\n  [bold cyan]─── {session_name} ───[/bold cyan]")
        console.print(f"  [dim]Showing last {len(messages)} message(s)[/dim]\n")

        for msg in messages:
            role       = msg.get("role", "unknown")
            content    = msg.get("content", "").strip()
            timestamp  = msg.get("created_at", "")
            role_color = "bold green" if role == "user" else "bold magenta"
            role_label = "You" if role == "user" else "Agent OTG"

            if len(content) > _HISTORY_MSG_TRUNCATE:
                content = content[:_HISTORY_MSG_TRUNCATE] + f"… [{len(content) - _HISTORY_MSG_TRUNCATE} more chars]"

            console.print(f"  [{role_color}]{role_label}[/{role_color}] [dim]{timestamp}[/dim]")
            console.print(f"  {content}")
            console.print("  " + "─" * 55, style="dim")

    except Exception as e:
        console.print(f"  [dim]Error viewing session: {e}[/dim]")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Initialise database ─────────────────────────────────────────────────
    db_ok = db.init_db()
    if db_ok:
        stage("🗄️  [Database]", "Session store connected and ready.", style="bold green")
    else:
        stage("⚠️  [Database]", f"Session store unavailable — sessions will NOT be saved. ({db.get_error()[:80]})", style="bold red")

    # ── Session name ────────────────────────────────────────────────────────
    session_name = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    router.start_new_session(session_name)

    print_welcome_banner()

    current_file_path    = None
    current_file_content = None

    while True:
        try:
            # Styled prompt
            console.print("💬 [bold cyan]You[/bold cyan] [bold green]❯[/bold green] ", end="")
            query = input().strip()

            if not query:
                continue

            # ── Exit ──────────────────────────────────────────────────────
            if query.lower() in ["exit", "quit", ":q"]:
                console.print("\n[bold cyan]👋 Thank you for using Agent OTG. Goodbye! — DWE Team[/bold cyan]\n")
                break

            # ── Help ──────────────────────────────────────────────────────
            if query.lower() in ["help", "/help", "?", "--help"]:
                print_help_guide()
                continue

            # ── Clear screen ──────────────────────────────────────────────
            if query.lower() in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome_banner()
                continue

            # ── Reset memory ──────────────────────────────────────────────
            if query.lower() in ["reset", "/reset"]:
                clear_history()
                current_file_path    = None
                current_file_content = None
                console.print(Panel(
                    "✅ Conversation memory, file attachment, and active context cleared.",
                    border_style="green", padding=(0, 2),
                ))
                print()
                continue

            # ── Handle /ask prefix ────────────────────────────────────────
            if query.startswith("/ask "):
                query = query[5:].strip()
            elif query.lower() == "/ask":
                console.print("  [dim]Usage: /ask <your question>[/dim]\n")
                continue

            # ── Inject active file content when referenced ─────────────────
            trigger_phrases = [
                "this file", "this document", "this sheet", "this pdf",
                "the uploaded file", "the file", "uploaded",
                "analyze this", "summarize this", "from this", "in this file",
                "in this document", "the data in", "this data", "the data",
                "the dataset", "about this data", "from the data", "analyze this data",
                "summarize this data", "explain this data",
            ]
            has_data_ref = any(phrase in query.lower() for phrase in trigger_phrases)

            if current_file_path and current_file_content and has_data_ref:
                query += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End of file content ---"
                stage("📎 [Attachment]", f"Attached: [yellow]{os.path.basename(current_file_path)}[/yellow]", style="dim")
            elif has_data_ref and not current_file_content:
                try:
                    from rag.vectorstore import get_vector_store
                    store = get_vector_store()
                    if store._collection.count() > 0:
                        stage("🔍 [RAG Context]", "Active file not in RAM — retrieving from knowledge base...", style="dim cyan")
                        from rag.pipeline import answer as _rag_answer
                        rag_res = _rag_answer(query)
                        rag_ans = rag_res.get("answer", "")
                        if rag_ans and "couldn't find sufficient information" not in rag_ans.lower():
                            query += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_ans}\n--- End of context ---"
                except Exception:
                    pass

            # ══════════════════════════════════════════════════════════════
            # COMMAND ROUTING
            # ══════════════════════════════════════════════════════════════

            # ── Utility commands ──────────────────────────────────────────
            if query.lower() in ["/files", "files", "/ls", "ls"]:
                handle_list_files()
                continue

            if query.lower() in ["history", "/history"]:
                handle_history()
                continue

            if query.lower() in ["/kb", "/kb-stats", "/kbstats"]:
                handle_kb_stats()
                continue

            if query.lower() in ["/models", "/config", "/env"]:
                handle_show_models()
                continue

            if query.lower() == "/cancel":
                current_file_path    = None
                current_file_content = None
                console.print(Panel("🚫 File attachment and active context cleared.", border_style="yellow", padding=(0, 2)))
                print()
                continue

            # ── Upload (legacy) ───────────────────────────────────────────
            if query.lower().startswith("upload "):
                raw_path = query[7:].strip()
                path     = _clean_path(raw_path)
                from file_readers import process_file
                result = process_file(path)
                if result.startswith("Error:"):
                    console.print(f"  ❌ [bold red]{result}[/bold red]\n")
                else:
                    current_file_path    = path
                    current_file_content = result
                    char_count = len(result)
                    stage("📄 [Uploaded]", f"[bold yellow]{os.path.basename(path)}[/bold yellow] ({char_count:,} chars)", style="bold green")
                    console.print("  [dim]Ask questions about it naturally (e.g. 'analyze this data')[/dim]\n")
                continue

            # ── Explicit /rag command ──────────────────────────────────────
            if query.startswith("/rag ") or query.lower() == "/rag":
                rest = query[5:].strip() if query.startswith("/rag ") else ""
                if not rest:
                    console.print("  [dim]Usage: /rag <question>  or  /rag <file path> [optional prompt][/dim]\n")
                else:
                    is_file, filepath, user_prompt = is_file_path_arg(rest)
                    if is_file:
                        doc_path, prompt = handle_doc(rest)
                        if doc_path:
                            current_file_path = doc_path
                            try:
                                from file_readers import process_file
                                res = process_file(doc_path)
                                if res and not res.startswith("Error:"):
                                    current_file_content = res
                            except Exception:
                                pass
                            if prompt:
                                console.print(f"  💬 [bold cyan]Executing prompt:[/bold cyan] [yellow]{prompt}[/yellow]\n")
                                prompt_to_run = prompt
                                if current_file_path and current_file_content:
                                    prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                                ask_anything(prompt_to_run)
                            else:
                                console.print("  ✅ [bold green]File indexed into RAG KB![/bold green]\n  [dim]Query naturally or use /rag <question>[/dim]\n")
                    else:
                        handle_rag(rest)
                continue

            # ── Explicit /doc command ──────────────────────────────────────
            elif query.startswith("/doc ") or query.lower() == "/doc":
                rest = query[5:].strip() if query.startswith("/doc ") else ""
                if not rest:
                    console.print("  [dim]Usage: /doc <file path> [optional prompt][/dim]\n")
                else:
                    doc_path, user_prompt = handle_doc(rest)
                    if doc_path:
                        current_file_path = doc_path
                        try:
                            from file_readers import process_file
                            res = process_file(doc_path)
                            if res and not res.startswith("Error:"):
                                current_file_content = res
                        except Exception:
                            pass
                        if user_prompt:
                            console.print(f"  💬 [bold cyan]Executing prompt:[/bold cyan] [yellow]{user_prompt}[/yellow]\n")
                            prompt_to_run = user_prompt
                            if current_file_path and current_file_content:
                                prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                            ask_anything(prompt_to_run)
                        else:
                            console.print("  ✅ [bold green]File indexed into RAG KB![/bold green]\n  [dim]Query naturally or use /rag <question>[/dim]\n")
                continue

            # ── Image commands ─────────────────────────────────────────────
            elif query.startswith("/image ") or query.startswith("/img "):
                prefix_len = 7 if query.startswith("/image ") else 5
                rest = query[prefix_len:].strip()
                img_path, q_text = parse_image_command(rest)
                handle_image(img_path, q_text)
                continue

            # ── Explicit direct Qwen 14B ───────────────────────────────────
            elif router.is_complex_command(query):
                if not router.strip_complex_command(query):
                    console.print("  [dim]Usage: /complex <deep reasoning request>[/dim]\n")
                else:
                    ask_anything(query)
                continue

            # ── Explicit /agent command ────────────────────────────────────
            elif query.startswith("/agent ") or query.lower() == "/agent":
                agent_query = query[len("/agent "):].strip() if query.startswith("/agent ") else ""
                if not agent_query:
                    console.print("  [dim]Usage: /agent <instruction>  (e.g. /agent Create a PDF report on AI)[/dim]\n")
                else:
                    handle_agent(agent_query)
                continue

            else:
                # ════════════════════════════════════════════════════════
                # SMART DISPATCH — No command prefix required
                # ════════════════════════════════════════════════════════

                # 1. Check if raw file path was pasted
                is_file, filepath, user_prompt = is_file_path_arg(query)
                if is_file and (os.path.isfile(filepath) or "." in os.path.basename(filepath)):
                    doc_path, prompt = handle_doc(query)
                    if doc_path:
                        current_file_path = doc_path
                        try:
                            from file_readers import process_file
                            res = process_file(doc_path)
                            if res and not res.startswith("Error:"):
                                current_file_content = res
                        except Exception:
                            pass
                        if prompt:
                            console.print(f"  💬 [bold cyan]Executing prompt:[/bold cyan] [yellow]{prompt}[/yellow]\n")
                            prompt_to_run = prompt
                            if current_file_path and current_file_content:
                                prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                            ask_anything(prompt_to_run)
                        else:
                            console.print("  ✅ [bold green]File indexed into RAG KB![/bold green]\n  [dim]Query naturally or use /rag <question>[/dim]\n")
                    continue

                # 2. Smart intent detection (RAG / agent auto-routing)
                if _cfg.SMART_ROUTING_ENABLED:
                    intent = _detect_intent(query)
                    if intent == "rag":
                        _show_intent_badge("rag", query)
                        handle_rag(query)
                        continue
                    elif intent == "agent":
                        _show_intent_badge("agent", query)
                        handle_agent(query)
                        continue
                    elif intent == "image_gen":
                        _show_intent_badge("image_gen", query)
                        handle_image_generation(query)
                        continue

                # 3. Standard classified routing
                ask_anything(query)

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]👋 Session paused. Type exit or Ctrl+C again to quit.[/bold cyan]\n")
            continue
        except Exception as exc:
            console.print(Panel(
                f"[bold red]Unexpected Error[/bold red]\n\n{exc}\n\n[dim]{traceback.format_exc().strip()[:500]}[/dim]",
                border_style="red", padding=(0, 2),
            ))


if __name__ == "__main__":
    main()
