"""
ask.py - The Interactive Terminal Client for Agent OTG

A high-performance, multi-model AI assistant running locally and fully offline.
Features:
    - Automatic Routing: routes coding, reasoning, casual chit-chat, and multi-part questions
    - Code + Walkthrough: ordered sequential pipeline for code generation followed by explanation
    - Vision: inspect local images or web URLs using qwen2.5vl:7b
    - Document Ingestion: upload and query Excel (.xlsx), Word (.docx), and PDF (.pdf) files
    - LangGraph Autonomous Agent: multi-step structured agent with multi-tool execution
    - Interactive Chat History & Context Memory Management
"""

import sys
import time
import base64
import os
import threading
import datetime
import json
import requests as _requests

from offline_guard import enable_offline_mode, allow_external
enable_offline_mode()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich import box

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

# ── Status phrases for spinner ────────────────────────────────────────────────
THINKING_PHRASES = [
    "🧠 Analyzing neural context...",
    "⚡ Routing to optimal model...",
    "💭 Composing precise response...",
    "✨ Synthesizing output...",
]

# ── Max characters to show per message in history view ───────────────────────
_HISTORY_MSG_TRUNCATE = 200
_HISTORY_MAX_MSGS     = 20


def _clean_path(path: str) -> str:
    """Clean surrounding quotes and whitespace from paths."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


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
        sys.stdout.write(f"\r  \033[96m{frames[i % len(frames)]}\033[0m \033[93m{messages[msg_index]}\033[0m \033[90m({elapsed:.1f}s)\033[0m ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.08)
    sys.stdout.write("\r" + " " * 90 + "\r")
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
    """
    Shows a spinner until the first token arrives, then streams tokens live.
    Tool-call notification lines are highlighted; all other tokens print normally.
    """
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

            # Tool notification lines — print with colour, not inline with answer text
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
# INTERACTIVE WELCOME & HELP
# ══════════════════════════════════════════════════════════════════════════════

def _check_ollama_models() -> dict:
    """Check which configured models are available in Ollama."""
    available = {}
    try:
        import ollama as _ollama
        with allow_external():
            pulled = {m["name"] for m in _ollama.list()["models"]}
        # Normalize: strip digest tags for comparison
        pulled_base = {n.split(":")[0] for n in pulled} | pulled
        for role, model in AVAILABLE_MODELS.items():
            base = model.split(":")[0]
            available[model] = (model in pulled) or (base in pulled_base)
    except Exception:
        # If we can't check, assume all are available
        for model in AVAILABLE_MODELS.values():
            available[model] = True
    return available


def print_welcome_banner():
    # ── DWE Team Credits (like OpenAI's "ChatGPT is made by OpenAI") ──────────
    import config as _cfg
    credits_text = (
        f"[bold white]{_cfg.APP_NAME}[/bold white]  [dim]v{_cfg.APP_VERSION}[/dim]\n\n"
        f"[dim]Crafted with ❤️  by the [/dim][bold bright_magenta]{_cfg.APP_TEAM}[/bold bright_magenta]\n"
        "[dim]Sanchit  •  DWE Member 2  •  DWE Member 3[/dim]\n\n"
        "[dim]🔒 100% On-Premise  •  Air-Gapped  •  No cloud  •  No tracking[/dim]\n"
        "[dim]Your data never leaves this machine.[/dim]"
    )
    console.print(Panel(
        credits_text,
        box=box.DOUBLE,
        border_style="bright_magenta",
        padding=(1, 4),
        title="[bold bright_magenta]★  Agent OTG  ★[/bold bright_magenta]",
        subtitle="[dim]Made by DWE Team[/dim]",
    ))

    banner_text = (
        "[bold cyan]🤖 AGENT OTG — Local Autonomous Multi-Model AI System[/bold cyan]\n"
        "[dim]100% On-Premise • Air-Gapped / Private • RAG Knowledge Base • LangGraph Tools[/dim]"
    )
    console.print(Panel(banner_text, box=box.ROUNDED, border_style="cyan", padding=(1, 2)))

    table = Table(
        title="🚀 Capabilities & Shortcut Command Guide",
        box=box.SIMPLE_HEAVY,
        border_style="bright_blue",
    )
    table.add_column("Capability",       style="bold green", no_wrap=True)
    table.add_column("Shortcut / Trigger", style="yellow")
    table.add_column("Model",            style="magenta")
    table.add_column("Description",      style="white")

    table.add_row("💻 Code Generation",    "Write a Python script for...",         CODER_MODEL,                    "Auto-routes programming, bugs, scripts, SQL")
    table.add_row("🔄 Code + Walkthrough", "Write X and explain how it works",      f"{CODER_MODEL} ➔ {MAIN_MODEL}",  "Sequential 2-step pipeline")
    table.add_row("🧠 Deep Reasoning",     "Explain quantum computing...",          MAIN_MODEL,                     "Essays, logic, architecture, analysis")
    table.add_row("⚡ Quick Chat",          "Hello / What is 25 * 4?",               FAST_MODEL,                     "Instant responses, small-talk")
    table.add_row("🔍 RAG Search",         "/rag <question>",                       MAIN_MODEL,                     "Search knowledge base with query expansion")
    table.add_row("📂 Ingest to RAG KB",   "/doc <file path>",                     "Vector Store",                 "Index PDF/DOCX/CSV/Excel into knowledge base")
    table.add_row("📊 KB Stats",           "/kb",                                  "Vector Store",                 "Show indexed chunk count & collection info")
    table.add_row("🗂️ Multi-Task",          "/complex <query>",                     "Dynamic Multi-Model",          "Split into independent parallel sub-tasks")
    table.add_row("👁️ Vision / Image",      "/image <path|url> [q]  or  /img",     IMAGE_MODEL,                    "Inspect diagrams, photos, screenshots")
    table.add_row("📄 Upload (legacy)",     "upload <file path>",                  "Context Injector",             "Attach file content directly to prompt")
    table.add_row("🛠️ LangGraph Agent",     "/agent <instruction>",                "qwen2.5:14b + Tools",          "Autonomous multi-tool agent (PDF, Word, CSV…)")
    table.add_row("🤖 Show Models",        "/models",                              "Config (.env)",                "Display models currently set in .env")
    table.add_row("🚫 Cancel Context",     "/cancel",                              "Memory",                       "Clear active file attachment from prompt")
    table.add_row("📜 Session History",    "history",                              "PostgreSQL",                   "Browse saved chat sessions from DB")
    table.add_row("🗑️ List Output Files",  "/files",                               "File Manager",                 "List all generated files (PDFs, Word, CSVs)")
    table.add_row("🧹 Reset Memory",       "reset  or  /reset",                    "Memory Guard",                 "Clear conversation context from RAM")
    table.add_row("❓ Help",               "help  or  /help",                      "Guide",                        "Re-display this command matrix")

    console.print(table)
    console.print("[dim]Type your question or use any shortcut above. Type [bold red]exit[/bold red] to quit.[/dim]\n")


def print_help_guide():
    console.print("\n" + "=" * 70, style="bright_blue")
    console.print("📖 [bold cyan]Agent OTG — Practical Example Guide[/bold cyan]", style="bold")
    console.print("=" * 70, style="bright_blue")

    examples = [
        ("💻 Coding", 'Write a Python FastAPI middleware for request timing and explain it.'),
        ("🧠 Reasoning", 'Compare PostgreSQL vs MongoDB for high-write telemetry systems.'),
        ("👁️ Vision", '/image "C:\\Users\\me\\Pictures\\diagram.png" What architecture is shown here?'),
        ("👁️ Vision (Web)", '/image https://example.com/logo.png Describe this logo in detail.'),
        ("📄 Excel File", '1) upload "C:\\data\\sales.xlsx"\n   2) What is the total revenue in this sheet?'),
        ("📄 PDF File", '1) upload "C:\\docs\\manual.pdf"\n   2) Summarize the safety guidelines.'),
        ("🛠️ Agent — Word", '/agent Create a Word document titled "Q3 Report" with 3 bullet highlights about AI trends.'),
        ("🛠️ Agent — PDF", '/agent Generate a PDF titled "Meeting Notes" with agenda items for a project kickoff.'),
        ("🛠️ Agent — CSV", '/agent Create a CSV file with columns Name, Score, Grade and 5 sample student rows.'),
        ("🛠️ Agent — Extract", '/agent Extract all text from "generated_files/sample.pdf" and give me a summary.'),
        ("🛠️ Agent — Chain", '/agent Generate a PDF about Python basics, then tell me how many pages it has.'),
        ("🗂️ Complex Split", '/complex Write a marketing email for a product launch AND generate SQL to find top buyers.'),
        ("🧹 Reset", 'reset (clears conversation context so you can start a fresh topic).'),
        ("🗑️ List Files", '/files (shows all files created in generated_files/ directory)'),
    ]

    for cat, ex in examples:
        console.print(f"\n[bold green]{cat}:[/bold green]")
        console.print(f"  [yellow]{ex}[/yellow]")

    console.print("\n" + "=" * 70 + "\n", style="bright_blue")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def handle_single(query, info):
    model = info["model"]
    stage("🧠 [Route Decision]", f'Category: [bold magenta]"{info["category"]}"[/bold magenta] ➔ Model: [bold cyan]{model}[/bold cyan]')
    stage("💡 [Reason]", f'{info["reason"]}', style="dim")
    console.print("-" * 65, style="dim")

    start_time = time.time()
    token_count = stream_with_spinner(model, query)
    elapsed = round(time.time() - start_time, 2)

    console.print("-" * 65, style="dim")
    console.print(f"  ✅ [bold green]Done in {elapsed}s[/bold green] | Model: [cyan]{model}[/cyan] | ~[yellow]{token_count}[/yellow] words/tokens\n")


def handle_sequential(query):
    """
    Handles 'Code + Explain' queries in two ordered, chained steps:
      Step 1: Coder model generates code.
      Step 2: Smart 14b model explains that exact code from shared history.
    """
    stage("🔄 [Pipeline]", "Code + Explanation Combo Detected! Running 2-Step Pipeline...", style="bold magenta")
    console.print("-" * 65, style="dim")

    start_time = time.time()

    # Step 1: Code Generation
    stage("💻 [Step 1/2]", f"Generating code with [bold cyan]{CODER_MODEL}[/bold cyan]...", style="bold green")
    code_prompt = (
        f"The user asked: \"{query}\"\n\n"
        "Your job for this step: write ONLY the code. "
        "Do not explain it yet — just provide clean, well-commented code."
    )
    tokens_code = stream_with_spinner(CODER_MODEL, code_prompt)
    console.print()

    # Step 2: Explanation
    stage("🧠 [Step 2/2]", f"Writing step-by-step walkthrough with [bold cyan]{MAIN_MODEL}[/bold cyan]...", style="bold green")
    explain_prompt = (
        "Now explain the code you just wrote above, step by step. "
        "Be clear and beginner-friendly. Cover what each part does and why."
    )
    tokens_explain = stream_with_spinner(MAIN_MODEL, explain_prompt)

    elapsed = round(time.time() - start_time, 2)
    console.print("-" * 65, style="dim")
    console.print(
        f"  ✅ [bold green]Sequential Pipeline Completed in {elapsed}s[/bold green] | "
        f"Models: [cyan]{CODER_MODEL}[/cyan] & [cyan]{MAIN_MODEL}[/cyan] | "
        f"~[yellow]{tokens_code + tokens_explain}[/yellow] total tokens\n"
    )


def handle_multi(query):
    stage("🗂️ [Planning]", "Multi-part request detected. Decomposing into independent sub-tasks...", style="bold yellow")
    tasks = run_with_spinner(["Analyzing and structuring sub-tasks..."], lambda: break_into_tasks(query))

    table = Table(title=f"📋 Execution Plan ({len(tasks)} Sub-Tasks)", box=box.ROUNDED, border_style="yellow")
    table.add_column("#", justify="right", style="bold cyan")
    table.add_column("Sub-Task Title", style="bold white")
    table.add_column("Assigned Model", style="bold magenta")
    table.add_column("Category", style="yellow")

    for i, task in enumerate(tasks, 1):
        table.add_row(str(i), task["label"], task["model"], task["category"])
    console.print(table)
    console.print()

    start_time = time.time()
    for i, task in enumerate(tasks, 1):
        stage(f"⚙️ [Executing {i}/{len(tasks)}]", f"{task['label']} via [bold cyan]{task['model']}[/bold cyan]")
        console.print("-" * 65, style="dim")
        stream_with_spinner(task["model"], task["task"])
        console.print("-" * 65 + "\n", style="dim")

    elapsed = round(time.time() - start_time, 2)
    console.print(f"  ✅ [bold green]All {len(tasks)} sub-tasks completed in {elapsed}s[/bold green]\n")


def ask_anything(query, force_multi=False):
    stage("🧠 [Understanding]", "Analyzing question structure and context...", style="bold blue")
    info = run_with_spinner(["Evaluating intent and routing logic..."], lambda: classify_question(query))

    if force_multi:
        handle_multi(query)
    elif info.get("has_code_and_explain"):
        handle_sequential(query)
    elif info.get("is_multi_part"):
        handle_multi(query)
    else:
        handle_single(query, info)


def handle_agent(query):
    stage("🛠️ [LangGraph Agent]", "Initializing multi-tool reasoning graph (classify ➔ tools? ➔ classify ➔ ... ➔ respond)...", style="bold magenta")
    stage("🤖 [Model]", "qwen2.5:14b with dynamic tool calling (up to 5 tools per request)", style="cyan")
    console.print("-" * 65, style="dim")

    start_time = time.time()
    try:
        from langgraph_agent import run_agent
    except ImportError as e:
        console.print(f"  ❌ [bold red]LangGraph not available:[/bold red] {e}")
        console.print("  [dim]Install requirements via: py -m pip install langgraph langchain-ollama[/dim]")
        return

    result = run_with_spinner(
        ["Analyzing intent...", "Invoking registered tools...", "Checking if more tools needed...", "Synthesizing comprehensive response..."],
        lambda: run_agent(query, history=get_history()),
    )

    elapsed = round(time.time() - start_time, 2)

    if result.get("needs_tool") and result.get("tool_log"):
        tool_table = Table(title="🛠️ Tools Executed by Agent", box=box.ROUNDED, border_style="green")
        tool_table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
        tool_table.add_column("Tool", style="bold yellow")
        tool_table.add_column("Args", style="dim")
        tool_table.add_column("Result", style="white")
        for i, entry in enumerate(result["tool_log"], 1):
            args_str = ", ".join(f"{k}={repr(str(v))[:35]}" for k, v in entry.get("args", {}).items())
            result_str = str(entry.get("result", ""))[:150]
            tool_table.add_row(str(i), entry["tool"], args_str, result_str)
        console.print(tool_table)
        console.print()

    final_answer = result.get("final_answer", "").strip()
    if not final_answer:
        final_answer = "Agent completed the task. Check the generated_files/ directory for any output files."

    # Render with Markdown so code blocks, headings, bold text all display correctly
    console.print(Markdown(final_answer))
    console.print("-" * 65, style="dim")

    tool_count = len(result.get("tool_log", []))
    tool_info  = f" | [green]{tool_count} tool(s) called[/green]" if tool_count else ""
    console.print(f"  ✅ [bold green]Done in {elapsed}s[/bold green]{tool_info} | Autonomous LangGraph Agent | [cyan]qwen2.5:14b[/cyan]\n")


def parse_image_command(rest: str):
    r"""
    Safely parses image command argument to handle paths with spaces and quotes:
      /image "C:\path with spaces\test.png" What is this?
      /image https://example.com/pic.jpg Describe this
      /image ./test.png
    """
    rest = rest.strip()
    if not rest:
        return "", ""

    # Check for quotes around the path
    if rest.startswith('"'):
        end_idx = rest.find('"', 1)
        if end_idx != -1:
            source   = rest[1:end_idx]
            question = rest[end_idx + 1:].strip()
            return source, question
    elif rest.startswith("'"):
        end_idx = rest.find("'", 1)
        if end_idx != -1:
            source   = rest[1:end_idx]
            question = rest[end_idx + 1:].strip()
            return source, question

    # Check for URL — split at first whitespace after the URL
    if rest.startswith("http://") or rest.startswith("https://"):
        parts    = rest.split(" ", 1)
        source   = parts[0]
        question = parts[1].strip() if len(parts) > 1 else ""
        return source, question

    # For local paths: find the last valid path-like prefix by checking existence
    # Try progressively longer tokens until the path exists, then use the rest as the question
    tokens = rest.split(" ")
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        norm = os.path.normpath(candidate)
        if os.path.isfile(norm):
            question = " ".join(tokens[end:]).strip()
            return norm, question

    # Last resort: split on first space
    parts    = rest.split(" ", 1)
    source   = parts[0]
    question = parts[1].strip() if len(parts) > 1 else ""
    return source, question


def handle_image(source, question):
    source = _clean_path(source).strip("<>")
    if not source:
        console.print("  ❌ [bold red]Error:[/bold red] Missing image path or URL.")
        return

    is_url = source.startswith("http://") or source.startswith("https://")

    # Fetch from URL
    if is_url:
        stage("🌐 [Vision]", f"Fetching remote image from URL...", style="bold cyan")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            with allow_external():
                resp = _requests.get(source, headers=headers, timeout=15)
            resp.raise_for_status()
            image_b64 = base64.b64encode(resp.content).decode()
            label = source.split("/")[-1].split("?")[0] or "remote_image"
        except Exception as e:
            console.print(f"  ❌ [bold red]Could not fetch URL:[/bold red] {e}\n")
            return

    # Read from local disk
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
        question = "Describe this image in detail."

    stage("👁️ [Vision Analysis]", f'Target: [bold yellow]{label}[/bold yellow] | Prompt: "[cyan]{question}[/cyan]"')
    stage("🤖 [Model]", f"Routing to vision model: [bold cyan]{IMAGE_MODEL}[/bold cyan]")
    console.print("-" * 65, style="dim")

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
    console.print("-" * 65, style="dim")
    console.print(f"  ✅ [bold green]Done in {elapsed}s[/bold green] | Model: [cyan]{IMAGE_MODEL}[/cyan] | ~[yellow]{token_count}[/yellow] tokens\n")


def handle_list_files():
    """Show all files in the generated_files/ directory."""
    from tools import list_generated_files
    result = list_generated_files()
    console.print(Panel(result, title="🗂️ Generated Files", border_style="green", padding=(0, 1)))
    console.print()


def handle_rag(query: str):
    """Directly search the RAG knowledge base with query expansion."""
    if not query.strip():
        console.print("  [dim]Usage: /rag <your question>[/dim]\n")
        return
    stage("🔍 [RAG Search]", f'Knowledge base query: "[cyan]{query[:70]}[/cyan]"', style="bold blue")
    console.print("-" * 65, style="dim")
    start = time.time()
    try:
        from rag.pipeline import answer as _rag_answer
        result = run_with_spinner(
            ["Expanding query variants...", "Retrieving relevant chunks...", "Synthesizing answer..."],
            lambda: _rag_answer(query),
        )
        ans      = result.get("answer", "No answer found.")
        sources  = result.get("sources", [])
        strategy = result.get("retrieval_strategy", "similarity")

        console.print(Markdown(ans))

        if sources:
            console.print("\n  [dim]Sources cited:[/dim]")
            for s in sources[:6]:
                src  = s.get("source", "unknown")
                page = f", page {s['page']}" if s.get("page") else ""
                console.print(f"  [dim]  • {src}{page}[/dim]")

        elapsed = round(time.time() - start, 2)
        console.print("-" * 65, style="dim")
        console.print(
            f"  ✅ [bold green]RAG done in {elapsed}s[/bold green] "
            f"| Strategy: [cyan]{strategy}[/cyan] "
            f"| [yellow]{len(sources)}[/yellow] source(s)\n"
        )
    except Exception as exc:
        console.print(f"  ❌ [bold red]RAG search failed:[/bold red] {exc}\n")


def handle_doc(path: str):
    """Ingest a file into the RAG vector knowledge base."""
    path = _clean_path(path)
    if not path or not os.path.isfile(path):
        console.print(f"  ❌ [bold red]File not found:[/bold red] {path or '(no path given)'}\n")
        return
    stage("📂 [RAG Ingest]", f"Indexing [bold yellow]{os.path.basename(path)}[/bold yellow] into knowledge base...", style="bold green")
    console.print("-" * 65, style="dim")
    start = time.time()
    try:
        from rag.pipeline import ingest_path as _ingest
        result  = run_with_spinner(
            ["Loading file...", "Splitting into chunks...", "Embedding and indexing..."],
            lambda: _ingest(path, replace_existing=True),
        )
        elapsed = round(time.time() - start, 2)
        chunks  = result.get("chunks_indexed", 0)
        docs    = result.get("documents_loaded", 0)
        console.print("-" * 65, style="dim")
        stage(
            "✅ [Ingested]",
            f"[bold green]{os.path.basename(path)}[/bold green] → "
            f"[yellow]{docs}[/yellow] doc(s), [yellow]{chunks}[/yellow] chunk(s) in {elapsed}s",
            style="bold green",
        )
        console.print("  [dim]Now ask: /rag <question about this document>[/dim]\n")
    except Exception as exc:
        console.print(f"  ❌ [bold red]Ingestion failed:[/bold red] {exc}\n")


def handle_kb_stats():
    """Show RAG knowledge base statistics."""
    try:
        from rag.vectorstore import get_vector_store
        store = get_vector_store()
        count = store._collection.count()
        console.print(Panel(
            f"[bold cyan]Knowledge Base Statistics[/bold cyan]\n\n"
            f"  Chunks indexed : [bold yellow]{count}[/bold yellow]\n"
            f"  Collection     : [dim]{store._collection.name}[/dim]\n\n"
            f"  [dim]Use /doc <file> to add more  •  /rag <question> to search[/dim]",
            title="📊 RAG Knowledge Base",
            border_style="cyan",
            padding=(0, 2),
        ))
    except Exception as exc:
        console.print(f"  ❌ [bold red]Could not read KB stats:[/bold red] {exc}\n")
    print()


def handle_show_models():
    """Display currently configured models loaded from .env."""
    import config as _cfg
    table = Table(title="🤖 Active Model Configuration  (edit .env to change)", box=box.ROUNDED, border_style="cyan")
    table.add_column("Role",         style="bold green",  no_wrap=True)
    table.add_column("Model Name",   style="bold yellow")
    table.add_column("Env Variable", style="dim")
    table.add_row("Code",      _cfg.CODER_MODEL,         "CODER_MODEL")
    table.add_row("Main/RAG",  _cfg.MAIN_MODEL,          "MAIN_MODEL")
    table.add_row("Fast",      _cfg.FAST_MODEL,          "FAST_MODEL")
    table.add_row("Vision",    _cfg.IMAGE_MODEL,         "IMAGE_MODEL")
    table.add_row("Embedding", _cfg.RAG_EMBEDDING_MODEL, "RAG_EMBEDDING_MODEL")
    table.add_row("RAG LLM",   _cfg.RAG_LLM_MODEL,      "RAG_LLM_MODEL")
    console.print(table)
    console.print("[dim]  Restart ask.py after editing .env for changes to take effect.[/dim]\n")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def handle_history():
    """Reads sessions and messages from PostgreSQL and displays them."""
    if not db.is_ready():
        console.print(Panel(
            f"[bold red]PostgreSQL is not available.[/bold red]\n[dim]{db.get_error()}[/dim]\n\n"
            "Check your credentials in [yellow]db_config.json[/yellow] and ensure PostgreSQL is running.",
            title="Database Unavailable",
            border_style="red",
        ))
        return

    sessions = db.get_all_sessions(limit=15)
    if not sessions:
        console.print("  [dim]No sessions found in the database yet.[/dim]\n")
        return

    hist_table = Table(
        title=f"Saved Chat Sessions ({db.get_session_count()} total in DB)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    hist_table.add_column("#",            justify="right",  style="bold cyan",    no_wrap=True)
    hist_table.add_column("Session Name",                   style="yellow")
    hist_table.add_column("Started",                        style="dim")
    hist_table.add_column("Last Activity",                  style="dim")
    hist_table.add_column("Messages",     justify="right",  style="green")

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

        selected    = sessions[idx - 1]
        session_name = selected["session_name"]
        messages    = db.get_session_messages(session_name, limit=_HISTORY_MAX_MSGS)

        console.print(f"\n  [bold cyan]--- {session_name} ---[/bold cyan]")
        console.print(f"  [dim]Showing last {len(messages)} message(s)[/dim]\n")

        for msg in messages:
            role       = msg.get("role", "unknown")
            content    = msg.get("content", "").strip()
            timestamp  = msg.get("created_at", "")
            role_color = "bold green" if role == "user" else "bold magenta"
            role_label = "You" if role == "user" else "Agent"

            # Truncate long messages
            if len(content) > _HISTORY_MSG_TRUNCATE:
                content = content[:_HISTORY_MSG_TRUNCATE] + f"... [{len(content) - _HISTORY_MSG_TRUNCATE} more chars]"

            console.print(f"  [{role_color}]{role_label}[/{role_color}] [dim]{timestamp}[/dim]")
            console.print(f"  {content}")
            console.print("  " + "-" * 55, style="dim")

    except Exception as e:
        console.print(f"  [dim]Error viewing session: {e}[/dim]")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Sessions stored only in PostgreSQL — no local JSON or chat_sessions/ folder needed

    # ── Initialise PostgreSQL ────────────────────────────────────────────────
    db_ok = db.init_db()
    if db_ok:
        stage("🗄️  [Database]", "PostgreSQL session store connected and ready.", style="bold green")
    else:
        stage("⚠️  [Database]", f"PostgreSQL unavailable — sessions will NOT be saved. ({db.get_error()[:80]})", style="bold red")
        console.print("  [dim]Check db_config.json and ensure PostgreSQL is running on localhost.[/dim]")

    # ── Session name (human-readable, used as DB primary key) ────────────────
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

            # ── Exit ──────────────────────────────────────────────────────────
            if query.lower() in ["exit", "quit", ":q"]:
                console.print("\n[bold cyan]👋 Thank you for using Agent OTG. Goodbye![/bold cyan]\n")
                break

            # ── Help ──────────────────────────────────────────────────────────
            if query.lower() in ["help", "/help", "?", "--help"]:
                print_help_guide()
                continue

            # ── Clear screen ──────────────────────────────────────────────────
            if query.lower() in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome_banner()
                continue

            # ── Reset memory ──────────────────────────────────────────────────
            if query.lower() == "reset":
                clear_history()
                current_file_path    = None
                current_file_content = None
                stage("🧹 [Memory]", "Conversation context and loaded file cleared!", style="bold green")
                print()
                continue

            # ── List generated files ──────────────────────────────────────────
            if query.lower() in ["/files", "files", "/ls", "ls"]:
                handle_list_files()
                continue

            # ── Upload document ───────────────────────────────────────────────
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
                    stage("📄 [Uploaded]", f"Successfully ingested [bold yellow]{os.path.basename(path)}[/bold yellow] ({char_count:,} chars)", style="bold green")
                    console.print("  [dim]You can now ask questions about it by mentioning 'this file', 'this document', etc.[/dim]\n")
                continue

            # ── History inspection ────────────────────────────────────────────
            if query.lower() in ["history", "/history"]:
                handle_history()
                continue

            # ── Inject active file content if referenced ──────────────────────
            if current_file_path and current_file_content:
                trigger_phrases = [
                    "this file", "this document", "this sheet", "this pdf",
                    "the uploaded file", "the data", "the file", "uploaded",
                ]
                if any(phrase in query.lower() for phrase in trigger_phrases):
                    query += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End of file content ---"
                    stage("📎 [Attachment]", f"Attached content from [yellow]{os.path.basename(current_file_path)}[/yellow] to prompt", style="dim")

            # ── Command routing ───────────────────────────────────────────────
            if query.startswith("/rag "):
                handle_rag(query[5:].strip())
            elif query.lower() == "/rag":
                console.print("  [dim]Usage: /rag <question>[/dim]\n")
            elif query.startswith("/doc "):
                handle_doc(query[5:].strip())
            elif query.lower() == "/doc":
                console.print("  [dim]Usage: /doc <file path>[/dim]\n")
            elif query.startswith("/img "):
                rest = query[5:].strip()
                img_path, q_text = parse_image_command(rest)
                handle_image(img_path, q_text)
            elif query.lower() in ["/kb", "/kb-stats", "/kbstats"]:
                handle_kb_stats()
            elif query.lower() in ["/models", "/config", "/env"]:
                handle_show_models()
            elif query.lower() == "/cancel":
                current_file_path    = None
                current_file_content = None
                stage("🚫 [Cancelled]", "File attachment and active context cleared.", style="bold yellow")
                print()
            elif query.lower() in ["/reset", "reset"]:
                clear_history()
                current_file_path    = None
                current_file_content = None
                stage("🧹 [Memory]", "Conversation context and loaded file cleared!", style="bold green")
                print()
            elif query.startswith("/complex "):
                ask_anything(query[len("/complex "):].strip(), force_multi=True)
            elif query.startswith("/agent "):
                handle_agent(query[len("/agent "):].strip())
            elif query.startswith("/image "):
                rest = query[len("/image "):].strip()
                img_path, q_text = parse_image_command(rest)
                handle_image(img_path, q_text)
            else:
                ask_anything(query)

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]👋 Session paused. Type exit or Ctrl+C again to quit.[/bold cyan]\n")
            continue
        except Exception as exc:
            console.print(f"\n❌ [bold red]Unexpected Error:[/bold red] {exc}\n")


if __name__ == "__main__":
    main()