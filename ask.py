"""
ask.py - The Interactive Terminal Client for Agent OTG

A high-performance, multi-model AI assistant running locally and fully offline.
Features:
    - Automatic Routing: routes coding, reasoning, casual chit-chat, and multi-part questions
    - Code + Walkthrough: ordered sequential pipeline for code generation followed by explanation
    - Vision: inspect local images or web URLs using qwen2.5vl:7b
    - Document Ingestion: upload and query Excel (.xlsx), Word (.docx), and PDF (.pdf) files
    - LangGraph Autonomous Agent: multi-step structured agent with tool execution
    - Interactive Chat History & Context Memory Management
"""

import sys
import time
import base64
import os
import random
import threading
import datetime
import glob
import json
import re
import shlex
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
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def stage(badge: str, text: str, style="bold cyan"):
    console.print(f"  [{style}]{badge}[/{style}] {text}")


def run_with_spinner(messages, work_fn):
    """Runs work_fn() while an animated spinner displays."""
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(messages, stop, start_time))
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
    """
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time))
    spinner.start()

    full_answer = ""
    spinner_stopped = False
    try:
        for token in stream_answer(model, query):
            if not spinner_stopped:
                stop[0] = True
                spinner.join()
                spinner_stopped = True
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

def print_welcome_banner():
    banner_text = (
        "[bold cyan]🤖 AGENT OTG — Local Autonomous Multi-Model System[/bold cyan]\n"
        "[dim]100% On-Premise • Air-Gapped / Private • Real-Time Routing • LangGraph Tools[/dim]"
    )
    console.print(Panel(banner_text, box=box.ROUNDED, border_style="cyan", padding=(1, 2)))

    table = Table(title="🚀 Available Capabilities & Command Guide", box=box.SIMPLE_HEAVY, border_style="bright_blue")
    table.add_column("Capability", style="bold green", no_wrap=True)
    table.add_column("Command / Trigger", style="yellow")
    table.add_column("Model Assigned", style="magenta")
    table.add_column("Description", style="white")

    table.add_row(
        "💻 Code Generation & Debugging",
        "Write a Python script for...",
        CODER_MODEL,
        "Auto-routes programming, bugs, scripts, SQL"
    )
    table.add_row(
        "🔄 Code + Walkthrough",
        "Write X and explain how it works",
        f"{CODER_MODEL} ➔ {MAIN_MODEL}",
        "Sequential 2-step pipeline (Code first, then explanation)"
    )
    table.add_row(
        "🧠 Deep Reasoning & Essays",
        "Explain quantum computing...",
        MAIN_MODEL,
        "Complex rationale, essays, logic, architecture"
    )
    table.add_row(
        "⚡ Quick Chat & Math",
        "Hello / What is 25 * 4?",
        FAST_MODEL,
        "Instant responses, small-talk, rapid answers"
    )
    table.add_row(
        "🗂️ Multi-Task Decomposition",
        "/complex <your query>",
        "Dynamic Multi-Model",
        "Splits request into independent parallel sub-tasks"
    )
    table.add_row(
        "👁️ Vision / Image Q&A",
        '/image <path_or_url> [question]',
        IMAGE_MODEL,
        "Inspect diagrams, photos, screenshots, charts"
    )
    table.add_row(
        "📄 Document Ingestion",
        'upload <path_to_file>',
        "Context Injector",
        "Extracts and queries .xlsx, .pdf, or .docx data"
    )
    table.add_row(
        "🛠️ LangGraph Agent",
        '/agent <instruction>',
        "qwen2.5:14b + Tools",
        "Autonomous agent with tools (PDF, Word, RAG search)"
    )
    table.add_row(
        "📜 Past Sessions",
        'history',
        "Session Store",
        "View and browse past interactive chat transcripts"
    )
    table.add_row(
        "🧹 Reset Memory",
        'reset',
        "Memory Guard",
        "Clears current conversation turns from RAM"
    )
    table.add_row(
        "❓ Full Help / Cheat Sheet",
        'help or /help',
        "Guide",
        "Re-displays this command matrix with sample prompts"
    )

    console.print(table)
    console.print("[dim]Type your question directly or use any command above. Type [bold red]exit[/bold red] to quit.[/dim]\n")


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
        ("📄 PDF File", '1) upload "C:\\docs\\manual.pdf"\n   2) Summarize the safety guidelines in this pdf.'),
        ("🛠️ Agent Tools", '/agent Create a Word document titled "Quarterly Update" with 3 key highlights.'),
        ("🛠️ PDF Tools", '/agent Extract all text from "generated_files/sample.pdf" and summarize it.'),
        ("🗂️ Complex Split", '/complex Write a marketing email for product launch AND generate SQL to find buyers.'),
        ("🧹 Reset", 'reset (clears conversation context so you can start a fresh topic).'),
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
    stage("🛠️ [LangGraph Agent]", "Initializing structured reasoning graph (classify ➔ tools? ➔ respond)...", style="bold magenta")
    stage("🤖 [Model]", "qwen2.5:14b with dynamic tool calling", style="cyan")
    console.print("-" * 65, style="dim")

    start_time = time.time()
    try:
        from langgraph_agent import run_agent
    except ImportError as e:
        console.print(f"  ❌ [bold red]LangGraph not available:[/bold red] {e}")
        console.print("  [dim]Install requirements via: py -m pip install langgraph langchain-ollama[/dim]")
        return

    result = run_with_spinner(
        ["Analyzing intent...", "Invoking registered tools...", "Synthesizing comprehensive response..."],
        lambda: run_agent(query, history=get_history()),
    )

    elapsed = round(time.time() - start_time, 2)

    if result.get("needs_tool") and result.get("tool_log"):
        tool_table = Table(title="🛠️ Tools Executed by Agent", box=box.ROUNDED, border_style="green")
        tool_table.add_column("Tool", style="bold yellow")
        tool_table.add_column("Result", style="white")
        for entry in result["tool_log"]:
            tool_table.add_row(entry["tool"], str(entry["result"])[:150])
        console.print(tool_table)
        console.print()

    console.print(result["final_answer"])
    console.print("-" * 65, style="dim")
    console.print(f"  ✅ [bold green]Done in {elapsed}s[/bold green] | Autonomous LangGraph Agent | [cyan]qwen2.5:14b[/cyan]\n")


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
            source = rest[1:end_idx]
            question = rest[end_idx + 1:].strip()
            return source, question
    elif rest.startswith("'"):
        end_idx = rest.find("'", 1)
        if end_idx != -1:
            source = rest[1:end_idx]
            question = rest[end_idx + 1:].strip()
            return source, question

    # Fallback to standard single space split
    parts = rest.split(" ", 1)
    source = parts[0]
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
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time))
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs("chat_sessions", exist_ok=True)
    session_filename = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S.json")
    session_path = os.path.join("chat_sessions", session_filename)
    router.start_new_session(session_path)

    print_welcome_banner()

    current_file_path = None
    current_file_content = None

    while True:
        try:
            # Styled prompt
            console.print("💬 [bold cyan]You[/bold cyan] [bold green]❯[/bold green] ", end="")
            query = input().strip()

            if not query:
                continue

            # Check exit
            if query.lower() in ["exit", "quit", ":q"]:
                console.print("\n[bold cyan]👋 Thank you for using Agent OTG. Goodbye![/bold cyan]\n")
                break

            # Help
            if query.lower() in ["help", "/help", "?", "--help"]:
                print_help_guide()
                continue

            # Clear screen
            if query.lower() in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome_banner()
                continue

            # Reset memory
            if query.lower() == "reset":
                clear_history()
                stage("🧹 [Memory]", "Conversation context memory has been cleared!", style="bold green")
                print()
                continue

            # Upload document
            if query.lower().startswith("upload "):
                raw_path = query[7:].strip()
                path = _clean_path(raw_path)
                from file_readers import process_file
                result = process_file(path)
                if result.startswith("Error:"):
                    console.print(f"  ❌ [bold red]{result}[/bold red]\n")
                else:
                    current_file_path = path
                    current_file_content = result
                    stage("📄 [Uploaded]", f"Successfully ingested [bold yellow]{os.path.basename(path)}[/bold yellow] ({len(result):,} characters)", style="bold green")
                    console.print("  [dim]You can now ask questions about it by mentioning 'this file', 'this document', etc.[/dim]\n")
                continue

            # History inspection
            if query.lower() == "history":
                sessions = sorted(glob.glob("chat_sessions/*.json"))
                if not sessions:
                    console.print("  [dim]No saved chat sessions found.[/dim]\n")
                    continue

                hist_table = Table(title="📜 Saved Chat Sessions", box=box.ROUNDED, border_style="cyan")
                hist_table.add_column("#", justify="right", style="cyan")
                hist_table.add_column("Session File", style="yellow")
                for i, s in enumerate(sessions[-10:], 1):
                    hist_table.add_row(str(i), os.path.basename(s))
                console.print(hist_table)

                try:
                    choice = input("  Pick a session number to inspect (or press Enter to cancel): ").strip()
                    if choice and choice.isdigit():
                        idx = int(choice)
                        if 1 <= idx <= min(10, len(sessions)):
                            selected_session = sessions[-10:][idx - 1]
                            with open(selected_session, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                console.print(f"\n--- Transcripts for {os.path.basename(selected_session)} ---", style="bold cyan")
                                for msg in data:
                                    role_color = "green" if msg.get("role") == "user" else "magenta"
                                    console.print(f"[{msg.get('timestamp')}] [{role_color}]{msg.get('role', '').upper()}:[/{role_color}]")
                                    console.print(msg.get("content", "").strip())
                                    console.print("-" * 50, style="dim")
                except Exception as e:
                    console.print(f"  [dim]Error viewing session: {e}[/dim]")
                print()
                continue

            # Inject active file content if referenced
            if current_file_path and current_file_content:
                trigger_phrases = ["this file", "this document", "this sheet", "this pdf", "the uploaded file", "the data"]
                if any(phrase in query.lower() for phrase in trigger_phrases):
                    query += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End of file content ---"
                    stage("📎 [Attachment]", f"Attached content from {os.path.basename(current_file_path)} to prompt", style="dim")

            # Routing command triggers
            if query.startswith("/complex "):
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