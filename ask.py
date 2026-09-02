"""
ask.py - The Terminal Chat Client, with a visible staged process:
    Understanding -> Planning (if multi-part) -> Working (with live status) -> Done

Commands:
    - Type any question normally (auto-detects if it's multi-part)
    - '/complex ...' to force-split a question
    - '/image <path> [question]' to ask about an image file on disk
    - 'reset' to clear conversation memory
    - 'exit' to quit
"""

import sys
import time
import base64
import os
import random
import threading
import requests as _requests   # used to fetch images from URLs

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from router import classify_question, stream_answer, stream_image_answer, break_into_tasks, clear_history

# Rotating status phrases shown while waiting on a model - purely cosmetic,
# just so the terminal never looks frozen while a big model "thinks".
THINKING_PHRASES = [
    "Thinking...",
    "Working through it...",
    "Composing a response...",
    "Almost there...",
]


def show_spinner(messages, stop_flag, start_time):
    frames = ["|", "/", "-", "\\"]
    i = 0
    msg_index = 0
    last_switch = start_time
    while not stop_flag[0]:
        now = time.time()
        elapsed = now - start_time
        if len(messages) > 1 and now - last_switch > 2.5:
            msg_index = (msg_index + 1) % len(messages)
            last_switch = now
        sys.stdout.write(f"\r  {frames[i % 4]} {messages[msg_index]} ({elapsed:.1f}s) ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


def stage(text):
    print(f"  >> {text}")


def run_with_spinner(messages, work_fn):
    """Runs work_fn() while a spinner shows, returns work_fn()'s result."""
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(messages, stop, start_time))
    spinner.start()
    result = work_fn()
    stop[0] = True
    spinner.join()
    return result


def stream_with_spinner(model, query):
    """
    Keeps the spinner running until the FIRST token actually arrives
    (this is the part that was missing - it covers the model's "thinking"
    delay before it starts producing output), then prints tokens live.
    """
    start_time = time.time()
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time))
    spinner.start()

    token_count = 0
    spinner_stopped = False
    for token in stream_answer(model, query):
        if not spinner_stopped:
            stop[0] = True
            spinner.join()
            spinner_stopped = True
        sys.stdout.write(token)
        sys.stdout.flush()
        token_count += 1

    if not spinner_stopped:
        stop[0] = True
        spinner.join()

    return token_count


def handle_single(query, info):
    model = info["model"]
    stage(f'Decision: category = "{info["category"]}" -> routing to {model}')
    stage(f'Reason: {info["reason"]}')
    print("-" * 60)

    start_time = time.time()
    token_count = stream_with_spinner(model, query)
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "-" * 60)
    print(f"  Done in {elapsed}s | Model: {model} | ~{token_count} tokens\n")


def handle_multi(query):
    stage("This looks like a multi-part request - breaking it down...")
    tasks = run_with_spinner(["Planning sub-tasks..."], lambda: break_into_tasks(query))

    print(f"\n  Plan ({len(tasks)} sub-tasks):")
    for i, task in enumerate(tasks, 1):
        print(f"    {i}. {task['label']}  ->  {task['model']}  ({task['category']})")
    print()

    start_time = time.time()
    for i, task in enumerate(tasks, 1):
        stage(f"Working on sub-task {i}/{len(tasks)}: {task['label']} [{task['model']}]")
        print("-" * 60)
        stream_with_spinner(task["model"], task["task"])
        print("\n" + "-" * 60 + "\n")

    elapsed = round(time.time() - start_time, 2)
    print(f"  All {len(tasks)} sub-tasks done in {elapsed}s\n")


def ask_anything(query, force_multi=False):
    stage("Understanding your question...")
    info = run_with_spinner(["Analyzing..."], lambda: classify_question(query))

    if force_multi or info["is_multi_part"]:
        handle_multi(query)
    else:
        handle_single(query, info)


def handle_image(source, question):
    """
    Accepts either a local file path OR a public image URL.
    Encodes the image to base64 and sends it to qwen2.5vl:7b.

    HOW TO USE:
        /image <url_or_path> [optional question]

    Examples (URL):
        /image https://example.com/photo.jpg
        /image https://example.com/photo.jpg What breed is this dog?

    Examples (local file):
        /image C:\\Users\\me\\photo.jpg
        /image ./diagram.png Explain what this diagram shows.

    Supported formats : PNG, JPEG, JPG, WEBP, GIF
    """
    # Strip angle brackets in case user pastes <https://...> style URLs
    source = source.strip("<>").strip()

    is_url = source.startswith("http://") or source.startswith("https://")

    # ── Fetch from URL ────────────────────────────────────────────
    if is_url:
        stage(f"Fetching image from URL...")
        try:
            # Use a browser User-Agent - some servers (e.g. Wikipedia) block
            # Python's default requests agent with a 403 Forbidden response.
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            resp = _requests.get(source, headers=headers, timeout=15)
            resp.raise_for_status()
            image_b64 = base64.b64encode(resp.content).decode()
            label = source.split("/")[-1].split("?")[0] or "image"
        except Exception as e:
            print(f"  [error] Could not fetch URL: {e}\n")
            return

    # ── Read from local disk ──────────────────────────────────────
    else:
        if not os.path.isfile(source):
            print(f"  [error] File not found: {source}\n")
            return
        ext = os.path.splitext(source)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            print(f"  [error] Unsupported format '{ext}'. Use PNG, JPEG, WEBP, or GIF.\n")
            return
        with open(source, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        label = os.path.basename(source)

    if not question:
        question = "Describe this image in detail."

    stage(f'Image: {label}  |  Question: "{question}"')
    stage("Routing to vision model: qwen2.5vl:7b")
    print("-" * 60)

    start_time = time.time()

    # Spinner until first token arrives, then stream live
    stop = [False]
    spinner = threading.Thread(target=show_spinner, args=(THINKING_PHRASES, stop, start_time))
    spinner.start()

    token_count  = 0
    spinner_done = False
    for token in stream_image_answer(image_b64, question):
        if not spinner_done:
            stop[0] = True
            spinner.join()
            spinner_done = True
        sys.stdout.write(token)
        sys.stdout.flush()
        token_count += 1

    if not spinner_done:
        stop[0] = True
        spinner.join()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "-" * 60)
    print(f"  Done in {elapsed}s | Model: qwen2.5vl:7b | ~{token_count} tokens\n")


def main():
    print("=" * 60)
    print("  LOCAL AI ROUTER - Interactive Chat")
    print("=" * 60)
    print("  Commands:")
    print("    - Type any question (auto-splits multi-part requests)")
    print("    - /complex <question> to force-split")
    print("    - /image <url_or_path> [question] to ask about an image")
    print("        URL:   /image https://example.com/photo.jpg What is this?")
    print("        File:  /image C:\\Users\\me\\photo.jpg What is this?")
    print("        Note:  question is optional (defaults to 'Describe this image')")
    print("        Formats: PNG, JPEG, WEBP, GIF")
    print("    - reset to clear conversation memory")
    print("    - exit / quit to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if query.lower() == "reset":
                clear_history()
                print("Conversation memory cleared.\n")
                continue
            if query.startswith("/complex "):
                ask_anything(query[len("/complex "):].strip(), force_multi=True)
            elif query.startswith("/image "):
                # Format: /image <path> [optional question]
                rest   = query[len("/image "):].strip()
                parts  = rest.split(" ", 1)        # split on first space only
                path   = parts[0]
                q_text = parts[1].strip() if len(parts) > 1 else ""
                handle_image(path, q_text)
            else:
                ask_anything(query)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()