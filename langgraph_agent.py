"""
langgraph_agent.py - Production LangGraph agent for Agent OTG
=============================================================

This module integrates LangGraph into the live Agent OTG application.
It runs ALONGSIDE the existing router.py system — it does NOT replace it.

HOW IT FITS IN:
  - main.py exposes a  POST /ask/agent  endpoint that calls run_agent()
  - ask.py  exposes a  /agent <query>   terminal command that calls run_agent()
  - router.py, tools.py, and all other existing files are untouched in design.

HOW THE GRAPH WORKS:
  [START]
      |
      v
  +-----------+
  |  classify  |  <- reads question, decides: "needs_tool" or "direct"
  +-----------+
      |
  +---+-------------------+
  |  route_after_classify  |  <- branching condition
  +---+-------------------+
"needs_tool"|       |"direct"
            v       v
       +--------+  (skip to respond)
       |  tools  |      |
       +--------+       |
            |           |
        [loop back to classify up to MAX_TOOL_ITERATIONS times]
            |           |
            +-----------+
                  |
                  v
           +---------+
           |  respond  |  <- final answer node, always reached
           +---------+
                  |
                 END

MULTI-TOOL LOOP:
  After executing a tool, the agent re-enters classify to check
  if another tool is needed, up to MAX_TOOL_ITERATIONS times.
  This allows chained operations like: generate PDF → get page count.
"""

import os
import sys
import json
import re
from typing import Literal, TypedDict, Annotated
import operator

# ── Telemetry / analytics OFF ─────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"]       = "false"
os.environ["LANGGRAPH_CLI_NO_ANALYTICS"] = "1"
os.environ["ANONYMIZED_TELEMETRY"]       = "false"

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

# Maximum tool calls per agent invocation (prevents infinite loops)
MAX_TOOL_ITERATIONS = 5

# ── Script directory for resolving relative paths ─────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_system_prompt() -> str:
    """Load system prompt from disk, relative to this file's directory."""
    prompt_path = os.path.join(_SCRIPT_DIR, "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are a helpful, honest, on-premise AI assistant."


def _history_to_lc_messages(history: list) -> list:
    """
    Convert router.py HISTORY (list of {role, content} dicts) into
    LangChain message objects. Only the last 6 turns are included.
    """
    messages = []
    for turn in history[-6:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _extract_json_from_response(raw: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles:
      - Plain JSON
      - ```json ... ``` fences
      - ```...``` fences without language tag
      - JSON embedded mid-text (finds the first { ... } block)
    """
    text = raw.strip()

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response: {raw[:200]}")


def _tool_summary() -> str:
    """Build a human-readable tool list from TOOL_SCHEMAS."""
    return "\n".join(
        f"  - {s['function']['name']}: {s['function']['description']}"
        for s in TOOL_SCHEMAS
    )


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    question:      str
    history:       list
    needs_tool:    bool
    tool_name:     str
    tool_args:     dict
    tool_log:      Annotated[list, operator.add]   # accumulates across iterations
    final_answer:  str
    iteration:     int   # how many tool calls have been made so far


# ══════════════════════════════════════════════════════════════════════════════
# THE MODEL
# ══════════════════════════════════════════════════════════════════════════════

import config as _cfg

_llm = ChatOllama(
    model       = _cfg.MAIN_MODEL,
    base_url    = _cfg.OLLAMA_BASE_URL,
    temperature = 0,
)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFY_SYSTEM_TEMPLATE = """\
You are a routing assistant deciding if a user's request needs a tool.

Available tools:
{tool_summary}

Previous tool results (if any):
{{tool_context}}

Reply with ONLY valid JSON — no markdown, no explanation, nothing before or after it:
{{
  "needs_tool": true or false,
  "tool_name": "exact_function_name or empty string if no tool",
  "tool_args": {{argument key-value pairs, or empty object if no tool}}
}}

Rules:
- If needs_tool is false, set tool_name to "" and tool_args to {{}}.
- If needs_tool is true, fill in tool_name with exactly one name from the list above.
- Only include argument keys that are listed in that tool's parameter description.
- Do NOT invent argument keys.
- The content/text argument should contain the FULL relevant content, not a placeholder.
- If a previous tool already produced the result needed to answer, set needs_tool to false.
"""

_CLASSIFY_SYSTEM_BASE = _CLASSIFY_SYSTEM_TEMPLATE.format(tool_summary=_tool_summary())


def classify_node(state: AgentState) -> dict:
    """NODE 1: Determines if the question requires a tool call."""
    # If we've hit the iteration cap, force direct response
    if state.get("iteration", 0) >= MAX_TOOL_ITERATIONS:
        _log(f"[classify] Hit MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS}), forcing direct response")
        return {"needs_tool": False, "tool_name": "", "tool_args": {}}

    history_msgs = _history_to_lc_messages(state["history"])

    # Build tool context summary for re-classification after tool use
    tool_context = ""
    if state.get("tool_log"):
        parts = []
        for entry in state["tool_log"]:
            parts.append(f"  Tool '{entry['tool']}' → {str(entry['result'])[:300]}")
        tool_context = "\n".join(parts)

    classify_system = _CLASSIFY_SYSTEM_BASE.replace("{tool_context}", tool_context or "None yet")

    messages = (
        [SystemMessage(content=classify_system)]
        + history_msgs
        + [HumanMessage(content=state["question"])]
    )

    try:
        response = _llm.invoke(messages)
        raw      = response.content.strip()

        parsed     = _extract_json_from_response(raw)
        needs_tool = bool(parsed.get("needs_tool", False))
        tool_name  = str(parsed.get("tool_name", "")).strip()
        tool_args  = dict(parsed.get("tool_args", {}))

        # Safety: if needs_tool but tool_name isn't a real tool, disable it
        if needs_tool and tool_name not in TOOL_FUNCTIONS:
            _log(f"[classify] LLM suggested unknown tool '{tool_name}', disabling")
            needs_tool = False
            tool_name  = ""
            tool_args  = {}

        _log(f"[classify] needs_tool={needs_tool} tool={tool_name!r} iteration={state.get('iteration', 0)}")
    except Exception as exc:
        _log(f"[classify] JSON parse failed: {exc} — defaulting to direct response")
        needs_tool = False
        tool_name  = ""
        tool_args  = {}

    return {
        "needs_tool": needs_tool,
        "tool_name":  tool_name,
        "tool_args":  tool_args,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BRANCHING CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def route_after_classify(state: AgentState) -> Literal["needs_tool", "direct"]:
    return "needs_tool" if state["needs_tool"] else "direct"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def tools_node(state: AgentState) -> dict:
    """NODE 2: Executes the tool selected by classify_node, then loops back."""
    tool_name = state["tool_name"]
    tool_args = state["tool_args"]

    # Defensive: args might arrive as a JSON string
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except Exception:
            tool_args = {}

    if tool_name not in TOOL_FUNCTIONS:
        result = f"Error: '{tool_name}' is not a registered tool."
        _log(f"[tools] Unknown tool: {tool_name}")
    else:
        try:
            _log(f"[tools] Calling '{tool_name}' with args: {list(tool_args.keys())}")
            result = TOOL_FUNCTIONS[tool_name](**tool_args)
            _log(f"[tools] '{tool_name}' result: {str(result)[:120]}")
        except TypeError as e:
            # Missing or unexpected keyword arguments
            result = f"Error: wrong arguments for tool '{tool_name}': {e}"
            _log(f"[tools] TypeError in '{tool_name}': {e}")
        except Exception as e:
            result = f"Error running tool '{tool_name}': {e}"
            _log(f"[tools] Exception in '{tool_name}': {e}")

    log_entry = {
        "tool":   tool_name,
        "args":   tool_args,
        "result": str(result),
    }

    return {
        "tool_log":  [log_entry],          # Annotated[list] reducer appends this
        "iteration": state.get("iteration", 0) + 1,
        # Reset tool fields — classify will repopulate if another tool is needed
        "needs_tool": False,
        "tool_name":  "",
        "tool_args":  {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: RESPOND
# ══════════════════════════════════════════════════════════════════════════════

def respond_node(state: AgentState) -> dict:
    """NODE 3: Generates the final human-readable answer."""
    sys_prompt   = _load_system_prompt()
    history_msgs = _history_to_lc_messages(state["history"])

    tool_context = ""
    if state.get("tool_log"):
        parts = []
        for entry in state["tool_log"]:
            args_summary = ", ".join(f"{k}={repr(str(v))[:60]}" for k, v in entry.get("args", {}).items())
            parts.append(
                f"Tool '{entry['tool']}' was called with ({args_summary}).\n"
                f"Result: {entry['result']}"
            )
        tool_context = "\n\n".join(parts)

    if tool_context:
        messages = (
            [SystemMessage(content=sys_prompt)]
            + history_msgs
            + [
                HumanMessage(content=state["question"]),
                AIMessage(content=f"[Tool results]\n{tool_context}"),
                HumanMessage(content=(
                    "Using the tool results above, give a clear, complete, and friendly final answer "
                    "to the original question. If a file was created or saved, mention its exact path. "
                    "If data was extracted, summarize the key findings."
                )),
            ]
        )
    else:
        messages = (
            [SystemMessage(content=sys_prompt)]
            + history_msgs
            + [HumanMessage(content=state["question"])]
        )

    try:
        response = _llm.invoke(messages)
        final_answer = response.content.strip()
    except Exception as exc:
        final_answer = f"I encountered an error generating a response: {exc}"

    if not final_answer:
        final_answer = "I completed the requested actions. Please check the generated_files/ directory for any output files."

    _log(f"[respond] Final answer length: {len(final_answer)} chars")
    return {"final_answer": final_answer}


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    """Internal logger — prints to stderr so it doesn't interfere with stdout streaming."""
    try:
        print(f"[agent] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH (compiled once at import time)
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("tools",    tools_node)
    builder.add_node("respond",  respond_node)

    builder.set_entry_point("classify")

    # After classify: go to tools or skip directly to respond
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "needs_tool": "tools",
            "direct":     "respond",
        },
    )

    # After tools: loop back to classify (to check if another tool is needed)
    # classify will eventually route to "direct" → respond when done
    builder.add_edge("tools",    "classify")
    builder.add_edge("respond",  END)

    return builder.compile()


_GRAPH = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str, history: list | None = None) -> dict:
    """
    Run the LangGraph agent on a single question.

    Args:
        query   : the user's question or instruction
        history : list of {role, content} dicts from router.HISTORY (may be empty)

    Returns:
        final_answer  : str  — the agent's response
        needs_tool    : bool — whether any tool was called
        tool_log      : list — details of every tool call made
    """
    if history is None:
        history = []

    initial_state: AgentState = {
        "question":     query,
        "history":      history,
        "needs_tool":   False,
        "tool_name":    "",
        "tool_args":    {},
        "tool_log":     [],
        "final_answer": "",
        "iteration":    0,
    }

    try:
        result = _GRAPH.invoke(initial_state)
    except Exception as exc:
        _log(f"[run_agent] Graph execution error: {exc}")
        return {
            "final_answer": f"Agent encountered an error: {exc}",
            "needs_tool":   False,
            "tool_log":     [],
        }

    tool_log   = result.get("tool_log", [])
    needs_tool = len(tool_log) > 0  # true if any tool was actually called

    return {
        "final_answer": result.get("final_answer", ""),
        "needs_tool":   needs_tool,
        "tool_log":     tool_log,
    }
