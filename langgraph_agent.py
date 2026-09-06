"""
langgraph_agent.py - Production LangGraph agent for Agent OTG
=============================================================

This module integrates LangGraph into the live Agent OTG application.
It runs ALONGSIDE the existing router.py system — it does NOT replace it.

HOW IT FITS IN:
  - main.py exposes a  POST /ask/agent  endpoint that calls run_agent()
  - ask.py  exposes a  /agent <query>   terminal command that calls run_agent()
  - router.py, tools.py, and all other existing files are untouched in design.

WHAT THIS ADDS OVER PLAIN router.py:
  - Structured multi-step reasoning graph (not just model routing)
  - Explicit tool-calling loop: the model can call multiple real tools in sequence
  - Clear node-level visibility: each step is labelled and traceable
  - Retry-capable architecture: easy to add retry edges later

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
            +-----------+
                  |
                  v
           +---------+
           |  respond  |  <- final answer node, always reached
           +---------+
                  |
                 END
"""

import os

# ── Telemetry / analytics OFF — must be set BEFORE any langchain/langgraph import.
#
# WHY THESE LINES EXIST AND MUST NEVER BE REMOVED:
#   This project is fully offline and sovereign — no queries, usage stats,
#   or telemetry of any kind should leave the machine.
#   LangChain/LangSmith and LangGraph try to phone home by default.
#   These three env vars are the official switch to disable all of that.
os.environ["LANGCHAIN_TRACING_V2"]       = "false"   # disables LangSmith tracing
os.environ["LANGGRAPH_CLI_NO_ANALYTICS"] = "1"       # disables LangGraph CLI telemetry
os.environ["ANONYMIZED_TELEMETRY"]       = "false"   # disables chromadb / other anon stats

# ── Standard library ──────────────────────────────────────────────────────────
import json
from typing import Literal, TypedDict, Annotated
import operator

# ── LangGraph / LangChain ─────────────────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, END

# ── Real tools from this project ──────────────────────────────────────────────
# We import the actual TOOL_FUNCTIONS and TOOL_SCHEMAS from tools.py so the
# LangGraph agent can call write_docx, generate_pdf_from_text, search_documents, etc.
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_system_prompt() -> str:
    """Read the shared system_prompt.txt — same file router.py uses."""
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are a helpful, honest, on-premise AI assistant."


def _history_to_lc_messages(history: list) -> list:
    """
    Convert the HISTORY list from router.py (list of {role, content} dicts)
    into LangChain message objects that ChatOllama understands.
    Only the last 6 turns are included to keep context short and fast.
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


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# Each field below is passed from node to node as the graph runs.
# Annotated[list, operator.add] means: when two nodes both write to "tool_log",
# LangGraph appends the lists together instead of overwriting. For all other
# fields it simply overwrites with the latest value.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    question:    str                               # original user question, never changes
    history:     list                              # conversation turns from router.HISTORY
    needs_tool:  bool                              # did classify node decide a tool is needed?
    tool_name:   str                               # which tool to call (e.g. "generate_pdf_from_text")
    tool_args:   dict                              # arguments to pass to that tool
    tool_log:    Annotated[list, operator.add]     # growing log of tool calls + results
    final_answer: str                              # the completed response


# ══════════════════════════════════════════════════════════════════════════════
# THE MODEL
# Same local Ollama server as router.py — no external connections.
# ══════════════════════════════════════════════════════════════════════════════

_llm = ChatOllama(
    model       = "qwen2.5:14b",
    base_url    = "http://localhost:11434",
    temperature = 0,
)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: CLASSIFY
# Asks the model: "Does this question need a tool, and if so which one?"
# Returns: needs_tool (bool), tool_name (str), tool_args (dict)
# ══════════════════════════════════════════════════════════════════════════════

# Build a short summary of available tools to include in the classify prompt.
_TOOL_SUMMARY = "\n".join(
    f"  - {s['function']['name']}: {s['function']['description']}"
    for s in TOOL_SCHEMAS
)

_CLASSIFY_SYSTEM = f"""You are a routing assistant deciding if a user's request needs a tool.

Available tools:
{_TOOL_SUMMARY}

Reply with ONLY valid JSON in this exact format — nothing before or after it:
{{
  "needs_tool": true or false,
  "tool_name": "exact_function_name or empty string if no tool",
  "tool_args": {{argument key-value pairs, or empty object if no tool}}
}}

If needs_tool is false, set tool_name to "" and tool_args to {{}}.
If needs_tool is true, fill in tool_name with one of the exact function names above,
and tool_args with the arguments that function needs.
Do not invent argument keys — use only the parameters described for that tool.
"""


def classify_node(state: AgentState) -> dict:
    """
    NODE 1: CLASSIFY
    Determines if the question requires a tool call.
    Asks the LLM to produce structured JSON so we can parse the decision cleanly.
    """
    sys_prompt = _load_system_prompt()

    # Include recent conversation history so the model has context.
    history_msgs = _history_to_lc_messages(state["history"])

    messages = (
        [SystemMessage(content=_CLASSIFY_SYSTEM)]
        + history_msgs
        + [HumanMessage(content=state["question"])]
    )

    response = _llm.invoke(messages)
    raw      = response.content.strip()

    # Try to parse the JSON the model returned.
    # If parsing fails for any reason, fall back to "no tool needed".
    try:
        # Strip markdown code fences in case the model added them.
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        parsed     = json.loads(raw)
        needs_tool = bool(parsed.get("needs_tool", False))
        tool_name  = str(parsed.get("tool_name", "")).strip()
        tool_args  = dict(parsed.get("tool_args", {}))

        # Safety: if needs_tool but tool_name isn't a real tool, disable it.
        if needs_tool and tool_name not in TOOL_FUNCTIONS:
            needs_tool = False
            tool_name  = ""
            tool_args  = {}
    except Exception:
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
# Called automatically by LangGraph after classify_node.
# Returns the label of the next node to run.
# ══════════════════════════════════════════════════════════════════════════════

def route_after_classify(state: AgentState) -> Literal["needs_tool", "direct"]:
    """
    If classify_node said a tool is needed, go to the tools node.
    Otherwise jump straight to respond.
    """
    return "needs_tool" if state["needs_tool"] else "direct"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: TOOLS
# Only reached when classify said needs_tool=True.
# Calls the real tool from tools.py and stores the result in tool_log.
# ══════════════════════════════════════════════════════════════════════════════

def tools_node(state: AgentState) -> dict:
    """
    NODE 2: TOOLS
    Executes the tool that classify_node selected.
    Logs what was called and what the result was so respond_node can use it.
    """
    tool_name = state["tool_name"]
    tool_args = state["tool_args"]

    if tool_name not in TOOL_FUNCTIONS:
        # Shouldn't happen because classify_node already validated, but be safe.
        log_entry = {
            "tool":   tool_name,
            "args":   tool_args,
            "result": f"Error: '{tool_name}' is not a registered tool.",
        }
        return {"tool_log": [log_entry]}

    try:
        result = TOOL_FUNCTIONS[tool_name](**tool_args)
    except Exception as e:
        result = f"Error running {tool_name}: {e}"

    log_entry = {
        "tool":   tool_name,
        "args":   tool_args,
        "result": str(result),
    }
    # tool_log uses operator.add, so returning a list appends to any existing entries.
    return {"tool_log": [log_entry]}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: RESPOND
# Always the last node. Generates the final human-readable answer.
# If tools were called, their results are injected into context so the LLM
# can reference them naturally in its reply.
# ══════════════════════════════════════════════════════════════════════════════

def respond_node(state: AgentState) -> dict:
    """
    NODE 3: RESPOND
    Always runs last. Produces the final answer.
    Has full visibility of what tools ran and what they returned.
    """
    sys_prompt = _load_system_prompt()
    history_msgs = _history_to_lc_messages(state["history"])

    # Build a context block from any tool results.
    tool_context = ""
    if state.get("tool_log"):
        parts = []
        for entry in state["tool_log"]:
            parts.append(
                f"Tool '{entry['tool']}' was called with args {entry['args']}.\n"
                f"Result: {entry['result']}"
            )
        tool_context = "\n\n".join(parts)

    if tool_context:
        # Tell the LLM what the tool returned, then ask for a friendly summary.
        messages = (
            [SystemMessage(content=sys_prompt)]
            + history_msgs
            + [
                HumanMessage(content=state["question"]),
                AIMessage(content=f"[Tool results]\n{tool_context}"),
                HumanMessage(content=(
                    "Using the tool results above, give a clear and friendly final answer "
                    "to the original question. If a file was created, mention its path."
                )),
            ]
        )
    else:
        # No tools were used — answer directly from knowledge.
        messages = (
            [SystemMessage(content=sys_prompt)]
            + history_msgs
            + [HumanMessage(content=state["question"])]
        )

    response = _llm.invoke(messages)
    return {"final_answer": response.content.strip()}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# Wires nodes and edges together, then compiles into a runnable object.
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph():
    """
    Assembles and compiles the LangGraph agent graph.
    The compiled object is cached in the module-level variable _GRAPH below
    so we only build it once per process.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify", classify_node)   # Node 1: classify intent
    builder.add_node("tools",    tools_node)       # Node 2: run tool (optional)
    builder.add_node("respond",  respond_node)     # Node 3: generate final answer

    # Entry point
    builder.set_entry_point("classify")

    # Conditional branch after classify
    builder.add_conditional_edges(
        "classify",             # source
        route_after_classify,   # function that decides which way to go
        {
            "needs_tool": "tools",    # -> go to tools node
            "direct":     "respond",  # -> skip tools, go straight to respond
        },
    )

    # After tools always go to respond
    builder.add_edge("tools", "respond")

    # After respond the graph ends
    builder.add_edge("respond", END)

    return builder.compile()


# Compile once at import time — reused for every call.
_GRAPH = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# These are the two functions that main.py and ask.py import.
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str, history: list | None = None) -> dict:
    """
    Run the LangGraph agent on a single question and return the result.

    Args:
        query   : the user's question or instruction
        history : list of {role, content} dicts from router.HISTORY (may be empty)

    Returns a dict with:
        final_answer  : str  — the agent's response
        needs_tool    : bool — whether a tool was called
        tool_log      : list — details of every tool call made
    """
    if history is None:
        history = []

    initial_state: AgentState = {
        "question":    query,
        "history":     history,
        "needs_tool":  False,
        "tool_name":   "",
        "tool_args":   {},
        "tool_log":    [],
        "final_answer": "",
    }

    result = _GRAPH.invoke(initial_state)

    return {
        "final_answer": result["final_answer"],
        "needs_tool":   result["needs_tool"],
        "tool_log":     result.get("tool_log", []),
    }
