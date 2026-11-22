"""Lesson 6.6 — the SAME agent, ported to Pydantic AI (side-by-side with agent.py).

The tool *bodies* are reused verbatim from tools.py. What the framework replaces
is everything around them:

    agent.py (hand-written)            agent_pydantic.py (framework)
    while-loop in _loop                agent.run_sync() runs the loop internally
    httpx POST to Ollama (_chat)       OllamaModel + OllamaProvider
    7 hand-written JSON schemas        @agent.tool_plain  (schema from signature + docstring)
    AGENT_SYSTEM string                Agent(system_prompt=...)
    MAX_STEPS / seen / MAX_SECONDS     built-in step + usage limits
    _execute try/except                framework catches tool exceptions
    needs_approval + approval_request  requires_approval=True  (declared on the tool)
      + resume_agent (two-phase gate)  + DeferredToolRequests -> DeferredToolResults

`drive()` returns our own Event stream so the agent eval scores this agent with the
exact same code it uses for the hand-written one.
"""

from collections.abc import Iterator

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ToolDenied,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from assistant import tools
from assistant.agent import AGENT_SYSTEM, MAX_STEPS
from assistant.respond import PRIMARY_LOCAL_MODEL, Event

# Ollama speaks the OpenAI chat-completions API on /v1; no API key for a local server.
_model = OllamaModel(
    PRIMARY_LOCAL_MODEL,
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

# output_type lists DeferredToolRequests so an approval-gated call PAUSES the run
# (the framework's "stop-the-world" flow) instead of executing — the same shape as
# our run_agent yielding `approval_request` and stopping.
agent = Agent(
    _model,
    system_prompt=AGENT_SYSTEM,
    output_type=[str, DeferredToolRequests],
)


# --- read tools (no approval) ------------------------------------------------
# The docstring becomes the tool description; the `Args:` lines become the per-
# parameter descriptions. That is the whole TOOL_SCHEMAS block, generated.
@agent.tool_plain
def search_files(query: str) -> str:
    """Search the user's personal files and return matching passages.

    Args:
        query: What to search the user's files for.
    """
    return tools._search_files(query)


@agent.tool_plain
def read_file(filename: str) -> str:
    """Given one known filename, return that file's full text.

    Args:
        filename: Name of the file whose text you want.
    """
    return tools._read_file(filename)


@agent.tool_plain
def list_events(date: str) -> str:
    """List the user's calendar events on a specific date.

    Args:
        date: The date in 'YYYY-MM-DD' to list events for.
    """
    return tools._list_events(date)


@agent.tool_plain
def find_event(title: str) -> str:
    """Find calendar events by title or keyword across ALL dates. Use when the user
    names an event without giving its date (e.g. 'my dentist appointment') and you
    need to know when it is.

    Args:
        title: A word or phrase from the event's title to search for.
    """
    return tools._find_event(title)


@agent.tool_plain
def list_reminders() -> str:
    """List all of the user's reminders."""
    return tools._list_reminders()


# --- write tools: the approval gate, now one declarative flag -----------------
# `requires_approval=True` is the framework's version of our read_only_hint annotation
# + needs_approval(): risk declared on the tool. The framework stops the run before
# these ever execute.
@agent.tool_plain(requires_approval=True)
def add_event(
    title: str, date: str, time: str = "09:00", duration_min: int = 60
) -> str:
    """Add an event to the user's calendar.

    Args:
        title: Short title of the event.
        date: Event date in 'YYYY-MM-DD'.
        time: Start time in 'HH:MM' (24h). Defaults to 09:00.
        duration_min: Length in minutes. Defaults to 60.
    """
    return tools._add_event(title, date, time, duration_min)


@agent.tool_plain(requires_approval=True)
def add_reminder(text: str, date: str) -> str:
    """Add a reminder for the user on a given date.

    Args:
        text: What to remind the user about.
        date: Reminder date in 'YYYY-MM-DD'.
    """
    return tools._add_reminder(text, date)


# --- translate the framework run into our Event stream -----------------------
def _return_str(part: ToolReturnPart) -> str:
    content = getattr(part, "content", None)
    return content if isinstance(content, str) else part.model_response_str()


def _events_from(result) -> Iterator[Event]:
    """Walk the framework's message history and yield the same tool_call /
    tool_result events the hand-written loop emits, so scoring is identical."""
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                yield Event(
                    "tool_call", text=part.tool_name, data={"args": part.args_as_dict()}
                )
            elif isinstance(part, ToolReturnPart):
                yield Event(
                    "tool_result",
                    text=part.tool_name,
                    data={"result": _return_str(part)},
                )


def drive(task_text: str, approve: bool = True) -> list[Event]:
    """Run one task to completion, auto-approving (or denying) every write.
    Returns the full Event list, ending in a `done` event carrying the answer."""
    result = agent.run_sync(task_text)
    for _ in range(MAX_STEPS):  # bound the approve/resume cycles
        if not isinstance(result.output, DeferredToolRequests):
            break
        answers = DeferredToolResults()
        for call in result.output.approvals:
            answers.approvals[call.tool_call_id] = (
                True if approve else ToolDenied("The user declined to run this.")
            )
        result = agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=answers,
        )

    events = list(_events_from(result))
    answer = result.output if isinstance(result.output, str) else ""
    events.append(Event("done", text=answer, data={}))
    return events
