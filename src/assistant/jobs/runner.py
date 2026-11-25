"""Headless job runner. A job is a prompt + a schedule + a budget (Lesson 9.1).

The schedule lives in the scheduler (APScheduler glue); the budget is the loop's
own MAX_STEPS / MAX_SECONDS ceilings. So a Job here only needs an id and a prompt.

`run_job` is the KERNEL (learner-typed): run the agent headless, queue the first
proposed write, always write the run's trace. A job may restrict its tool surface
(`tools`) and ask for its final text to be delivered (`deliver`) — e.g. the 9.2 brief.
"""

import logging
from dataclasses import dataclass

from ..agent import loop
from ..mcp import client
from ..respond import Event
from ..trace.recorder import record_turn, write_trace
from . import deliveries
from .pending import queue_pending

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str  # stable slug, e.g. "morning-brief"
    title: str  # one-line label shown in the Inbox ("Morning brief")
    prompt: str  # the user message the agent runs
    tools: list[str] | None = None  # restrict the tool surface (None = all tools)
    deliver: bool = False  # deliver the final answer text (a brief), not just writes
    require_tool_use: bool = (
        False  # don't deliver if the run called no tools (anti-hallucination)
    )
    max_steps: int = 6  # per-job model-call budget (a background job runs unwatched)
    max_seconds: int = 90  # per-job wall-clock budget


def _schemas_for(job: Job) -> list[dict]:
    """The tool schemas a job may use. `tools` names an allow-list (least privilege
    for a read-only brief); None hands over the full surface."""
    if job.tools is None:
        return client.SCHEMAS
    allowed = set(job.tools)
    return [s for s in client.SCHEMAS if s["function"]["name"] in allowed]


def run_job(job: Job) -> int | None:
    """Run one job headless and return the queued pending item's id, or None if the
    job finished without proposing a write.

    Reads execute; the first proposed write stops the agent loop and becomes a
    pending item for the Inbox; the cloud is never reached (run_agent is local).
    The run's trace is always written, so a queued job is still visible.
    """
    # Drive the agent to completion OR to its first write. The loop yields an
    # approval_request and returns at that write, so this scan sees it last.
    events: list[Event] = []
    pending: Event | None = None
    for event in loop.run_agent(
        job.prompt,
        [],
        _schemas_for(job),
        max_steps=job.max_steps,
        max_seconds=job.max_seconds,
    ):
        events.append(event)
        if event.kind == "approval_request":
            pending = event
            break

    record = record_turn(job.prompt, events)
    write_trace(record)

    if pending is not None:  # a write to approve -> queue it (9.1)
        return queue_pending(
            job.id,
            job.title,
            job.prompt,
            pending.text,
            pending.data["args"],
            pending.data["state"],
            pending.data["tainted"],
            record["turn_id"],
        )

    if job.deliver:  # a read-only brief -> deliver its final text (9.2)
        called_tools = any(e.kind == "tool_call" for e in events)
        if job.require_tool_use and not called_tools:
            # The model answered without gathering -> likely hallucinated. Refuse to
            # deliver fabricated content; the absence is safer than a fake brief.
            log.warning(
                "job %s produced a brief without calling any tool; not delivering "
                "(require_tool_use). trace=%s",
                job.id,
                record["turn_id"],
            )
            return None
        final_text = events[-1].text if events else ""
        deliveries.deliver(job.id, job.title, final_text, record["turn_id"])
    return None
