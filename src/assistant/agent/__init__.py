"""The agent: the tool-use loop, its tools, and the injection defences.

Re-export the loop's public surface so callers keep using `agent.run_agent(...)`,
`agent.resume_agent(...)` and `agent.format_approval(...)` after the package move.
Submodules `tools`, `security` and `pydantic_loop` are imported directly where used.
"""

from .loop import (  # noqa: F401
    format_approval,
    needs_approval,
    resume_agent,
    run_agent,
)
