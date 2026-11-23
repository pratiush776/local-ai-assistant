"""MCP client: connect to our MCP server, turn the tools it advertises into the
agent's tool-menu format, and read each tool's own risk annotation to decide approval.

Why in-process (`Client(server)`, not a subprocess): the reminder tools read the
per-task temporary DB through `tools._temporary_db`, a ContextVar that lives only
in THIS process. A subprocess server couldn't see it, so the agent eval's per-task
isolation — and its score — would break. (Verified: an in-process call lands in the
temp DB and leaks nothing to disk.) A real subprocess wall is for untrusted /
third-party servers, e.g. Google in 7.5.

Risk lives ON each tool (the standard MCP vocabulary): MCP tools carry a
`read_only_hint` annotation set by the server; direct tools.py schemas carry the same
hint in their `annotations` block. A tool is gated for approval unless it declared
itself read-only (unknown -> ask). No separate write-tool set to keep in sync.
"""

import anyio
from mcp import Client
from mcp.types import TextContent

from assistant.agent import tools
from assistant.mcp.server import mcp as _server


def _to_agent_schema(t) -> dict:
    """One MCP tool -> the {type, function:{name, description, parameters}} the model sees."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema,
        },
    }


def _wire(schema: dict) -> dict:
    """The pure menu entry the model receives — drop our internal annotations block."""
    return {"type": "function", "function": schema["function"]}


async def _alist() -> list:
    async with Client(_server) as client:
        return (await client.list_tools()).tools


# Fetched once at import: the server's tools (with their annotations).
_mcp_tools = anyio.run(_alist)
MCP_TOOL_NAMES: set[str] = {t.name for t in _mcp_tools}

# The non-MCP tools still served directly by tools.py.
_direct = [s for s in tools.TOOL_SCHEMAS if s["function"]["name"] not in MCP_TOOL_NAMES]

# The full menu sent to the model: direct tools + MCP tools, all pure {type, function}.
SCHEMAS: list[dict] = [_wire(s) for s in _direct] + [
    _to_agent_schema(t) for t in _mcp_tools
]

# approval decision per tool, read from each tool's OWN risk annotation.
_APPROVAL: dict[str, bool] = {}
for s in _direct:
    _read_only = bool(s.get("annotations", {}).get("read_only_hint", False))
    _APPROVAL[s["function"]["name"]] = not _read_only
for t in _mcp_tools:
    _read_only = bool(getattr(t.annotations, "read_only_hint", None)) if t.annotations else False
    _APPROVAL[t.name] = not _read_only


def needs_approval(name: str) -> bool:
    """A tool needs approval unless it declared itself read-only (unknown -> ask)."""
    return _APPROVAL.get(name, True)


# name -> the tool's menu description (what the model reads to pick it). The approval
# screen shows this as context; it stays subordinate to the real args (see format_approval).
DESCRIPTIONS: dict[str, str] = {
    s["function"]["name"]: s["function"].get("description", "") for s in SCHEMAS
}


def describe(name: str) -> str:
    """The tool's own description, or '' if unknown."""
    return DESCRIPTIONS.get(name, "")


async def _acall(name: str, args: dict) -> str:
    async with Client(_server) as client:
        res = await client.call_tool(name, args)
        return "".join(b.text for b in res.content if isinstance(b, TextContent))


def run(name: str, args: dict) -> str:
    """Call an MCP tool in-process and return its text result (sync facade)."""
    return anyio.run(_acall, name, args)
