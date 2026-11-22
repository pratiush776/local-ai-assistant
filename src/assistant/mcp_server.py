# --- block 1: import MCPServer + your existing tools module, create the server ---
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from assistant import tools

mcp = MCPServer("Local MCP")


# --- block 2: expose EACH reminder tool as an @mcp.tool() ---
# Each tool declares its own risk via a read_only_hint annotation; the client reads
# it to decide approval (read-only -> no approval; a write -> the gate fires).
@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_reminders() -> str:
    """List all of the user's reminders."""
    return tools.run_tool("list_reminders", {})


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False))
def add_reminder(text: str, date: str) -> str:
    """Add a reminder for the user on a given date (YYYY-MM-DD)."""
    return tools.run_tool("add_reminder", {"text": text, "date": date})


# --- block 3: run over stdio when executed as a script ---
if __name__ == "__main__":
    mcp.run()