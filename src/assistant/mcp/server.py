# --- block 1: import MCPServer + your existing tools module, create the server ---
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from assistant.agent import tools

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


# --- block 2b: the mailbox, read-only (7.2). Same plug, more tools. ---
@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_emails(query: str) -> str:
    """Search the mailbox by keyword across subject and body; returns matching
    emails (with their full bodies), newest-first."""
    return tools.run_tool("search_emails", {"query": query})


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_unread_emails() -> str:
    """List all of the user's unread emails, newest-first."""
    return tools.run_tool("get_unread_emails", {})


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_email(email_id: str) -> str:
    """Retrieve one full email by its id from the user's mailbox."""
    return tools.run_tool("get_email", {"email_id": email_id})


# --- block 2c: write actions (7.3). read_only_hint=False -> the approval gate fires. ---
@mcp.tool(annotations=ToolAnnotations(read_only_hint=False))
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. Writes to the local outbox for review;
    nothing leaves the machine. Requires the user's approval before it runs."""
    return tools.run_tool("send_email", {"to": to, "subject": subject, "body": body})


# --- block 3: run over stdio when executed as a script ---
if __name__ == "__main__":
    mcp.run()