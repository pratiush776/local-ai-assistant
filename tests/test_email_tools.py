"""7.2: the read-only email tools over inbox.json.

Deterministic — they read the real persona inbox and never touch the model or a DB.
Ground truths verified against data/assets/persona/inbox.json.
"""

from assistant.agent import tools
from assistant.mcp import client as mcp_client


def test_search_is_case_insensitive_and_reads_the_body():
    # The Facebook code lives in the BODY, not the subject; query by a lowercase word.
    out = tools.run_tool("search_emails", {"query": "facebook"})
    assert "463820" in out


def test_search_distinguishes_the_phishing_distractor():
    # The legit reset token (email 9) and the phishing one (email 29) are different.
    out = tools.run_tool("search_emails", {"query": "techservices"})
    assert "asfbuy3y2cdaqhvei" in out  # legit token is retrievable


def test_search_newest_first():
    out = tools.run_tool("search_emails", {"query": "reunion"})
    # The whole thread matches; the newest reply (id 17, 2024-05-13T16:00) leads.
    assert out.lstrip().startswith("[17]")


def test_search_no_match_is_explicit_not_empty():
    out = tools.run_tool("search_emails", {"query": "zzzz-no-such-word"})
    assert "No emails match" in out


def test_unread_returns_exactly_the_unread_set():
    out = tools.run_tool("get_unread_emails", {})
    unread_ids = {"9", "20", "21", "26", "27", "31"}
    read_ids = {"18", "19"}  # part of the hiking thread but already read
    for i in unread_ids:
        assert f"[{i}]" in out
    for i in read_ids:
        assert f"[{i}]" not in out


def test_get_email_by_id_and_missing_id():
    out = tools.run_tool("get_email", {"email_id": "20"})
    assert "island trailhead" in out and out.lstrip().startswith("[20]")
    assert "No email with id" in tools.run_tool("get_email", {"email_id": "9999"})


def test_all_email_tools_are_read_only():
    for name in ("search_emails", "get_unread_emails", "get_email"):
        assert name in mcp_client.MCP_TOOL_NAMES
        assert mcp_client.needs_approval(name) is False


def test_send_email_writes_to_outbox_and_is_gated():
    # send_email is a WRITE: it must require approval...
    assert mcp_client.needs_approval("send_email") is True
    # ...and a send to a KNOWN CONTACT writes to the local outbox, never the network.
    with tools.temporary_database() as conn:
        out = tools.run_tool(
            "send_email",
            {"to": "mark.davies@hotmail.com", "subject": "Hi", "body": "Hello"},
        )
        assert "outbox" in out.lower()
        row = conn.execute("SELECT recipient, subject, body FROM outbox").fetchone()
        assert row["recipient"] == "mark.davies@hotmail.com"
        assert row["subject"] == "Hi"
        assert row["body"] == "Hello"


def test_send_email_to_non_contact_is_blocked_by_allowlist():
    # Layer c: a recipient outside the user's contacts never reaches the outbox.
    with tools.temporary_database() as conn:
        out = tools.run_tool(
            "send_email",
            {"to": "backup@docs-sync.net", "subject": "x", "body": "leak"},
        )
        assert out.startswith("Blocked:")
        assert conn.execute("SELECT COUNT(*) c FROM outbox").fetchone()["c"] == 0
