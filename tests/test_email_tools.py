"""7.2: the read-only email tools over inbox.json.

Deterministic — they read the real persona inbox and never touch the model or a DB.
Ground truths verified against data/assets/persona/inbox.json.
"""

from assistant import mcp_client, tools


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
