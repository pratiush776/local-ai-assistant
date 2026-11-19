"""6.3: a write tool cannot run without approval; denial is handled gracefully.

Fake tools + a scripted model, so these are deterministic and never touch Ollama.
"""

from assistant import agent, tools

# A scripted assistant turn that asks to run the write tool `add_reminder`.
WRITE_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "function": {
                "name": "add_reminder",
                "arguments": {"text": "x", "date": "2024-05-16"},
            }
        }
    ],
}
# A scripted final turn (no tool_calls) -> the loop ends with an answer.
FINAL = {"role": "assistant", "content": "All done.", "tool_calls": None}


def _scripted_chat(script):
    responses = list(script)

    def fake_chat(messages, tool_schemas):
        return responses.pop(0)

    return fake_chat


def _recording_run_tool(ran):
    def fake(name, args):
        ran.append((name, args))
        return "reminder set"

    return fake


def test_write_waits_for_approval(monkeypatch):
    ran = []
    monkeypatch.setattr(agent, "_chat", _scripted_chat([WRITE_CALL]))
    monkeypatch.setattr(tools, "run_tool", _recording_run_tool(ran))

    events = list(agent.run_agent("remind me to do x", [], []))
    kinds = [e.kind for e in events]

    assert kinds[-1] == "approval_request"  # the run paused asking to approve
    assert "tool_result" not in kinds  # the write never produced a result
    assert ran == []  # the tool BODY never ran
    ap = events[-1]
    assert ap.text == "add_reminder"
    assert "state" in ap.data and "args" in ap.data  # resumable state is carried


def test_approve_runs_the_write(monkeypatch):
    ran = []
    monkeypatch.setattr(agent, "_chat", _scripted_chat([WRITE_CALL, FINAL]))
    monkeypatch.setattr(tools, "run_tool", _recording_run_tool(ran))

    events = list(agent.run_agent("remind me", [], []))
    ap = events[-1]
    resumed = list(
        agent.resume_agent(ap.data["state"], ap.text, ap.data["args"], True, [])
    )

    assert ran == [
        ("add_reminder", {"text": "x", "date": "2024-05-16"})
    ]  # ran once, after yes
    assert resumed[-1].kind == "done"


def test_deny_skips_the_write_and_replies(monkeypatch):
    ran = []
    monkeypatch.setattr(agent, "_chat", _scripted_chat([WRITE_CALL, FINAL]))
    monkeypatch.setattr(tools, "run_tool", _recording_run_tool(ran))

    events = list(agent.run_agent("remind me", [], []))
    ap = events[-1]
    resumed = list(
        agent.resume_agent(ap.data["state"], ap.text, ap.data["args"], False, [])
    )

    assert ran == []  # the write NEVER ran on denial
    tr = next(e for e in resumed if e.kind == "tool_result")
    assert "declined" in tr.data["result"].lower()
    assert resumed[-1].kind == "done"  # still produces a sensible final reply
