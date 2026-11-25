"""9.1: run_job runs headless, queues the FIRST proposed write, always traces.

Scripted model + spies, so these are deterministic and never touch Ollama or the
real app DB. The store round-trip uses an isolated tmp_path DB.
"""

from assistant import agent
from assistant.jobs import Job, run_job, runner
from assistant.jobs.pending import (
    get_pending,
    list_pending,
    queue_pending,
    resolve_pending,
)

# A scripted assistant turn asking to run the write tool `add_reminder`.
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
# A scripted final turn (no tool_calls) -> the job ends with no write.
FINAL = {"role": "assistant", "content": "Here is your brief.", "tool_calls": None}

# A scripted read-tool call (counts as "gathered").
READ_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"function": {"name": "list_reminders", "arguments": {}}}],
}


def _scripted_chat(script):
    responses = list(script)

    def fake_chat(messages, tool_schemas):
        return responses.pop(0)

    return fake_chat


def _spy_queue(calls):
    def fake(job_id, title, prompt, name, args, state, tainted, trace_id):
        calls.append(
            {
                "job_id": job_id,
                "title": title,
                "prompt": prompt,
                "name": name,
                "args": args,
                "state": state,
                "tainted": tainted,
                "trace_id": trace_id,
            }
        )
        return 42

    return fake


def test_run_job_queues_the_first_write(monkeypatch):
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([WRITE_CALL]))
    monkeypatch.setattr(
        runner, "record_turn", lambda prompt, events: {"turn_id": "t-test"}
    )
    monkeypatch.setattr(runner, "write_trace", lambda record: None)
    calls = []
    monkeypatch.setattr(runner, "queue_pending", _spy_queue(calls))

    pid = run_job(Job(id="brief", title="Morning brief", prompt="remind me to do x"))

    assert pid == 42  # returned the queued row's id
    assert len(calls) == 1
    c = calls[0]
    assert (c["job_id"], c["title"], c["prompt"]) == (
        "brief",
        "Morning brief",
        "remind me to do x",
    )
    assert c["name"] == "add_reminder"
    assert c["args"] == {"text": "x", "date": "2024-05-16"}
    assert c["tainted"] is False
    assert c["trace_id"] == "t-test"
    assert c["state"]  # the resumable messages were carried through


def test_run_job_returns_none_when_no_write(monkeypatch):
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([FINAL]))
    monkeypatch.setattr(
        runner, "record_turn", lambda prompt, events: {"turn_id": "t-test"}
    )
    monkeypatch.setattr(runner, "write_trace", lambda record: None)
    calls = []
    monkeypatch.setattr(runner, "queue_pending", _spy_queue(calls))

    pid = run_job(Job(id="brief", title="Morning brief", prompt="just chat"))

    assert pid is None  # nothing proposed -> nothing queued
    assert calls == []


def test_pending_store_roundtrip(tmp_path):
    db = tmp_path / "assistant.db"
    row_id = queue_pending(
        "brief",
        "Morning brief",
        "remind me",
        "add_reminder",
        {"text": "x", "date": "2024-05-16"},
        [{"role": "user", "content": "hi"}],
        False,
        "trace-abc",
        path=db,
    )
    pending = list_pending(path=db)
    assert len(pending) == 1
    assert pending[0].title == "Morning brief"
    assert pending[0].args == {"text": "x", "date": "2024-05-16"}
    assert pending[0].status == "pending"
    assert get_pending(row_id, path=db).trace_id == "trace-abc"

    assert resolve_pending(row_id, True, path=db) is True
    assert list_pending(path=db) == []  # resolved items leave the queue
    assert resolve_pending(row_id, True, path=db) is False  # can't resolve twice


def _spy_deliver(calls):
    def fake(job_id, title, body, trace_id):
        calls.append({"job_id": job_id, "title": title, "body": body})
        return 1

    return fake


def test_brief_delivered_when_tools_were_called(monkeypatch):
    # A tool call happened -> the brief is trustworthy -> deliver it.
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([READ_CALL, FINAL]))
    monkeypatch.setattr(runner, "record_turn", lambda p, e: {"turn_id": "t"})
    monkeypatch.setattr(runner, "write_trace", lambda r: None)
    delivered = []
    monkeypatch.setattr(runner.deliveries, "deliver", _spy_deliver(delivered))

    job = Job(
        id="brief",
        title="Brief",
        prompt="brief me",
        deliver=True,
        require_tool_use=True,
    )
    assert run_job(job) is None  # a delivery, not a queued write
    assert len(delivered) == 1
    assert delivered[0]["body"] == "Here is your brief."


def test_brief_not_delivered_when_no_tool_was_called(monkeypatch):
    # No tool call -> likely hallucinated -> the guard refuses to deliver.
    monkeypatch.setattr(agent.loop, "_chat", _scripted_chat([FINAL]))
    monkeypatch.setattr(runner, "record_turn", lambda p, e: {"turn_id": "t"})
    monkeypatch.setattr(runner, "write_trace", lambda r: None)
    delivered = []
    monkeypatch.setattr(runner.deliveries, "deliver", _spy_deliver(delivered))

    job = Job(
        id="brief",
        title="Brief",
        prompt="brief me",
        deliver=True,
        require_tool_use=True,
    )
    assert run_job(job) is None
    assert delivered == []  # nothing delivered: absence beats a fabricated brief


def test_job_budget_is_passed_to_the_loop(monkeypatch):
    # The Job's per-job budget reaches run_agent, not the module default.
    seen = {}

    def fake_run_agent(prompt, history, schemas, *, max_steps, max_seconds):
        seen["max_steps"] = max_steps
        seen["max_seconds"] = max_seconds
        return iter([])  # no events -> run_job just traces and returns None

    monkeypatch.setattr(runner.loop, "run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "record_turn", lambda p, e: {"turn_id": "t"})
    monkeypatch.setattr(runner, "write_trace", lambda r: None)

    run_job(Job(id="j", title="J", prompt="p", max_steps=3, max_seconds=30))
    assert seen == {"max_steps": 3, "max_seconds": 30}


def test_kill_switch_skips_scheduled_jobs(monkeypatch):
    from assistant.jobs import control, scheduler

    ran = []
    monkeypatch.setattr(scheduler, "run_job", lambda job: ran.append(job.id))

    control.set_jobs_enabled(False)
    try:
        scheduler._run_safely(Job(id="paused", title="x", prompt="p"))
        assert ran == []  # kill switch on -> job did not run
        control.set_jobs_enabled(True)
        scheduler._run_safely(Job(id="live", title="x", prompt="p"))
        assert ran == ["live"]  # switch off -> job runs
    finally:
        control.set_jobs_enabled(True)  # don't leak state into other tests
