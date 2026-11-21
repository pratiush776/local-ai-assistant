from assistant import agent, tools
from assistant.evals import agent_eval


def test_agent_eval_seeds_and_scores_disposable_state(tmp_path, monkeypatch):
    app_db = tmp_path / "assistant.db"
    calendar = tmp_path / "calendar.json"
    calendar.write_text(
        '{"events": [{"id": "dentist", "title": "Dentist appointment", '
        '"start": "2024-05-18T09:00", "end": "2024-05-18T10:00"}]}'
    )
    monkeypatch.setattr(tools, "DB_PATH", app_db)
    monkeypatch.setattr(tools, "CALENDAR_SEED", calendar)
    monkeypatch.setattr(tools, "_db_ready", False)
    monkeypatch.setattr(agent_eval, "capture", lambda *_: None)

    tools.run_tool("add_reminder", {"text": "Keep me", "date": "2024-05-21"})
    original = app_db.read_bytes()

    def scripted_chat(messages, _schemas):
        if messages[-1]["role"] == "tool":
            return {"role": "assistant", "content": "Done"}
        assert "Dentist appointment" in tools.run_tool(
            "list_events", {"date": "2024-05-18"}
        )
        assert "Submit expense report" in tools.run_tool("list_reminders", {})
        assert "Keep me" not in tools.run_tool("list_reminders", {})
        assert "milk" not in tools.run_tool("list_reminders", {})
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "add_reminder",
                        "arguments": {"text": "Buy milk", "date": "2024-05-16"},
                    }
                }
            ],
        }

    monkeypatch.setattr(agent, "_chat", scripted_chat)
    task = {"id": "OWN2", "task": "Remind me to buy milk", "kind": "write"}
    task["check"] = "check_milk_reminder"
    assert agent_eval.run_task(task)["passed"]
    assert agent_eval.run_task(task)["passed"]
    assert app_db.read_bytes() == original
    assert tools.run_tool("list_reminders", {}) == "2024-05-21: Keep me"
