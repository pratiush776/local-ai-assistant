from importlib import import_module

import pytest

from assistant.context import compact, estimate_tokens, messages_to_text, trim_history


# helper: builds a valid message with n_tokens long
def make_msg(role: str, n_tokens: int) -> dict:
    return {"role": role, "content": "x" * (n_tokens * 4)}


def test_estimate_tokens():
    assert estimate_tokens("abcdefgh") == 2


def test_trim_under_budget_keeps_everything():
    # build messages: user, assistant pairs
    messages = [make_msg("user", 10), make_msg("assistant", 10)] * 2

    # have context < budget -> expected messages and context is unchanged
    trimmed, context = trim_history(messages, context=40, budget=100)

    assert trimmed == messages
    assert context == 40


def test_trim_drops_oldest_pair_first():
    old = [make_msg("user", 10), make_msg("assistant", 10)]
    new = [make_msg("user", 5), make_msg("assistant", 5)]

    # call trim_history with both old + new, context = 10+10+5+5, budget < old_context
    trimmed, context = trim_history(old + new, context=30, budget=15)

    assert trimmed == new
    assert context == 10


def test_trim_doesnot_change_callers_list():
    messages = [make_msg("user", 10), make_msg("assistant", 10)] * 2

    trim_history(messages, 40, 15)

    assert len(messages) == 4


def test_trim_atleast_keeps_last_pair_even_over_budget():
    messages = [make_msg("user", 10), make_msg("assistant", 10)] * 2

    trimmed, _context = trim_history(messages, 40, 5)

    assert len(trimmed) == 2


def test_context_to_text_empty():
    text = messages_to_text([])

    assert text == ""


def test_context_to_text_formats_turns():
    messages = [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Hi Sam, how can I help you today?"},
    ]
    text = messages_to_text(messages)
    expected = """user: My name is Sam.\nassistant: Hi Sam, how can I help you today?"""
    assert text == expected


@pytest.mark.parametrize(
    "messages", [[], [make_msg("user", 10), make_msg("assistant", 10)]]
)
def test_compact_preserves_history_under_budget(messages):
    def unexpected_summary(_):
        pytest.fail("History that fits should skip summarization")

    assert compact(messages, unexpected_summary) is messages


def test_compact_passes_only_the_older_prefix_to_the_summarizer():
    older = [
        {"role": "assistant", "content": "Earlier conversation summary: old facts"},
        make_msg("user", 150),
        make_msg("assistant", 150),
    ]
    recent = [make_msg("user", 10), make_msg("assistant", 10)]
    history = older + recent

    def summarize(text):
        assert text == messages_to_text(older)
        return "Updated recap."

    result = compact(history, summarize, budget=300)

    assert result[0] == {
        "role": "assistant",
        "content": "Earlier conversation summary:\nUpdated recap.",
    }
    assert result[1:] == recent
    assert history == older + recent


def test_compact_rejects_a_summary_that_overflows_the_budget():
    older = [make_msg("user", 150), make_msg("assistant", 150)]
    recent = [make_msg("user", 10), make_msg("assistant", 10)]

    with pytest.raises(ValueError, match="exceeds its token budget"):
        compact(older + recent, lambda _: "x" * 1200, budget=300)


def test_respond_routes_and_answers_with_the_same_compacted_history(monkeypatch):
    module = import_module("assistant.respond")
    from assistant.router.decide import Decision

    older = [make_msg("user", 3300), make_msg("assistant", 3300)]
    recent = [make_msg("user", 10), make_msg("assistant", 10)]
    history = older + recent
    recap = {"role": "assistant", "content": "Earlier conversation summary:\nRecap."}

    def summarize(text, model):
        assert text == messages_to_text(older)
        return "Recap."

    def decide(user_msg, context):
        assert context == messages_to_text([recap, *recent])
        return Decision(route="local", label="chat", confidence=1)

    def stream(messages, system, model):
        assert messages == [recap, *recent, {"role": "user", "content": "Continue."}]
        yield "delta", "Continuing."

    monkeypatch.setattr(module, "summarize_history", summarize)
    monkeypatch.setattr(module, "decide", decide)
    monkeypatch.setattr(module, "_stream", stream)
    events = list(module.respond("Continue.", history))

    assert next(ev for ev in events if ev.kind == "compaction").data["history"] == [
        recap,
        *recent,
    ]
    assert events[-1].text == "Continuing."
    assert history == older + recent


def test_failed_compaction_falls_back_without_replacing_history(monkeypatch):
    module = import_module("assistant.respond")
    older = [make_msg("user", 3300), make_msg("assistant", 3300)]
    recent = [make_msg("user", 10), make_msg("assistant", 10)]

    def reject_summary(text, model):
        raise ValueError("recap exceeded available space")

    monkeypatch.setattr(module, "summarize_history", reject_summary)
    prepared, event = module._prepare_history(older + recent)

    assert prepared == recent
    assert event.data["error"] == "ValueError"
    assert "history" not in event.data
