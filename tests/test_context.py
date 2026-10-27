from assistant.context import estimate_tokens, trim_history


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
