def estimate_tokens(text: str) -> float:
    return len(text) / 4


def trim_history(messages: list, context: float, budget: int) -> tuple[list, float]:
    trimmed_messages = messages[:]
    while len(trimmed_messages) > 2 and context > budget:
        for _ in range(2):
            msg = trimmed_messages.pop(0)
            context -= estimate_tokens(msg["content"])
    return trimmed_messages, context
