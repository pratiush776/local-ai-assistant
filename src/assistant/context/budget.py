from collections.abc import Callable

from ..models.ollama import LOCAL_CONTEXT_WINDOW

HISTORY_PROMPT_RESERVE = 2048
LOCAL_HISTORY_BUDGET = LOCAL_CONTEXT_WINDOW - HISTORY_PROMPT_RESERVE
SUMMARY_TOKEN_RESERVE = 256


def estimate_tokens(text: str) -> float:
    return len(text) / 4


def trim_history(messages: list, context: float, budget: int) -> tuple[list, float]:
    trimmed_messages = messages[:]
    while len(trimmed_messages) > 2 and context > budget:
        for _ in range(2):
            msg = trimmed_messages.pop(0)
            context -= estimate_tokens(msg["content"])
    return trimmed_messages, context


def messages_to_text(messages: list) -> str:
    text = ""
    for msg in messages:
        text += msg["role"] + ": " + msg["content"] + "\n"
    return text.strip()


def compact(
    history: list[dict[str, str]],
    summarize: Callable[[str], str],
    *,
    budget: int = LOCAL_HISTORY_BUDGET,
) -> list[dict[str, str]]:
    """Summarize older exchanges while preserving a recent user-led suffix."""
    if estimate_tokens(messages_to_text(history)) <= budget:
        return history

    cutoff = 0
    recent_tokens = 0.0
    for index in range(len(history) - 1, 0, -1):
        msg = history[index]
        recent_tokens += estimate_tokens(f"{msg['role']}: {msg['content']}\n")
        if recent_tokens > budget - SUMMARY_TOKEN_RESERVE:
            break
        if msg["role"] == "user":
            cutoff = index
    if cutoff == 0:
        raise ValueError(
            "The history budget cannot fit the latest exchange and a recap."
        )

    summary = summarize(messages_to_text(history[:cutoff])).strip()
    if not summary:
        raise ValueError("The summarizer returned an empty recap.")
    recap = {
        "role": "assistant",
        "content": f"Earlier conversation summary:\n{summary}",
    }
    compacted = [recap, *history[cutoff:]]
    if estimate_tokens(messages_to_text(compacted)) > budget:
        raise ValueError("The compacted history exceeds its token budget.")
    return compacted
