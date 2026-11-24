"""Context-window management: token estimation and history trimming."""

from .budget import (  # noqa: F401
    LOCAL_HISTORY_BUDGET,
    compact,
    estimate_tokens,
    messages_to_text,
    trim_history,
)
