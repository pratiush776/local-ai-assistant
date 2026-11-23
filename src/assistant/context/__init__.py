"""Context-window management: token estimation and history trimming."""

from .budget import (  # noqa: F401
    estimate_tokens,
    messages_to_text,
    trim_history,
)
