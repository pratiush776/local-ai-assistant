"""Retrieval backend selection.

One env var picks the engine; the rest of the app never knows which is live:
    RETRIEVAL_BACKEND=paradedb   uv run ...
The default `sqlite` backend runs in-process without Docker. ParadeDB stays behind the seam as the scale-out
path (hosted/large corpus). Backend modules are imported lazily (inside
get_backend) so neither can trigger a circular import while `rag` is still loading.
"""

import os

from .base import RetrievalBackend

_BACKEND: RetrievalBackend | None = None


def get_backend() -> RetrievalBackend:
    """The active backend, instantiated once per process."""
    global _BACKEND
    if _BACKEND is None:
        name = os.environ.get("RETRIEVAL_BACKEND", "sqlite").lower()
        if name == "sqlite":
            from .sqlite import SQLiteBackend

            _BACKEND = SQLiteBackend()
        elif name == "paradedb":
            from .paradedb import ParadeDBBackend

            _BACKEND = ParadeDBBackend()
        else:
            raise ValueError(
                f"unknown RETRIEVAL_BACKEND={name!r}; use 'sqlite' or 'paradedb'"
            )
    return _BACKEND


__all__ = ["RetrievalBackend", "get_backend"]
