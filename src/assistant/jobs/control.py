"""The background-jobs kill switch (Lesson 9.4).

An always-on system needs an off switch. This is one global flag the scheduler and
the scan check before they run: when it is off, scheduled work is skipped (the jobs
stay registered, they just no-op until re-enabled). In-memory for now — it resets to
ON on restart, which is fine while jobs are re-registered on each launch anyway;
persisting it is parked.
"""

import logging

log = logging.getLogger(__name__)

_ENABLED = True


def jobs_enabled() -> bool:
    return _ENABLED


def set_jobs_enabled(on: bool) -> bool:
    """Flip the switch. Returns the new state."""
    global _ENABLED
    _ENABLED = bool(on)
    log.info("background jobs %s", "ENABLED" if _ENABLED else "PAUSED (kill switch)")
    return _ENABLED
