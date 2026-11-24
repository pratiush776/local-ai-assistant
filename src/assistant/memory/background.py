"""Run post-turn consolidation and record its results by source trace ID."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .store import store_turn

_log = logging.getLogger(__name__)
# Preserve submission order within this process and avoid competing memory jobs
# for the local model. Pending jobs live in this process's memory.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")


def after_pending_writes(operation, *args, **kwargs):
    """Apply a panel edit after chat writes already submitted to this process."""
    return _worker.submit(operation, *args, **kwargs).result()


def _remember(
    turn: str,
    *,
    observed_at: datetime,
    source_trace_id: str,
    model: str,
    results_path: Path,
) -> None:
    result = {
        "source_trace_id": source_trace_id,
        "status": "completed",
        "started_at": datetime.now(UTC).isoformat(),
        "writes": [],
    }
    try:
        for write in store_turn(
            turn, observed_at=observed_at, source_trace_id=source_trace_id, model=model
        ):
            result["writes"].append(write)
    except Exception as exc:  # noqa: BLE001 -- record any failed job in its source trace
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_at"] = datetime.now(UTC).isoformat()

    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pending = results_path.with_suffix(".tmp")
        pending.write_text(json.dumps(result) + "\n", encoding="utf-8")
        pending.replace(results_path)
    except OSError:
        _log.exception("Could not record memory results for turn %s", source_trace_id)


def enqueue_turn(
    turn: str,
    *,
    observed_at: datetime,
    source_trace_id: str,
    model: str,
    results_path: Path,
) -> None:
    _worker.submit(
        _remember,
        turn,
        observed_at=observed_at,
        source_trace_id=source_trace_id,
        model=model,
        results_path=results_path,
    )
