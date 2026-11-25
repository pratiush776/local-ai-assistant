"""Public API for the assistant's background jobs package (Unit 9)."""

from .control import jobs_enabled, set_jobs_enabled
from .deliveries import Delivery, deliver, list_deliveries
from .pending import (
    PendingAction,
    get_pending,
    list_pending,
    queue_pending,
    resolve_pending,
)
from .runner import Job, run_job
from .suggest import Proposal, propose_reminder, scan_inbox

__all__ = [
    "Delivery",
    "Job",
    "PendingAction",
    "Proposal",
    "deliver",
    "get_pending",
    "jobs_enabled",
    "list_deliveries",
    "list_pending",
    "propose_reminder",
    "queue_pending",
    "resolve_pending",
    "run_job",
    "scan_inbox",
    "set_jobs_enabled",
]
