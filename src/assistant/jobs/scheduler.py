"""Run jobs in-process on a schedule (Lesson 9.1 glue).

Tool check at unit start chose APScheduler 3.11 BackgroundScheduler, living inside
the app process (fallback was plain `cron` calling a CLI; in-process is simpler and
keeps one source of truth for the DB and traces). The scheduler owns the *when*;
`run_job` owns the *what* and the safety (reads run, writes queue, cloud unreachable).
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .control import jobs_enabled
from .runner import Job, run_job

log = logging.getLogger(__name__)


def build_scheduler() -> BackgroundScheduler:
    """A scheduler that runs jobs on background threads in this process."""
    return BackgroundScheduler()


def schedule_job(
    scheduler: BackgroundScheduler, job: Job, trigger: str = "cron", **trigger_args
) -> None:
    """Register `job` to fire on `trigger`. Examples:
    schedule_job(s, brief, "cron", hour=6)            -> every day at 06:00
    schedule_job(s, brief, "date", run_date=when)     -> once, at a moment
    schedule_job(s, brief, "interval", minutes=1)     -> every minute
    """
    scheduler.add_job(
        _run_safely,
        trigger,
        args=[job],
        id=job.id,
        replace_existing=True,  # re-registering the same id updates it, not duplicates
        **trigger_args,
    )


def _run_safely(job: Job) -> None:
    """Run one job, catching failures so a bad job logs instead of killing the
    scheduler thread. The kill switch is checked here, so a paused system's jobs
    stay registered but no-op. (The per-job budget lives on the Job itself.)"""
    if not jobs_enabled():
        log.info("job %s skipped — background jobs are paused (kill switch)", job.id)
        return
    try:
        pending_id = run_job(job)
        log.info("job %s ran; pending=%s", job.id, pending_id)
    except Exception:
        log.exception("job %s failed", job.id)
