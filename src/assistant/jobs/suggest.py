"""The suggestion trigger (Lesson 9.3): one email in → a proposed reminder, or silence.

This is the first time the assistant acts on its *own* initiative: a background scan
reads new mail, and when an email carries a real dated commitment it proposes an
`add_reminder` into the same approval queue a human already controls (9.1). Nothing
runs unapproved.

The hard part is not the plumbing — it is *precision*. A proactive assistant that
proposes on junk gets muted, so a missed commitment is cheaper than a false one. Two
layers enforce that: the PROMPT biases the model toward silence (the ceiling), and a
CODE guard refuses anything ungrounded or undated (the floor a prompt can't give).

Reuses the structured-extraction pattern from `memory/` — a tool-call the model fills,
validated by pydantic, then dropped unless a literal quote backs it.
"""

import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..agent import tools
from ..models.ollama import chat_message
from ..storage import APP_DB_PATH
from .pending import list_pending, queue_pending

log = logging.getLogger(__name__)


class Proposal(BaseModel):
    """What the model reports about one email."""

    has_commitment: bool
    what: str = ""  # "Hiking with Mark at the island trailhead, 8 AM"
    when: str = ""  # resolved calendar date, "YYYY-MM-DD"
    evidence: str = ""  # the literal phrase from the email that states the date


# The structured-output channel: we hand the model ONE tool and let it fill the fields.
# Tool-call (not forced JSON) is the pattern that parses cleanly on qwen3.5 here.
REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_commitment",
        "description": "Report whether this email contains a dated commitment for Emma.",
        "parameters": {
            "type": "object",
            "properties": {
                "has_commitment": {
                    "type": "boolean",
                    "description": "True only if a real, dated commitment for Emma is present.",
                },
                "what": {
                    "type": "string",
                    "description": "Short description of the commitment (who/what/where).",
                },
                "when": {
                    "type": "string",
                    "description": "The commitment's calendar date as YYYY-MM-DD, resolved from the email.",
                },
                "evidence": {
                    "type": "string",
                    "description": "The exact phrase, copied verbatim from the email, that states the date.",
                },
            },
            "required": ["has_commitment", "what", "when", "evidence"],
        },
    },
}

# ---- KERNEL: the precision-first instruction (the ceiling) ----
SUGGEST_SYSTEM = """You scan ONE email and report, via report_commitment, whether it \
contains a concrete dated commitment that Emma should be reminded about.

Today is {today}.

Set has_commitment = true ONLY when BOTH hold:
- the email is a genuine event, meeting, or personal commitment for Emma, from a real \
person (not a company blast); and
- it names a specific date, or a relative date you can resolve against today (e.g. \
"Saturday 18th", "next Wednesday", "tomorrow").

Set has_commitment = false for: promotions, marketing, newsletters, event offers, \
social-network notifications, security or verification codes, two-factor codes, \
password resets and other automated system mail; and for any request that has no date \
("when you get a chance", "sometime soon").

When has_commitment = true:
- what: a short phrase — who, what, where.
- when: the resolved calendar date as YYYY-MM-DD.
- evidence: copy, word for word, the phrase in the email that states the date.

When unsure, set has_commitment = false. A wrong reminder is worse than a missed one, \
because a noisy assistant gets switched off."""


def _reported(message: dict) -> Proposal | None:
    """Pull the one report_commitment tool-call out and validate it. Any deviation
    (no call, wrong name, bad shape) -> None: no clean signal means stay silent."""
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    fn = calls[0].get("function", {})
    if fn.get("name") != "report_commitment" or not isinstance(fn.get("arguments"), dict):
        return None
    try:
        return Proposal.model_validate(fn["arguments"])
    except ValidationError:
        return None


def propose_reminder(email_text: str, *, today: str, model: str) -> Proposal | None:
    """One email -> a grounded, dated Proposal, or None (propose nothing).

    Precision floor (the code guard a prompt can't replace): keep the proposal only if
    the model flagged a commitment, named something, gave a date that really parses,
    and backed it with a phrase that literally appears in the email.
    """
    message = chat_message(
        [
            {"role": "system", "content": SUGGEST_SYSTEM.format(today=today)},
            {"role": "user", "content": f"<email>\n{email_text}\n</email>"},
        ],
        model,
        tools=[REPORT_TOOL],
        options={"temperature": 0, "num_predict": 300},
    )

    p = _reported(message)
    if p is None or not p.has_commitment:
        return None  # model reported nothing, or an unparseable report
    if not p.what.strip():
        return None  # a commitment with no description is not actionable
    try:
        datetime.strptime(p.when, "%Y-%m-%d")  # a reminder needs a real date
    except ValueError:
        return None
    if not p.evidence.strip() or p.evidence.strip() not in email_text:
        return None  # grounding: the date phrase must come from the email, not the model
    return p


# ============================================================================
# GLUE (tutor-written): the background scan that turns one-email proposals into a
# real behavior — read unread mail, dedup, cap, and queue the hits for approval.
# ============================================================================

SUGGEST_JOB_ID = "suggest-reminders"


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    """(date, lowercased text) already covered — so a re-scan doesn't re-propose the
    same thing. Covers BOTH approved reminders and suggestions still waiting in the
    Inbox, because either one means 'already handled, don't ask again'."""
    keys: set[tuple[str, str]] = set()
    with closing(sqlite3.connect(path)) as db:
        try:
            for text, date in db.execute("SELECT text, date FROM reminders"):
                keys.add((date, text.strip().lower()))
        except sqlite3.OperationalError:
            pass  # no reminders table yet (fresh DB) -> nothing to dedup against
    for pending in list_pending(path=path):
        if pending.name == "add_reminder":
            args = pending.args
            keys.add((args.get("date", ""), args.get("text", "").strip().lower()))
    return keys


def scan_inbox(
    *, model: str, today: str, cap: int = 5, path: Path = APP_DB_PATH
) -> int:
    """Read unread email, propose reminders, queue the new ones. Returns how many
    proposals were queued.

    Three precision protections beyond the kernel's own guard:
    - dedup: skip a proposal already covered by a reminder or a pending suggestion;
    - cap: never queue more than `cap` per run, so a flood of mail can't spam the Inbox;
    - taint: each proposal is marked tainted (it was derived from untrusted email), so
      the Inbox shows the 'read untrusted content' banner before you approve.
    """
    seen = _existing_keys(path)
    queued = 0
    unread = sorted(
        (e for e in tools._inbox() if not e["read"]),
        key=lambda e: e["timestamp"],
        reverse=True,
    )
    for email in unread:
        if queued >= cap:
            break
        email_text = tools._fmt_email(email)
        proposal = propose_reminder(email_text, today=today, model=model)
        if proposal is None:
            continue
        key = (proposal.when, proposal.what.strip().lower())
        if key in seen:
            continue  # already a reminder or already queued — don't ask twice
        queue_pending(
            SUGGEST_JOB_ID,
            f"Reminder: {proposal.what}",
            f"From email [{email['id']}] {email['subject']!r} — proposed a dated reminder.",
            "add_reminder",
            {"text": proposal.what, "date": proposal.when},
            [],  # no paused agent to resume: this is a direct action
            True,  # derived from untrusted email -> taint banner in the Inbox
            "",  # the proposal is self-contained; no agent run to link
            path=path,
        )
        seen.add(key)
        queued += 1
    log.info("suggestion scan queued %d proposal(s) from %d unread", queued, len(unread))
    return queued
