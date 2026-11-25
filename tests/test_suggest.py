"""9.3: the suggestion trigger proposes a reminder only for a grounded, dated
commitment, and the scan dedups + caps. Scripted model + a tmp DB, so these never
touch Ollama or the real app DB."""

from assistant.jobs import suggest
from assistant.jobs.pending import list_pending, queue_pending


def _tool_message(args: dict) -> dict:
    """A scripted model reply: one report_commitment tool-call with these arguments."""
    return {
        "tool_calls": [{"function": {"name": "report_commitment", "arguments": args}}]
    }


def _fake_chat(args: dict):
    def chat(messages, model, tools, options):
        return _tool_message(args)

    return chat


EMAIL = "[20] 2024-05-15  from: mark@x.com  —  Re: Hiking Trip\nMeet at 8 AM on Saturday 18th."


def test_proposes_grounded_dated_commitment(monkeypatch):
    monkeypatch.setattr(
        suggest,
        "chat_message",
        _fake_chat(
            {
                "has_commitment": True,
                "what": "Hiking with Mark",
                "when": "2024-05-18",
                "evidence": "Saturday 18th",  # a substring of EMAIL
            }
        ),
    )
    p = suggest.propose_reminder(EMAIL, today="2024-05-15", model="m")
    assert p is not None
    assert (p.what, p.when) == ("Hiking with Mark", "2024-05-18")


def test_drops_when_evidence_not_in_email(monkeypatch):
    # The model "quoted" a phrase that isn't in the email -> ungrounded -> dropped.
    monkeypatch.setattr(
        suggest,
        "chat_message",
        _fake_chat(
            {
                "has_commitment": True,
                "what": "Hiking with Mark",
                "when": "2024-05-20",
                "evidence": "Monday the 20th",  # NOT in EMAIL
            }
        ),
    )
    assert suggest.propose_reminder(EMAIL, today="2024-05-15", model="m") is None


def test_drops_when_date_does_not_parse(monkeypatch):
    monkeypatch.setattr(
        suggest,
        "chat_message",
        _fake_chat(
            {
                "has_commitment": True,
                "what": "Hiking with Mark",
                "when": "Saturday",  # not YYYY-MM-DD
                "evidence": "Saturday 18th",
            }
        ),
    )
    assert suggest.propose_reminder(EMAIL, today="2024-05-15", model="m") is None


def test_none_when_no_commitment(monkeypatch):
    monkeypatch.setattr(
        suggest,
        "chat_message",
        _fake_chat({"has_commitment": False, "what": "", "when": "", "evidence": ""}),
    )
    assert suggest.propose_reminder(EMAIL, today="2024-05-15", model="m") is None


def _email(eid: str, body: str) -> dict:
    return {
        "id": eid,
        "subject": f"subject {eid}",
        "body": body,
        "sender": "x@y.com",
        "read": False,
        "timestamp": f"2024-05-15T0{eid}:00",
    }


def test_scan_dedups_and_caps(monkeypatch, tmp_path):
    db = tmp_path / "assistant.db"
    # Three unread emails; the scan proposes one reminder per email.
    monkeypatch.setattr(
        suggest.tools, "_inbox", lambda: [_email("1", "a"), _email("2", "b"), _email("3", "c")]
    )
    proposals = {
        "subject 1": suggest.Proposal(has_commitment=True, what="A", when="2024-05-18"),
        "subject 2": suggest.Proposal(has_commitment=True, what="B", when="2024-05-19"),
        "subject 3": suggest.Proposal(has_commitment=True, what="C", when="2024-05-20"),
    }

    def fake_propose(email_text, *, today, model):
        for subj, p in proposals.items():
            if subj in email_text:
                return p
        return None

    monkeypatch.setattr(suggest, "propose_reminder", fake_propose)

    # Pre-queue "A" so it is already pending -> the scan must dedup it.
    queue_pending(
        suggest.SUGGEST_JOB_ID, "Reminder: A", "x", "add_reminder",
        {"text": "A", "date": "2024-05-18"}, [], True, "", path=db,
    )

    # cap=1 means: of the two NEW proposals (B, C), only one is queued this run.
    n = suggest.scan_inbox(model="m", today="2024-05-15", cap=1, path=db)
    assert n == 1

    pending = list_pending(path=db)
    assert len(pending) == 2  # the pre-queued A + exactly one new one
    new_texts = {p.args["text"] for p in pending}
    assert "A" in new_texts  # A was deduped, not re-added
    assert len(new_texts & {"B", "C"}) == 1  # cap held at one new proposal
