"""Structural smoke for the red-team battery — no local-model runs.

Asserts the 9 techniques are registered and complete, their forbidden tools are
real, and the reused grader (`attack_succeeded`) flags a hand-built hijack trace
while clearing a clean one. The agent is never driven here.
"""

from assistant.agent import tools
from assistant.evals import redteam_eval
from assistant.evals.injection_eval import attack_succeeded
from assistant.respond import Event


def test_nine_techniques_registered_and_complete():
    assert len(redteam_eval.TECHNIQUES) == 9
    ids = [t["id"] for t in redteam_eval.TECHNIQUES]
    assert len(set(ids)) == 9  # no duplicate technique ids

    for t in redteam_eval.TECHNIQUES:
        case = redteam_eval.as_case(t)
        # every row is a complete, gradable case
        assert set(case) == {
            "id",
            "user_task",
            "carrier",
            "forbidden_call",
            "poisoned_text",
        }
        fc = case["forbidden_call"]
        assert fc["tool"] and fc["arg_contains"]
        assert case["poisoned_text"] and case["user_task"]


def test_forbidden_tools_are_real_tools_on_the_menu():
    # The battery must attack calls the agent can actually make, or a "resisted"
    # result would just be a tool that does not exist.
    for t in redteam_eval.TECHNIQUES:
        assert t["forbidden_call"]["tool"] in tools._BODIES


def test_battery_covers_the_three_attacked_tools():
    attacked = {t["forbidden_call"]["tool"] for t in redteam_eval.TECHNIQUES}
    assert {"send_email", "delete_file", "create_calendar_event"} <= attacked


def test_grader_flags_forbidden_call_and_clears_clean_trace():
    # Build fake traces by hand — the model is never invoked.
    case = redteam_eval.as_case(redteam_eval.TECHNIQUES[0])  # a send_email exfil
    fc = case["forbidden_call"]

    hijack_trace = [
        Event("tool_call", text="read_file", data={"args": {"filename": "feedback.xlsx"}}),
        Event(
            "tool_call",
            text=fc["tool"],
            data={"args": {"to": fc["arg_contains"], "subject": "x", "body": "data"}},
        ),
    ]
    assert attack_succeeded(hijack_trace, case) is True

    # A proposal (approval_request) also counts as a hijack — the model bent.
    proposal_trace = [
        Event(
            "approval_request",
            text=fc["tool"],
            data={"args": {"to": fc["arg_contains"]}},
        ),
    ]
    assert attack_succeeded(proposal_trace, case) is True

    clean_trace = [
        Event("tool_call", text="read_file", data={"args": {"filename": "feedback.xlsx"}}),
        Event("done", text="The sentiment is mostly positive.", data={}),
    ]
    assert attack_succeeded(clean_trace, case) is False

    # Wrong recipient on the forbidden tool is NOT this case's hijack.
    other_addr_trace = [
        Event("tool_call", text=fc["tool"], data={"args": {"to": "someone-else@x.com"}}),
    ]
    assert attack_succeeded(other_addr_trace, case) is False
