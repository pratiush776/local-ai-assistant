"""Suggestion-trigger eval (lesson 9.4): precision & recall of `propose_reminder`
against the labelled commitments set.

    uv run evals --suggest           # 1 sample per email (fast)
    uv run evals --suggest --samples 3   # majority vote, to see the model's variance

`data/assets/evals/commitments.jsonl` is the answer key: 13 emails, each with
`has_commitment` true/false. We run the real trigger on each and compare.

    precision = of the reminders it PROPOSED, how many were real commitments.
                Low precision = noise, and a noisy assistant gets muted.
    recall    = of the real commitments, how many it CAUGHT.

For a proactive suggester we read PRECISION first: a missed commitment is cheaper
than a false proposal (9.3's precision-first bias), so precision is the one to defend.
"""

import json
from pathlib import Path

from ..agent import tools
from ..jobs.suggest import propose_reminder
from ..respond import PRIMARY_LOCAL_MODEL as MODEL

CASES = Path("data/assets/evals/commitments.jsonl")


def _emails_by_id() -> dict[str, dict]:
    return {str(e["id"]): e for e in tools._inbox()}


def run(samples: int = 1) -> dict:
    rows = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    by_id = _emails_by_id()

    tp = fp = fn = tn = 0
    mistakes: list[tuple[str, dict, str]] = []  # (kind, row, what the trigger did)

    print(f"\nSuggestion trigger vs commitments.jsonl  (model={MODEL}, samples={samples})\n")
    for r in rows:
        email = by_id.get(r["email_id"])
        if email is None:
            continue
        text = tools._fmt_email(email)
        hits = [
            propose_reminder(text, today=tools.PERSONA_TODAY, model=MODEL)
            for _ in range(samples)
        ]
        proposed = sum(p is not None for p in hits) * 2 > samples  # majority
        actual = bool(r["has_commitment"])

        if proposed and actual:
            tp += 1
            mark = "✅ TP"
        elif proposed and not actual:
            fp += 1
            mark = "❌ FP"
            mistakes.append(("FP", r, next(p.what for p in hits if p)))
        elif not proposed and actual:
            fn += 1
            mark = "❌ FN"
            mistakes.append(("FN", r, "(stayed silent)"))
        else:
            tn += 1
            mark = "✅ TN"
        print(f"  {mark}  [{r['email_id']:>2}] {r['subject'][:42]:<42}  label={actual}")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n  confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision = {precision:.2f}   recall = {recall:.2f}   f1 = {f1:.2f}\n")
    if mistakes:
        print("  mistakes to read (where it was wrong, and why the label says so):")
        for kind, r, did in mistakes:
            note = r["note"] or "—"
            print(f"    {kind} [{r['email_id']}] {r['subject'][:38]}: {did}  | note: {note}")
        print()

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "model": MODEL, "samples": samples,
    }
