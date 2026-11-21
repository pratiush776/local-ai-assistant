"""Score the 40-row golden set — the knowledge-layer scoreboard in one command.

    uv run evals            # run all rows, print the scoreboard, save data/evals/run-*.json

Two measurements, kept apart on purpose (see run_row):
  - retrieval hit@5  : pure retrieval via rag.rag_retrieve(), answerable rows only,
                       so a router punt is never mislabelled as a retrieval miss.
  - answer behaviour : the full system via respond(), for refusal accuracy + (next
                       session) the judge's answer pass rate.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from .. import rag
from ..respond import respond
from ..trace import record_turn
from . import checks
from .judge import HAIKU_JUDGE, LOCAL_JUDGE, agreement, judge_one
from .stats import mcnemar, wilson

GOLDEN = Path("data/assets/evals/golden.jsonl")
LABELS = Path("data/evals/labels.jsonl")
OUT_DIR = Path("data/evals")
BASELINE = Path("data/evals/baseline.json")
K = 5


def load_golden(limit: int | None = None) -> list[dict]:
    with GOLDEN.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def run_row(row: dict) -> dict:
    """One golden row -> the flat facts the checks need, from two sources."""
    q = row["question"]

    # (1) pure retrieval, isolated from the router
    retrieved = rag.rag_retrieve(q)  # (cid, text, score), best-first
    cids = [cid for cid, _, _ in retrieved]
    id2src = rag.sources_for_chunks(cids)
    ret_sources = [id2src.get(cid, "") for cid in cids]

    # (2) the real system answer, through respond() — time just this, the real turn
    t0 = time.perf_counter()
    rec = record_turn(q, list(respond(q, [])))
    latency_s = time.perf_counter() - t0

    return {
        "id": row["id"],
        "question": q,
        "answerable": row["answerable"],
        "expected_answer": row["expected_answer"],
        "expected_sources": row["source_files"],
        "retrieved_sources": ret_sources,
        "latency_s": round(latency_s, 2),
        "answer": rec["answer"],
        "route": rec["route"],
        "label": rec["label"],  # fact/chat/... — this, not route, says if it retrieved
        "not_found": rec["not_found"],
        "citations": rec["citations"],
        "retrieved_ids": [c["id"] for c in rec["retrieved"]],
    }


def score(results: list[dict]) -> dict:
    """Fold per-row checks into counts. hit@5 is answerable-only; refusal and
    citation run on every row."""
    answerable = [r for r in results if r["answerable"]]
    hits = sum(
        checks.hit_at_k(r["retrieved_sources"], r["expected_sources"], K)
        for r in answerable
    )
    refusal_ok = sum(
        checks.refusal_correct(r["answer"], r["not_found"], r["answerable"])
        for r in results
    )
    cite_ok = sum(
        checks.citation_valid(r["citations"], r["retrieved_ids"]) for r in results
    )
    passed = sum(1 for r in results if r.get("judge") == "PASS")
    return {
        "n": len(results),
        "n_answerable": len(answerable),
        "hit_at_k": [hits, len(answerable)],
        "pass_rate": [passed, len(results)],
        "refusal": [refusal_ok, len(results)],
        "citation": [cite_ok, len(results)],
    }


def print_scoreboard(s: dict) -> None:
    def line(label: str, pair: list[int]) -> None:
        k, n = pair
        p, lo, hi = wilson(k, n)
        print(f"  {label:<20} {p:6.1%}  [{lo:4.0%}, {hi:4.0%}]   ({k}/{n})")

    print("\n" + "=" * 58)
    print(f"GOLDEN SCOREBOARD   n={s['n']}   ({s['n_answerable']} answerable)")
    print("=" * 58)
    line("retrieval hit@5", s["hit_at_k"])
    line("answer pass rate", s["pass_rate"])
    line("refusal accuracy", s["refusal"])
    line("citation validity", s["citation"])
    print("=" * 58)


def label_rows() -> None:
    """Walk the unlabelled rows and record my PASS/FAIL call for each."""
    rows = [json.loads(line) for line in LABELS.open() if line.strip()]
    todo = [r for r in rows if not r.get("label", "").strip()]
    if not todo:
        print("All rows already labelled. Edit labels.jsonl to change one.")
        return
    print(f"{len(todo)} to label.  p=PASS  f=FAIL  s=skip  q=quit\n")
    for i, r in enumerate(todo, 1):
        print("=" * 72)
        print(f"[{i}/{len(todo)}] {r['id']}   Q: {r['question']}")
        print(f"  expected: {r['expected']}")
        print(
            f"  actual  : {r['actual']}"
        )  # full text: a truncated view hides late answers
        ans = input("  your call [p/f/s/q]> ").strip().lower()
        if ans == "q":
            break
        if ans in ("p", "f"):
            r["label"] = "PASS" if ans == "p" else "FAIL"
    with LABELS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = sum(1 for r in rows if r.get("label", "").strip())
    print(f"\nSaved. {done}/{len(rows)} rows labelled.")


def validate_judge(model: str = LOCAL_JUDGE) -> None:
    """Run the judge over my hand-labelled rows and score it against my labels.
    Trust gate (method 5.4): both TPR and TNR must clear 0.80."""
    rows = [json.loads(line) for line in LABELS.open() if line.strip()]
    rows = [r for r in rows if r.get("label", "").strip().upper() in ("PASS", "FAIL")]
    if not rows:
        print(f"No labelled rows in {LABELS} — fill the 'label' field with PASS/FAIL.")
        return

    labels = [r["label"].strip().upper() for r in rows]
    verdicts, reasons = [], []
    print(f"Judging {len(rows)} labelled rows with {model}…")
    for i, r in enumerate(rows, 1):
        v, why = judge_one(r["question"], r["expected"], r["actual"], model)
        verdicts.append(v)
        reasons.append(why)
        print(
            f"  [{i:2}/{len(rows)}] {r['id']}  you={labels[i - 1]:4} judge={v:4}",
            flush=True,
        )

    tpr, tnr = agreement(verdicts, labels)
    print("\n" + "=" * 58)
    print(f"JUDGE VALIDATION   ({model} vs my labels)")
    print("=" * 58)
    print(
        f"  TPR (agree on my PASS): {tpr:5.0%}   ({labels.count('PASS')} PASS labels)"
    )
    print(
        f"  TNR (agree on my FAIL): {tnr:5.0%}   ({labels.count('FAIL')} FAIL labels)"
    )
    gate = (
        "TRUST — both ≥ 0.80"
        if tpr >= 0.80 and tnr >= 0.80
        else "BELOW GATE — escalate to Haiku"
    )
    print(f"  verdict: {gate}")
    print("=" * 58)
    dis = [
        (r["id"], labels[i], verdicts[i], reasons[i])
        for i, r in enumerate(rows)
        if labels[i] != verdicts[i]
    ]
    if dis:
        print("\nDisagreements (you vs judge):")
        for rid, me, jv, why in dis:
            print(f"  {rid}: you={me} judge={jv}  — {why[:80]}")


def run_golden(rows: list[dict]) -> tuple[list[dict], float]:
    """Run every golden row through the system + local judge. Returns (results, seconds)."""
    print(f"Running {len(rows)} golden rows through the assistant…")
    results, t0 = [], time.perf_counter()
    for i, row in enumerate(rows, 1):
        r = run_row(row)
        v, _ = judge_one(r["question"], r["expected_answer"], r["answer"], LOCAL_JUDGE)
        r["judge"] = v
        results.append(r)
        print(
            f"  [{i:2}/{len(rows)}] {r['id']} label={r['label'] or '—':6} "
            f"judge={v:4} {row['question'][:40]}",
            flush=True,
        )
    return results, time.perf_counter() - t0


METRICS = ("hit_at_k", "pass_rate", "refusal", "citation")


def save_baseline(run_path: str) -> None:
    """Freeze a run's scores as the regression floor (the decided config: OFF, keep-10).
    Stores the k/n counts, so --check can derive each metric's Wilson lower bound."""
    d = json.loads(Path(run_path).read_text())
    s = d["score"]
    counts = {m: s[m] for m in METRICS}
    base = {
        "from": Path(run_path).name,
        "rerank": d.get("rerank"),
        "counts": counts,
        "median_lat_s": d.get("median_lat_s"),
    }
    BASELINE.write_text(json.dumps(base, indent=2))
    print(f"Saved baseline from {base['from']} (rerank {base['rerank']}):")
    for m, (k, n) in counts.items():
        p, lo, _ = wilson(k, n)
        print(f"  {m:<12} {p:6.1%}  Wilson floor {lo:5.1%}  ({k}/{n})")


def check_baseline() -> int:
    """Run the suite at current defaults and fail (exit 1) only if a metric falls
    BELOW the baseline's 95% Wilson lower bound — i.e. a drop too large to be the
    local model's run-to-run sampling noise. A point-vs-point compare would flag
    ordinary noise (a few rows flipping on n=40) as a regression; this doesn't.
    Local-only guard: CI keeps running pytest + ruff."""
    if not BASELINE.exists():
        print(
            f"No baseline at {BASELINE}. Run: uv run evals --save-baseline <run.json>"
        )
        return 1
    base = json.loads(BASELINE.read_text())
    s = score(run_golden(load_golden())[0])
    print("\n" + "=" * 64)
    print("REGRESSION CHECK   (now vs baseline's 95% Wilson floor)")
    print("=" * 64)
    failed = False
    for m, (bk, bn) in base["counts"].items():
        bp, lo, _ = wilson(bk, bn)
        nk, nn = s[m]
        now = nk / nn if nn else 0.0
        bad = now < lo
        failed = failed or bad
        print(
            f"  {m:<12} now {now:6.1%}  base {bp:6.1%}  floor {lo:6.1%}  "
            f"{'FAIL' if bad else 'ok'}"
        )
    print("=" * 64)
    print(
        "REGRESSION — a score fell below the baseline's noise floor"
        if failed
        else "OK — every score within the baseline's noise band"
    )
    return 1 if failed else 0


def compare_runs(path_a: str, path_b: str) -> None:
    """Pair two saved runs by row id, McNemar on PASS/FAIL, median latency each.
    A = first file, B = second. A 'win' is PASS (judge == 'PASS')."""
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    rows_a = {r["id"]: r for r in a["results"]}
    rows_b = {r["id"]: r for r in b["results"]}
    ids = [i for i in rows_a if i in rows_b]  # same 40, paired by id

    pa = [rows_a[i].get("judge") == "PASS" for i in ids]
    pb = [rows_b[i].get("judge") == "PASS" for i in ids]
    a_only, b_only_wrong, chi = mcnemar(
        pa, pb
    )  # a_only: A pass & B fail; b: A fail & B pass

    lat_a = statistics.median(rows_a[i].get("latency_s", 0) for i in ids)
    lat_b = statistics.median(rows_b[i].get("latency_s", 0) for i in ids)
    tag_a, tag_b = a.get("rerank", "A"), b.get("rerank", "B")

    print("\n" + "=" * 58)
    print(
        f"COMPARE  A={tag_a} ({Path(path_a).name})  vs  B={tag_b} ({Path(path_b).name})"
    )
    print("=" * 58)
    print(f"  pass rate   A {sum(pa):2}/{len(ids)}   B {sum(pb):2}/{len(ids)}")
    print(f"  flips       A-only-right {a_only}   B-only-right {b_only_wrong}")
    print(f"  McNemar     chi^2(cc) = {chi:.2f}   (>3.84 = significant at .05)")
    print(
        f"  median lat  A {lat_a:5.1f}s   B {lat_b:5.1f}s   (B-A = {lat_b - lat_a:+.1f}s)"
    )
    print("=" * 58)
    disagree = [i for i in ids if pa[ids.index(i)] != pb[ids.index(i)]]
    if disagree:
        print("Rows that flipped:")
        for i in disagree:
            wa = "PASS" if rows_a[i].get("judge") == "PASS" else "FAIL"
            wb = "PASS" if rows_b[i].get("judge") == "PASS" else "FAIL"
            print(f"  {i}: A={wa} B={wb}  {rows_a[i]['question'][:45]}")


def main() -> None:
    import sys

    if "--agent" in sys.argv:
        from . import agent_eval

        lim = (
            int(sys.argv[sys.argv.index("--limit") + 1])
            if "--limit" in sys.argv
            else None
        )
        agent_eval.run(lim, framework="--framework" in sys.argv)
        return
    if "--label" in sys.argv:
        label_rows()
        return
    if "--validate-judge" in sys.argv:
        model = HAIKU_JUDGE if "--haiku" in sys.argv else LOCAL_JUDGE
        validate_judge(model)
        return
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        compare_runs(sys.argv[i + 1], sys.argv[i + 2])
        return
    if "--save-baseline" in sys.argv:
        save_baseline(sys.argv[sys.argv.index("--save-baseline") + 1])
        return
    if "--check" in sys.argv:
        sys.exit(check_baseline())

    # experiment knobs: force the reranker on/off and cap the kept-set (default = rag.py)
    if "--rerank" in sys.argv:
        mode = sys.argv[sys.argv.index("--rerank") + 1]
        rag.RERANK_ENABLED = mode == "on"
        if rag.RERANK_ENABLED:
            rag.warmup_reranker()  # load the model now, off the per-turn clock
    if "--keep-n" in sys.argv:
        rag.KEEP_N = int(sys.argv[sys.argv.index("--keep-n") + 1])
    rerank_tag = "on" if rag.RERANK_ENABLED else "off"
    if rag.KEEP_N is not None:
        rerank_tag += f"-k{rag.KEEP_N}"

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    rows = load_golden(limit)

    results, dur = run_golden(rows)

    s = score(results)
    median_lat = statistics.median(r["latency_s"] for r in results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"run-{rerank_tag}-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "ts": stamp,
                "dur_s": round(dur, 1),
                "rerank": rerank_tag,
                "median_lat_s": round(median_lat, 2),
                "score": s,
                "results": results,
            },
            indent=2,
        )
    )
    print_scoreboard(s)
    print(f"  {'median turn latency':<20} {median_lat:5.1f}s   (rerank {rerank_tag})")
    print("=" * 58)
    print(f"\nSaved {out}  ({dur:.0f}s)")
    if LABELS.exists() and not limit and "--rerank" not in sys.argv:
        validate_judge(LOCAL_JUDGE)  # judge agreement, against my hand labels


if __name__ == "__main__":
    main()
