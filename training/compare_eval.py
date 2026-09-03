# -*- coding: utf-8 -*-
"""Compare two eval runs, separating accuracy from generation failure.

    python training/compare_eval.py training/eval_before.json training/eval_after.json

A mean overlap of 0.465 can mean "usually right" or "right half the time and
silent the other half". Those are different results and the summary table cannot
tell them apart, because a generation that produces nothing scores 0 and is
averaged in beside genuine wrong answers.

That distinction decided the post-training run: introspective and guessing
differed by +0.076 overall, but produced empty output at different rates. Divided
through by their success rates the two framings landed within 0.002 of each
other, so the apparent gap was reliability, not accuracy.

Everything here is computed from the per-row records; no GPU, runs instantly.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("A_introspective", "A_guessing", "B_introspective", "B_guessing")


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in data:
        raise SystemExit(f"{path.name} has no per-row records; rerun the eval.")
    return data


def bootstrap_ci(pairs: list[tuple[float, float]], reps: int = 10000,
                 seed: int = 17) -> tuple[float, float, float]:
    """Paired difference (a - b) with a percentile CI over resampled rows."""
    if not pairs:
        return 0.0, 0.0, 0.0
    diffs = [a - b for a, b in pairs]
    point = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean(rng.choices(diffs, k=n)) for _ in range(reps))
    return point, means[int(0.025 * reps)], means[int(0.975 * reps)]


def describe(rows: list[dict], label: str) -> None:
    print(f"\n{label}   (n = {len(rows)} rows)")
    print(f"  {'condition':<18}{'empty':>8}{'overlap':>10}{'overlap':>10}"
          f"{'precision':>11}{'mean n':>9}")
    print(f"  {'':<18}{'':>8}{'(all)':>10}{'|answered':>10}{'|answered':>11}"
          f"{'|answered':>9}")
    for cond in CONDITIONS:
        ns = [r[f"{cond}_n"] for r in rows]
        answered = [r for r in rows if r[f"{cond}_n"] > 0]
        empty = 1 - len(answered) / len(rows)
        overlap_all = mean(r[cond] for r in rows)
        if answered:
            print(f"  {cond:<18}{empty:>7.0%}{overlap_all:>10.3f}"
                  f"{mean(r[cond] for r in answered):>10.3f}"
                  f"{mean(r[f'{cond}_prec'] for r in answered):>11.3f}"
                  f"{mean(r[f'{cond}_n'] for r in answered):>9.1f}")
        else:
            print(f"  {cond:<18}{empty:>7.0%}{overlap_all:>10.3f}"
                  f"{'--':>10}{'--':>11}{'--':>9}")
    del ns


def gaps(rows: list[dict], label: str) -> None:
    """The measurement: introspective minus guessing, paired within a row."""
    print(f"\n{label} -- introspective MINUS guessing, paired per row")
    print(f"  {'target':<10}{'metric':<22}{'difference':>12}{'95% CI':>22}")
    for mode in ("A", "B"):
        i, g = f"{mode}_introspective", f"{mode}_guessing"
        both = [r for r in rows if r[f"{i}_n"] > 0 and r[f"{g}_n"] > 0]
        for metric, name, subset in (
            ("", "overlap, all rows", rows),
            ("", "overlap, both answered", both),
            ("_prec", "precision, both answered", both),
        ):
            pairs = [(r[i + metric], r[g + metric]) for r in subset]
            d, lo, hi = bootstrap_ci(pairs)
            flag = "" if lo <= 0 <= hi else "  *"
            print(f"  {'list ' + mode:<10}{name:<22}{d:>+12.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{flag}")
        print(f"  {'':<10}{'rows where both answered':<22}{len(both):>12} of {len(rows)}")
    print("\n  * = CI excludes zero. Rows where either framing produced nothing are")
    print("    dropped from the 'both answered' lines, so those compare accuracy")
    print("    rather than willingness to answer.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path, nargs="?",
                   default=REPO / "training" / "eval_before.json")
    p.add_argument("after", type=Path, nargs="?",
                   default=REPO / "training" / "eval_after.json")
    args = p.parse_args()

    before, after = load(args.before), load(args.after)
    for name, data in (("before", before), ("after", after)):
        cfg = data.get("config", {})
        print(f"{name:<7} {args.before.name if name == 'before' else args.after.name}"
              f"   format_hint={cfg.get('format_hint')} "
              f"no_thinking={cfg.get('no_thinking')} "
              f"stratify={cfg.get('stratify')} "
              f"adapter={cfg.get('adapter')}")
    if before.get("config", {}).get("format_hint") != after.get("config", {}).get("format_hint"):
        print("\nNOTE: the two runs used DIFFERENT prompts (format_hint differs).")
        print("That is defensible -- the base model has never seen the output format")
        print("and the trained model has -- but say so when reporting the comparison.")

    describe(before["rows"], "BEFORE (base model)")
    describe(after["rows"], "AFTER (fine-tuned)")

    # before/after on the same rows, same condition
    ids = {r["example_id"] for r in before["rows"]} & {r["example_id"] for r in after["rows"]}
    b = {r["example_id"]: r for r in before["rows"] if r["example_id"] in ids}
    a = {r["example_id"]: r for r in after["rows"] if r["example_id"] in ids}
    print(f"\nTRAINING EFFECT on the {len(ids)} rows both runs scored")
    print(f"  {'condition':<18}{'before':>9}{'after':>9}{'change':>10}{'95% CI':>22}")
    for cond in CONDITIONS:
        pairs = [(a[i][cond], b[i][cond]) for i in sorted(ids)]
        d, lo, hi = bootstrap_ci(pairs)
        print(f"  {cond:<18}{mean(x[1] for x in pairs):>9.3f}"
              f"{mean(x[0] for x in pairs):>9.3f}{d:>+10.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}")

    gaps(before["rows"], "BEFORE")
    gaps(after["rows"], "AFTER")


if __name__ == "__main__":
    main()
