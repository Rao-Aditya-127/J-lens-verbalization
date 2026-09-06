# -*- coding: utf-8 -*-
"""Is the base-vs-fine-tuned injection difference real, or noise?

    python training/inject_stats.py inject_base_clean.json inject_ft_clean.json

The summary tables disagree with each other. At each model's own best strength
the base model has the better median rank and top-100 rate, while the fine-tuned
model has the better top-10 rate -- on 8 versus 13 successes out of 100, which is
the kind of gap that appears and disappears between runs.

So the point estimates are not the question. What matters is whether any of these
differences survive their own uncertainty:

  * Mann-Whitney U on the rank distributions, which is the right test for ranks:
    heavily skewed, bounded below at 1, unbounded above.
  * bootstrap CIs on the top-k rates.
  * the within-model dose-response, which is the claim that does not depend on
    the comparison at all.

The two arms use different strength grids on purpose -- the base model peaks at
0.05 and breaks by 0.1, the fine-tuned model plateaus from 0.1 to 0.3 -- so they
are compared at each model's own optimum rather than at a shared number.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_strength: dict[float, list[dict]] = {}
    for t in data["trials"]:
        by_strength.setdefault(t["strength"], []).append(t)
    return {"label": data["config"]["model"], "by_strength": by_strength}


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """U statistic as the common-language effect size, plus a normal-approx p.

    Returns (P(a < b), p). For ranks, lower is better, so P(a < b) > 0.5 means
    the first sample ranks the injected concept higher.
    """
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return 0.5, 1.0
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks: dict[int, float] = {0: 0.0, 1: 0.0}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1                      # average rank over the tie block
        for k in range(i, j + 1):
            ranks[combined[k][1]] += avg
        i = j + 1
    u1 = ranks[0] - n1 * (n1 + 1) / 2
    p_a_less = 1 - u1 / (n1 * n2)                  # a < b means a has smaller ranks
    mu = n1 * n2 / 2
    sigma = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    z = abs(u1 - mu) / sigma if sigma else 0.0
    # two-sided normal approximation to the tail, no scipy dependency
    p = 2 * (1 - 0.5 * (1 + _erf(z / 2 ** 0.5)))
    return p_a_less, max(min(p, 1.0), 0.0)


def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26, |error| < 1.5e-7."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * pow(2.718281828459045, -x * x)
    return sign * y


def wilcoxon(diffs: list[float]) -> tuple[int, int, float]:
    """Signed-rank test on paired differences. Returns (n_pos, n_used, p).

    The two arms score the IDENTICAL rows with the IDENTICAL target word per row,
    so the design is paired and an unpaired test throws that away. Ranking |d|
    rather than d keeps it robust to ranks that span 1 to 250,000.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return sum(d > 0 for d in nz), n, 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = abs(w_pos - mu) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + _erf(z / 2 ** 0.5)))
    return sum(d > 0 for d in nz), n, max(min(p, 1.0), 0.0)


def sign_test(n_pos: int, n: int) -> float:
    """Exact two-sided binomial test at p=0.5 -- the most readable paired result."""
    from math import comb
    if n == 0:
        return 1.0
    total = 2 ** n
    k = min(n_pos, n - n_pos)
    tail = sum(comb(n, i) for i in range(k + 1)) / total
    return min(2 * tail, 1.0)


def boot_diff(a: list[bool], b: list[bool], reps: int = 20000,
              seed: int = 17) -> tuple[float, float, float]:
    """Difference in rate (a - b) with a percentile CI. Unpaired: different models."""
    rng = random.Random(seed)
    point = sum(a) / len(a) - sum(b) / len(b)
    diffs = sorted(
        sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b)
        for _ in range(reps))
    return point, diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def best_strength(arm: dict) -> float:
    """The strength with the lowest median rank, ignoring the 0 control."""
    doses = [s for s in arm["by_strength"] if s > 0]
    return min(doses, key=lambda s: median(t["rank"] for t in arm["by_strength"][s]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path, nargs="?", default=REPO / "inject_base_clean.json")
    p.add_argument("ft", type=Path, nargs="?", default=REPO / "inject_ft_clean.json")
    args = p.parse_args()

    base, ft = load(args.base), load(args.ft)

    print("=" * 78)
    print("1. DOSE-RESPONSE WITHIN EACH MODEL  (injected vs control, same model)")
    print("=" * 78)
    print("   The claim that needs no cross-model comparison: does the report track")
    print("   the activations at all? Prompt identical, only the residual differs.\n")
    print(f"  {'model':<12}{'strength':>10}{'median rank':>14}{'vs control':>13}"
          f"{'P(better)':>12}{'p':>10}")
    peaks = {}
    for arm in (base, ft):
        ctrl = [t["rank"] for t in arm["by_strength"][0.0]]
        s = best_strength(arm)
        peaks[arm["label"]] = s
        inj = [t["rank"] for t in arm["by_strength"][s]]
        eff, pv = mann_whitney(inj, ctrl)
        print(f"  {arm['label']:<12}{s:>10}{median(inj):>14.0f}"
              f"{median(ctrl):>13.0f}{eff:>12.3f}{pv:>10.2g}")
    print("\n  P(better) = probability a random injected trial ranks the concept above")
    print("  a random control trial. 0.5 is chance.")

    print("\n" + "=" * 78)
    print("2. BASE vs FINE-TUNED, each at its own best strength")
    print("=" * 78)
    b_s, f_s = peaks["base"], peaks["fine-tuned"]
    b = base["by_strength"][b_s]
    f = ft["by_strength"][f_s]
    print(f"  base @ {b_s}   vs   fine-tuned @ {f_s}\n")

    eff, pv = mann_whitney([t["rank"] for t in b], [t["rank"] for t in f])
    print(f"  {'median rank':<16}{median(t['rank'] for t in b):>10.0f}"
          f"{median(t['rank'] for t in f):>14.0f}")
    print(f"  {'':16}{'Mann-Whitney P(base ranks higher) = ':>44}{eff:.3f}   p = {pv:.3g}")

    print()
    for k in (10, 100, 1000):
        ba = [t["rank"] <= k for t in b]
        fa = [t["rank"] <= k for t in f]
        d, lo, hi = boot_diff(ba, fa)
        flag = "" if lo <= 0 <= hi else "   *"
        print(f"  top-{k:<12}{sum(ba) / len(ba):>10.0%}{sum(fa) / len(fa):>14.0%}"
              f"   difference {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{flag}")

    print("\n  * = CI excludes zero. Where it does not, the point estimate is not")
    print("    evidence of a difference however suggestive it looks.")

    # Paired, because the arms score the same rows with the same target words.
    b_by = {t["example_id"]: t for t in b}
    f_by = {t["example_id"]: t for t in f}
    shared = sorted(set(b_by) & set(f_by))
    same_target = all(b_by[i]["target"] == f_by[i]["target"] for i in shared)
    if shared:
        diffs = [f_by[i]["rank"] - b_by[i]["rank"] for i in shared]   # >0: base better
        n_pos, n_used, p_w = wilcoxon(diffs)
        p_s = sign_test(n_pos, n_used)
        print("\n" + "-" * 78)
        print(f"  PAIRED  ({len(shared)} rows scored by both arms; "
              f"same target word per row: {same_target})")
        print("-" * 78)
        print(f"  base ranked the concept higher on {n_pos} of {n_used} rows")
        print(f"    sign test        p = {p_s:.3g}")
        print(f"    Wilcoxon signed-rank  p = {p_w:.3g}")
        print("  The design is paired, so this is the test with the most power. The")
        print("  Mann-Whitney above ignores the pairing and is therefore conservative.")

    print("\n" + "=" * 78)
    print("3. FULL CURVES")
    print("=" * 78)
    for arm in (base, ft):
        print(f"\n  {arm['label']}")
        print(f"    {'strength':<10}{'median rank':>13}{'top-10':>9}{'top-100':>9}"
              f"{'top-1000':>10}")
        for s in sorted(arm["by_strength"]):
            rows = arm["by_strength"][s]
            print(f"    {s:<10}{median(t['rank'] for t in rows):>13.0f}"
                  f"{sum(t['rank'] <= 10 for t in rows) / len(rows):>8.0%}"
                  f"{sum(t['rank'] <= 100 for t in rows) / len(rows):>9.0%}"
                  f"{sum(t['rank'] <= 1000 for t in rows) / len(rows):>10.0%}")


if __name__ == "__main__":
    main()
