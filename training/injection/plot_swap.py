# -*- coding: utf-8 -*-
"""Figure for the concept-swap dose-response.

    python training/injection/plot_swap.py

Reads `injection_curve_results.json` -- the raw output of the 25 x 6 run against
Neuronpedia's API -- and re-derives every rate, interval and p-value from it, so
nothing in the write-up is transcribed by hand.

The experiment is base-model only: `swapToken` is an API parameter and the API
hosts the base model, not the adapter. Its dose axis is the WIDTH of the swapped
layer band, because `swapToken` has no strength parameter -- which is exactly
why the base-vs-fine-tuned comparison had to be built on `steer` instead. See
`results/concept-swap/` and `results/activation-injection/`.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]

BASE_C = "#2a78d6"                       # validated categorical slot 1
LEAK_C = "#eb6834"                       # validated categorical slot 2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SURFACE = "#fcfcfb"

DOSES = ["narrow", "small", "wide", "full"]
K = 15                                   # the model is asked for 15 concepts


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval, by inverting the binomial CDF on a grid.

    scipy is not a dependency of this repo (see inject_stats.py), and a normal
    approximation is wrong at exactly the rates that matter here -- it puts a
    symmetric interval around 0/25, which is not a thing.
    """
    def cdf_at_least(p: float, k: int, n: int) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def cdf_at_most(p: float, k: int, n: int) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(target: float, side: str) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            got = cdf_at_least(mid, k, n) if side == "lower" else cdf_at_most(mid, k, n)
            if (got < target) == (side == "lower"):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(alpha / 2, "lower")
    high = 1.0 if k == n else solve(alpha / 2, "upper")
    return low, high


def _hyper(i: int, n: int, row1: int, col1: int) -> float:
    return comb(row1, i) * comb(n - row1, col1 - i) / comb(n, col1)


def fisher(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Fisher exact on the 2x2 [[a, b], [c, d]]. Returns (one-sided, two-sided).

    Two-sided is the conventional sum over every table no more probable than the
    observed one, not the one-sided value doubled. The two happen to differ by
    almost exactly 2x here, which is how the earlier 8.6e-05 in the decision log
    arose -- it is the two-sided number.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    one = sum(_hyper(i, n, row1, col1) for i in range(a, hi + 1))
    obs = _hyper(a, n, row1, col1)
    two = sum(_hyper(i, n, row1, col1) for i in range(lo, hi + 1)
              if _hyper(i, n, row1, col1) <= obs * (1 + 1e-9))
    return one, two


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path,
                   default=REPO / "injection_curve_results.json")
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "concept-swap" / "swap_figure.png")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    n = len(rows)

    layers, rates, cis, ranks = [], [], [], {}
    for dose in DOSES:
        cells = [r["doses"][dose] for r in rows]
        hits = [c for c in cells if c["hit"]]
        layers.append(cells[0]["n_layers"])
        rates.append(len(hits) / n)
        cis.append(clopper_pearson(len(hits), n))
        ranks[dose] = [c["rank"] for c in hits if c["rank"] is not None]

    base_hits = sum(1 for r in rows if r["baseline_hit"])
    leaks = sum(1 for r in rows if r["leak"])
    full_hits = sum(1 for r in rows if r["doses"]["full"]["hit"])
    clean = sum(1 for r in rows if r["doses"]["full"]["hit"] and not r["leak"])
    p_one, p_two = fisher(full_hits, n - full_hits, base_hits, n - base_hits)

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 9, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK,
    })
    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(11.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1.0, 0.95]})

    # --- A. dose-response -------------------------------------------------
    lo = [r - c[0] for r, c in zip(rates, cis)]
    hi = [c[1] - r for r, c in zip(rates, cis)]
    ax.errorbar(layers, rates, yerr=[lo, hi], fmt="none",
                ecolor=INK_2, elinewidth=1.1, capsize=3, zorder=3)
    ax.plot(layers, rates, "-o", color=BASE_C, lw=1.8, ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    # The control is a rate, not a point on this axis -- a flat reference line
    # is the honest way to draw "0/25 however many layers you swap".
    ax.axhline(base_hits / n, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    # Below the line, not beside it: at 0% the label would otherwise land inside
    # the first two error bars.
    ax.text(1.5, -0.055, f"no-swap baseline  {base_hits}/{n}",
            fontsize=7.5, color=INK_2, va="center", ha="left")
    # Anchored above each interval's cap rather than its point, so the 0/25
    # doses -- whose intervals are the tall ones -- do not sit inside them.
    for x, y, ci in zip(layers, rates, cis):
        ax.annotate(f"{y:.0%}", (x, ci[1]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8,
                    fontweight="bold", color=INK)
    ax.set_xticks(layers)
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.09, 0.80)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("layers swapped  (band centred on the workspace midpoint)")
    ax.set_ylabel("injected concept reported")
    ax.set_title("A. Detection rises with the width of the swap",
                 fontsize=9.5, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # --- B. rank tracks dose ---------------------------------------------
    shown = [d for d in DOSES if ranks[d]]
    for i, dose in enumerate(shown):
        vals = ranks[dose]
        # jitter-free: stack duplicates sideways so every trial is visible
        seen: dict[int, int] = {}
        for v in sorted(vals):
            off = seen.get(v, 0)
            seen[v] = off + 1
            bx.plot(i + (off - 0.0) * 0.075, v, "o", color=BASE_C, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        mean = sum(vals) / len(vals)
        bx.plot([i - 0.28, i + 0.28], [mean, mean], color=INK, lw=1.6, zorder=4)
        bx.text(i + 0.32, mean, f"mean {mean:.1f}", fontsize=7.5,
                color=INK, va="center")
    bx.set_xticks(range(len(shown)))
    bx.set_xticklabels([f"{d}\n{ranks and dict(zip(DOSES, layers))[d]} layers"
                        for d in shown])
    bx.set_xlim(-0.5, len(shown) - 0.15)
    bx.set_ylim(K + 0.5, 0.2)          # rank 1 at the top
    bx.set_yticks([1, 5, 10, 15])
    bx.set_ylabel("rank in the model's 15-concept list")
    bx.set_title("B. And the concept is placed higher", fontsize=9.5,
                 loc="left", pad=10)
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.set_axisbelow(True)

    # --- C. leak accounting ----------------------------------------------
    missed = n - full_hits
    parts = [(clean, BASE_C, "reported, no leak"),
             (full_hits - clean, LEAK_C, "reported, but also leaked"),
             (missed, GRID, "not reported")]
    left = 0
    for count, colour, _ in parts:
        cx.barh(0, count, left=left, height=0.5, color=colour,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        if count:
            cx.text(left + count / 2, 0, str(count), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=INK if colour == GRID else SURFACE, zorder=4)
        left += count
    cx.set_xlim(0, n)
    cx.set_ylim(-0.42, 0.42)
    cx.set_yticks([])
    cx.spines["left"].set_visible(False)
    cx.set_xticks([0, 5, 10, 15, 20, 25])
    cx.set_xlabel(f"the {n} trials at the full dose")
    cx.set_title("C. The hits are not spillover", fontsize=9.5, loc="left", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in parts]
    cx.legend(handles, [lab for _, _, lab in parts], frameon=False, fontsize=7.8,
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1,
              handlelength=1.2, handleheight=1.0)

    fig.text(0.5, -0.10,
             f"Base model through Neuronpedia's API, n = {n} rows. A source concept "
             f"in the readout is replaced by an unrelated target verified absent from "
             f"the question, the answer and the row's\nentire 250-deep readout, so no "
             f"guessing route to it exists. Full dose vs the no-swap baseline: "
             f"Fisher exact p = {p_two:.1e} (two-sided). Intervals are Clopper-Pearson.",
             ha="center", fontsize=7.6, color=INK_2, linespacing=1.5)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.30, wspace=0.42)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}\n")

    for dose, lay, rate, ci in zip(DOSES, layers, rates, cis):
        k = round(rate * n)
        mean = (sum(ranks[dose]) / len(ranks[dose])) if ranks[dose] else float("nan")
        print(f"  {dose:<7} {lay:>3} layers   {k:>2}/{n}  {rate:6.1%}   "
              f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]   mean rank {mean:.2f}")
    print(f"\n  baseline {base_hits}/{n}   leak {leaks}/{n}   "
          f"clean full-dose hits {clean}/{n}")
    print(f"  Fisher exact, full vs baseline: "
          f"p = {p_two:.3e} two-sided, {p_one:.3e} one-sided")


if __name__ == "__main__":
    main()
