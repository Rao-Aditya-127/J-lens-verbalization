# -*- coding: utf-8 -*-
"""Figures for the activation-injection result.

    python training/plot_inject.py

Three panels, because the data has three jobs:

  (a) change over a continuous variable -> line. Median rank of the injected
      concept against injection strength. Log y (ranks span 26,000 to 93) and
      inverted, so upward always means the model ranks the concept higher.

  (b) tell two distributions apart -> line (ECDF). The medians in (a) hide the
      finding: the two rank distributions CROSS. Only the full distribution
      shows that the fine-tuned model is all-or-nothing while the base model is
      graded.

  (c) compare magnitudes at one point -> bars. Top-k rates at each model's own
      optimum, with bootstrap 95% CIs, so a reader can see which differences
      survive their uncertainty -- here they go in opposite directions.

Colours are validated categorical slots 1 and 2 (blue / orange): CVD Delta E 24.7
protan, 33.6 normal, all six checks pass. Both series are also directly labelled,
so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# parents[2]: this file sits at training/<group>/, so the repo root is two
# levels up. Every data and output path below is relative to it.
REPO = Path(__file__).resolve().parents[2]

BASE_C, FT_C = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SURFACE = "#fcfcfb"


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    by: dict[float, list[dict]] = {}
    for t in data["trials"]:
        by.setdefault(t["strength"], []).append(t)
    return data["config"]["model"], by


def boot_ci(flags, reps=10000, seed=17):
    rng = random.Random(seed)
    n = len(flags)
    draws = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(reps))
    return draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def best_strength(by):
    doses = [s for s in by if s > 0]
    return min(doses, key=lambda s: median(t["rank"] for t in by[s]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=REPO / "inject_base_clean.json")
    p.add_argument("--ft", type=Path, default=REPO / "inject_ft_clean.json")
    # Writes into the folder that references it, so regenerating updates the copy
    # the write-up actually shows rather than leaving a stray at the repo root.
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "activation-injection" / "inject_figure.png")
    args = p.parse_args()

    (_, base), (_, ft) = load(args.base), load(args.ft)
    arms = (("base model", base, BASE_C), ("fine-tuned", ft, FT_C))
    n = len({t["example_id"] for rows in base.values() for t in rows})

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 8.5, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK,
    })
    fig, (ax, cx, bx) = plt.subplots(1, 3, figsize=(14.0, 4.1),
                                     gridspec_kw={"width_ratios": [1.15, 1.05, 1]})

    # ---- (a) dose-response --------------------------------------------------
    for label, by, colour in arms:
        xs = sorted(s for s in by if s > 0)
        ys = [median(t["rank"] for t in by[s]) for s in xs]
        ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label,
                clip_on=False, zorder=3)
        i = ys.index(min(ys))
        left = colour == BASE_C
        ax.annotate(f"{label}\nbest: rank {ys[i]:,.0f}", (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=(-9, 10) if left else (11, 13),
                    ha="right" if left else "left",
                    fontsize=7.5, color=colour, fontweight="bold")

    ctrl = median([t["rank"] for t in base[0.0]] + [t["rank"] for t in ft[0.0]])
    ax.axhline(ctrl, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no injection  (rank ~{ctrl:,.0f})", (0.012, ctrl),
                textcoords="offset points", xytext=(0, 7), ha="left",
                fontsize=7.5, color=INK_2)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.32)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlabel("injection strength  (fraction of ‖h‖)")
    ax.set_ylabel("median rank of the injected concept\n(higher on the axis = better)")
    ax.set_title("a.  Both reports track the activations", fontsize=9.5, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # ---- (b) the full distributions, which cross ---------------------------
    for label, by, colour in arms:
        ranks = sorted(t["rank"] for t in by[best_strength(by)])
        ys = [(i + 1) / len(ranks) for i in range(len(ranks))]
        cx.step(ranks, ys, where="post", color=colour, lw=2, label=label, zorder=3)

    cx.set_xscale("log")
    cx.set_xlim(1, 100000)
    cx.set_ylim(0, 1.02)
    cx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cx.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cx.set_xticks([1, 10, 100, 1000, 10000, 100000])
    cx.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
    cx.set_xlabel("rank of the injected concept  (lower = better)")
    cx.set_ylabel("share of trials at or below that rank")
    cx.set_title("b.  All-or-nothing vs graded", fontsize=9.5, loc="left", pad=10)
    cx.grid(color=GRID, lw=0.6)
    cx.set_axisbelow(True)
    # Direct labels on the curves, in empty quadrants. The earlier prose
    # annotations sat in the middle and collided; the interpretation belongs in
    # the title and caption, not stacked on top of the marks.
    cx.text(14, 0.80, "base model", color=BASE_C, fontsize=8, fontweight="bold")
    cx.text(9000, 0.22, "fine-tuned", color=FT_C, fontsize=8, fontweight="bold",
            ha="right")

    # Where the two distributions cross is the whole point of the panel.
    b_ranks = sorted(t["rank"] for t in base[best_strength(base)])
    f_ranks = sorted(t["rank"] for t in ft[best_strength(ft)])
    cross = next((r for r in range(2, 2000)
                  if sum(x <= r for x in b_ranks) > sum(x <= r for x in f_ranks)), None)
    if cross:
        cx.axvline(cross, color=INK_2, lw=1, ls=(0, (2, 3)), zorder=1)
        cx.text(cross * 1.25, 0.03, f"curves cross\nat rank ~{cross}",
                fontsize=7.2, color=INK_2)

    # ---- (c) top-k at each model's own optimum ------------------------------
    ks = (10, 100, 1000)
    width, gap = 0.34, 0.045
    for j, (label, by, colour) in enumerate(arms):
        rows = by[best_strength(by)]
        rates = [sum(t["rank"] <= k for t in rows) / len(rows) for k in ks]
        cis = [boot_ci([t["rank"] <= k for t in rows]) for k in ks]
        xs = [i + (j - 0.5) * (width + gap) for i in range(len(ks))]
        bx.bar(xs, rates, width, color=colour, label=label, zorder=3)
        bx.errorbar(xs, rates, yerr=[[r - lo for r, (lo, _) in zip(rates, cis)],
                                     [hi - r for r, (_, hi) in zip(rates, cis)]],
                    fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3, zorder=4)
        for x, r, (_, hi) in zip(xs, rates, cis):
            bx.text(x, hi + 0.03, f"{r:.0%}", ha="center", fontsize=7.5, color=INK)

    bx.set_xticks(range(len(ks)))
    bx.set_xticklabels([f"top-{k}" for k in ks])
    bx.set_ylim(0, 1.05)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    bx.set_ylabel("trials reaching the top k")
    bx.set_title("c.  The gap reverses at top-10", fontsize=9.5, loc="left", pad=10)
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=8, loc="upper left")

    b_s, f_s = best_strength(base), best_strength(ft)
    fig.text(0.5, -0.05,
             f"n = {n} held-out questions, identical rows and target words in both arms. "
             f"Each model at its own best strength (base {b_s}, fine-tuned {f_s}), chosen "
             f"by lowest median rank; bars show bootstrap 95% CIs.\nThe prompt is "
             f"byte-identical at every strength, so a text-only predictor is unaffected "
             f"by construction.",
             ha="center", fontsize=7.2, color=INK_2)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")

    print(f"\nnumbers in the figure  (n = {n})")
    for label, by, _ in arms:
        s = best_strength(by)
        rows = by[s]
        print(f"  {label:<12} best strength {s}   median rank "
              f"{median(t['rank'] for t in rows):.0f}   " +
              "  ".join(f"top-{k} {sum(t['rank'] <= k for t in rows) / len(rows):.0%}"
                        for k in ks))


if __name__ == "__main__":
    main()
