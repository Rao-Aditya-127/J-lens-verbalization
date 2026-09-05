# -*- coding: utf-8 -*-
"""Figures for the activation-injection result.

    python training/plot_inject.py

Two panels, because the data has two jobs:

  (a) change over a continuous variable -> a line. Median rank of the injected
      concept against injection strength, for both models. The y-axis is
      logarithmic (ranks span 26,000 to 66) and inverted, so upward always means
      the model ranks the concept higher.

  (b) comparison of magnitudes at one point -> bars. Top-k hit rates at each
      model's own optimum, with bootstrap 95% CIs, so a reader can see which
      differences survive their uncertainty and which do not.

Colours are the validated categorical slots 1 and 2 (blue / orange), which pass
the CVD, chroma, lightness and contrast checks; both series are also directly
labelled, so identity never rests on colour alone.
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

REPO = Path(__file__).resolve().parents[1]

BASE_C, FT_C = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


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
    # Writes straight into the folder that references it, so regenerating updates
    # the copy the write-up actually shows rather than leaving a stray at the root.
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "activation-injection" / "inject_figure.png")
    args = p.parse_args()

    (_, base), (_, ft) = load(args.base), load(args.ft)
    arms = (("base model", base, BASE_C), ("fine-tuned", ft, FT_C))

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 8.5, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- (a) dose-response -------------------------------------------------
    for label, by, colour in arms:
        xs = sorted(s for s in by if s > 0)
        ys = [median(t["rank"] for t in by[s]) for s in xs]
        ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=5.5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=label,
                clip_on=False, zorder=3)
        # direct label at the series' best point, so identity is not colour-alone.
        # Opposite sides of their own peaks, or the two labels overlap.
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
    ax.invert_yaxis()                    # upward = ranked higher = better
    ax.set_xlim(0, 0.32)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlabel("injection strength  (fraction of ‖h‖ added along the concept direction)")
    ax.set_ylabel("median rank of the injected concept\n(higher on the axis = better)")
    ax.set_title("a.  The report tracks the activations, in both models",
                 fontsize=9.5, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # ---- (b) top-k at each model's own optimum -----------------------------
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
    bx.set_ylabel("trials where the injected concept\nreached the top k")
    bx.set_title("b.  Training reduced that sensitivity", fontsize=9.5, loc="left", pad=10)
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=8, loc="upper left")

    b_s, f_s = best_strength(base), best_strength(ft)
    # n comes from the data, never hardcoded: a row whose randomly-chosen target is
    # not a single token is skipped, so --rows 100 actually ran 63.
    n = len({t["example_id"] for rows in base.values() for t in rows})
    fig.text(0.512, -0.055,
             f"n = {n} held-out questions, identical rows and targets in both arms. "
             f"Each model at its own best strength "
             f"(base {b_s}, fine-tuned {f_s}), chosen by lowest median rank. "
             f"Bars show bootstrap 95% CIs.\nThe prompt is byte-identical at every "
             f"strength, so a text-only predictor is unaffected by construction.",
             ha="center", fontsize=7.2, color=INK_2)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {args.out}")

    print("\nnumbers in the figure")
    for label, by, _ in arms:
        s = best_strength(by)
        rows = by[s]
        print(f"  {label:<12} best strength {s}   median rank "
              f"{median(t['rank'] for t in rows):.0f}   " +
              "  ".join(f"top-{k} {sum(t['rank'] <= k for t in rows) / len(rows):.0%}"
                        for k in ks))


if __name__ == "__main__":
    main()
