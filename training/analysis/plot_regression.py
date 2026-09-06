# -*- coding: utf-8 -*-
"""Figure for the capability-regression result.

    python training/analysis/plot_regression.py

Both models, both prefixes, the same 30 held-out questions at the same
256-token budget. The per-row data below is transcribed from the two
probe_thinking.py SUMMARY tables, and the tallies are re-derived here by the
same rule the probe uses, so the figure and the write-up cannot drift apart.

The outcome categories are ORDERED by how far the model got toward answering,
so this is an ordinal ramp on one hue rather than a categorical palette.
Validated with `--ordinal`: monotone lightness, adjacent gaps >= 0.06, light
end clearing the surface at 2.06:1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

# blue ordinal ramp, steps 250 / 450 / 650 -- least to most question-directed
TRAINED, RUNNING, ANSWERED = "#86b6ef", "#2a78d6", "#104281"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"

LABELS = [
    "emitted the trained concept list, no reasoning",
    "question-directed text, still running at the 256-token cut",
    "reached an answer",
]
COLOURS = [TRAINED, RUNNING, ANSWERED]

# probe_thinking.py's own threshold. A closed </think> is not evidence of
# reasoning: 13 fine-tuned rows close the block on 16-21 characters, and a
# tag-matching rule scores every one of them as intact reasoning.
STUB_CHARS = 25

# (chars of reasoning, SUMMARY label) per row.
#   probe_thinking.py --rows 30 --tasks answer --max-new-tokens 256 [--base-only]
BASE_OFF = [(0, "answered directly")] * 30
BASE_ON = [(c, "reasoned, TRUNCATED") for c in (
    1138, 1126, 859, 1178, 943, 1050, 1104, 1151, 909, 1193,
    892, 1051, 1037, 1123, 1069, 1092, 1135, 794, 1168, 1065,
    1082, 1077, 1221, 1038, 1218, 1124, 1094, 1170, 1021, 1192,
)]
FT_OFF = [(0, "trained list")] * 30
FT_ON = [
    (19, "reasoned, then answered"), (920, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (16, "reasoned, then answered"),
    (155, "reasoned, then answered"), (1143, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (1071, "reasoned, TRUNCATED"),
    (19, "reasoned, then answered"), (1076, "reasoned, TRUNCATED"),
    (857, "reasoned, TRUNCATED"), (17, "reasoned, then answered"),
    (500, "reasoned, then answered"), (1101, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (1020, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (849, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (635, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (1140, "reasoned, TRUNCATED"),
    (914, "reasoned, then answered"), (1081, "reasoned, TRUNCATED"),
    (21, "reasoned, then answered"), (702, "reasoned, TRUNCATED"),
    (17, "reasoned, then answered"), (1171, "reasoned, TRUNCATED"),
    (819, "reasoned, then answered"), (21, "reasoned, then answered"),
]


def category(chars: int, label: str) -> int:
    """0 = trained list, 1 = still reasoning at the cut, 2 = reached an answer.

    "reasoned, then answered" splits on length: a block closed after 16-21
    characters is the model escaping the reasoning slot, not using it, and what
    follows is the trained concept list.
    """
    if label == "trained list":
        return 0
    if label == "answered directly":
        return 2
    if label == "reasoned, TRUNCATED":
        return 1
    if label == "reasoned, then answered":
        return 2 if chars >= STUB_CHARS else 0
    raise ValueError(label)


def tally(rows: list[tuple[int, str]]) -> list[int]:
    counts = [0, 0, 0]
    for chars, label in rows:
        counts[category(chars, label)] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "capability-regression" / "regression_figure.png")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        ("base\nthinking off", tally(BASE_OFF)),
        ("base\nthinking ON", tally(BASE_ON)),
        ("fine-tuned\nthinking off", tally(FT_OFF)),
        ("fine-tuned\nthinking ON", tally(FT_ON)),
    ]
    n = 30
    for label, counts in rows:
        assert sum(counts) == n, (label, counts)

    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 9, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
        "text.color": INK, "axes.titlecolor": INK,
    })
    fig, ax = plt.subplots(figsize=(9.4, 3.4))

    for i, (_, counts) in enumerate(rows):
        left = 0
        for count, colour in zip(counts, COLOURS):
            if not count:
                continue
            # 2px surface gap keeps adjacent segments legible without an outline
            ax.barh(i, count, left=left, height=0.6, color=colour,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(left + count / 2, i, str(count), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=INK if colour == TRAINED else SURFACE, zorder=4)
            left += count

    # A hairline between the two models: the comparison that matters is base
    # against fine-tuned at the same prefix, not all four bars at once.
    ax.axhline(1.5, color=GRID, lw=1.0, zorder=2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=9)
    ax.set_xlim(0, n)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("30 held-out questions — one user turn, no system prompt, "
                  "nothing asking for introspection")
    ax.set_title("Fine-tuning captured the model's default response path",
                 fontsize=11.5, loc="left", pad=12)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
    ax.legend(handles, LABELS, frameon=False, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.48, -0.26), ncol=1,
              handlelength=1.2, handleheight=1.0)

    # Axes coordinates, not figure coordinates: the legend above is anchored in
    # axes space, and mixing the two leaves a gap that grows with figure height.
    ax.text(0.48, -0.80,
            "Same questions, same 256-token budget, same greedy decoding, both "
            "models. The base model reaches an answer on every question and, "
            "given an open\n<think>, reasons on every one of them — its shortest "
            "block is 794 characters. The fine-tuned model answers none, and "
            "when the prefix is forced it\nescapes the block on 13 of 30 after "
            "16-21 characters. The two populations do not overlap.",
            ha="center", va="top", fontsize=7.6, color=INK_2,
            linespacing=1.5, transform=ax.transAxes)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.87, bottom=0.40)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")
    for label, counts in rows:
        flat = label.replace("\n", " ")
        print(f"  {flat:<26} trained={counts[0]:>2}  running={counts[1]:>2}  "
              f"answered={counts[2]:>2}")
    on = [c for c, _ in BASE_ON]
    print(f"\nbase thinking-ON reasoning: min {min(on)}  max {max(on)}  "
          f"median {sorted(on)[len(on) // 2]}")
    stubs = [c for c, l in FT_ON if category(c, l) == 0 and c]
    print(f"fine-tuned stubs: n={len(stubs)}  min {min(stubs)}  max {max(stubs)}")


if __name__ == "__main__":
    main()
