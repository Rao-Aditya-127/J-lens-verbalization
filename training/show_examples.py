# -*- coding: utf-8 -*-
"""Print what the model actually said, next to what the lens actually recorded.

    python training/show_examples.py training/eval_after_nohint.json
    python training/show_examples.py training/eval_after_nohint.json \
        --before training/eval_before.json --n 5

Scores compress a 15-item list into one number and hide everything worth
looking at: whether misses are near-misses ("variable" vs "variables"), whether
the two framings produce similar lists or different ones, whether the model is
naming concepts from the text or genuinely novel ones.

Reads the per-row records written by eval_sft.py and joins them against
collected_answers.jsonl for the targets. No GPU, no model load. Rows are drawn
with a fixed seed so the examples are not cherry-picked -- say so when quoting
them.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

# J-lens returns tokens in every language the model represents -- 计算, 情感 and
# similar appear throughout the targets. A cp1252 console (Windows default)
# raises on them, so ask for UTF-8 and degrade rather than crash if refused.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - older/odd streams
        pass

REPO = Path(__file__).resolve().parents[1]
COLLECTED = REPO / "dataset" / "jlens" / "collected_answers.jsonl"
WIDTH = 96


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(" ".join(text.split()), width=WIDTH,
                         initial_indent=indent, subsequent_indent=indent)


def marked(preds: list[str], truth: set[str]) -> str:
    """+ for a hit, - for a miss, so it reads without colour or unicode."""
    return "  ".join(("+" if p in truth else "-") + p for p in preds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, help="eval JSON from eval_sft.py")
    p.add_argument("--before", type=Path, default=None,
                   help="a second eval JSON to show alongside (e.g. the base model)")
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--n", type=int, default=6, help="examples to show")
    p.add_argument("--seed", type=int, default=1234, help="fixed, so nothing is cherry-picked")
    p.add_argument("--mode", choices=["A", "B", "both"], default="both")
    p.add_argument("--source", default=None, help="restrict to one dataset, e.g. gsm8k")
    args = p.parse_args()

    truths = {}
    for line in args.collected.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            truths[row["example_id"]] = row

    after = json.loads(args.results.read_text(encoding="utf-8"))
    rows = after["rows"]
    before = None
    if args.before:
        before = {r["example_id"]: r
                  for r in json.loads(args.before.read_text(encoding="utf-8"))["rows"]}

    if args.source:
        rows = [r for r in rows if r.get("source") == args.source]
    if not rows:
        raise SystemExit("no rows matched")

    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    modes = ("A", "B") if args.mode == "both" else (args.mode,)
    cfg = after.get("config", {})
    print(f"{args.results.name}: adapter={cfg.get('adapter')} "
          f"format_hint={cfg.get('format_hint')}")
    print(f"{len(picked)} examples, seed {args.seed} -- randomly drawn, not selected\n")

    for rec in picked:
        src = truths.get(rec["example_id"])
        if src is None:
            continue
        print("=" * WIDTH)
        print(f"{rec['example_id']}")
        print("=" * WIDTH)
        print("  QUESTION")
        print(wrap(src["question"][:600]))
        print("  ANSWER (what the lens was measured on)")
        print(wrap(src["answer"][:600] + ("..." if len(src["answer"]) > 600 else "")))

        text = (src["question"] + " " + src["answer"]).lower()
        for mode in modes:
            key = "j_lens_top10" if mode == "A" else "j_lens_top10_novel"
            truth = [c["concept"].strip().lower() for c in src[key]]
            in_text = sum(1 for t in truth if t in text)
            label = "top-15 most active" if mode == "A" else "top-15 absent from the text"
            print(f"\n  LIST {mode} -- {label}"
                  f"   ({in_text}/15 of these appear in the text above)")
            print(wrap(", ".join(truth), "    "))

            tset = set(truth)
            for framing in ("introspective", "guessing"):
                k = f"{mode}_{framing}"
                if k not in rec:
                    continue
                line = f"    {framing:<14} {rec[k]:.2f}"
                if before and rec["example_id"] in before:
                    line += f"   (base model: {before[rec['example_id']][k]:.2f})"
                print("\n" + line)
                preds = rec.get(f"{k}_pred", [])
                print(wrap(marked(preds, tset) if preds
                           else "(produced nothing parseable)", "      "))
        print()

    print("+ = matched the lens readout,  - = did not")
    print("For list B every prediction found in the text is wrong by construction,")
    print("so a '-' there may still be the model copying rather than reporting.")


if __name__ == "__main__":
    main()
