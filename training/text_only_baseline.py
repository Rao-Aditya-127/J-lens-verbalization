# -*- coding: utf-8 -*-
"""How much of the J-lens readout is predictable from the TEXT alone?

    python training/text_only_baseline.py

List B contains only concepts that appear nowhere in the question or the answer,
so a model scoring well on it cannot be copying. That rules out copying. It does
not rule out prediction: "absent from the text" and "unpredictable from the text"
are different properties, and a concept like `photons` for a question about
sunglasses is absent yet entirely foreseeable.

This builds a predictor that has NO access to any activations, ever:

    tf-idf over question+answer  ->  k nearest TRAINING questions
                                ->  their most common list-B concepts

Whatever it scores is reachable from the text by a method with no introspective
access of any kind. It is a lower bound on the text-only ceiling, not the ceiling
itself -- a 27B model is a far better text predictor than nearest neighbours --
so it bounds the trivially-text-derivable component rather than the whole of it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]
COLLECTED = REPO / "dataset" / "jlens" / "collected_answers.jsonl"
WORD = re.compile(r"[a-z0-9']+")


def tokens(row: dict) -> list[str]:
    return WORD.findall((row["question"] + " " + row["answer"]).lower())


def targets(row: dict, mode: str) -> set[str]:
    key = "j_lens_top10" if mode == "A" else "j_lens_top10_novel"
    return {c["concept"].strip().lower() for c in row[key]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collected", type=Path, default=COLLECTED)
    p.add_argument("--k", type=int, default=15, help="neighbours to pool over")
    p.add_argument("--top", type=int, default=15, help="concepts to predict")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"{len(train)} training rows, {len(test)} test rows\n")

    # tf-idf over the training questions, with an inverted index so scoring a
    # test row touches only the documents that share a term with it
    docs = [Counter(tokens(r)) for r in train]
    df = Counter(t for d in docs for t in d)
    n = len(docs)
    idf = {t: math.log(n / (1 + c)) for t, c in df.items()}

    index: dict[str, list[tuple[int, float]]] = defaultdict(list)
    norms = []
    for i, d in enumerate(docs):
        vec = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in d.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        norms.append(norm)
        for t, v in vec.items():
            index[t].append((i, v / norm))

    for mode, label in (("A", "list A (top-15 most active)"),
                        ("B", "list B (absent from the text)")):
        truth = [targets(r, mode) for r in test]

        # question-blind floor: the globally most common concepts, always
        freq = Counter(c for r in train for c in targets(r, mode))
        const = {w for w, _ in freq.most_common(args.top)}
        const_score = mean(len(const & t) / args.top for t in truth)

        # text-only k-NN
        scores = []
        for row, t in zip(test, truth):
            q = Counter(tokens(row))
            qvec = {w: (1 + math.log(c)) * idf.get(w, 0.0) for w, c in q.items() if w in idf}
            qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
            sim: dict[int, float] = defaultdict(float)
            for w, v in qvec.items():
                for i, dv in index[w]:
                    sim[i] += (v / qnorm) * dv
            near = sorted(sim.items(), key=lambda kv: -kv[1])[: args.k]
            pool: Counter = Counter()
            for i, s in near:
                for c in targets(train[i], mode):
                    pool[c] += s
            pred = {w for w, _ in pool.most_common(args.top)}
            scores.append(len(pred & t) / args.top)

        print(f"{label}")
        print(f"  constant baseline (question-blind)   {const_score:.3f}")
        print(f"  text-only k-NN  (k={args.k}, no activations)   {mean(scores):.3f}")

    print("\nThe k-NN predictor never sees an activation. Whatever it reaches is")
    print("obtainable from the text by a method with no introspective access.")
    print("It is a LOWER bound on what text alone affords: a 27B model predicts")
    print("from text far better than nearest neighbours do.")


if __name__ == "__main__":
    main()
