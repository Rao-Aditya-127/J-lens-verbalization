# -*- coding: utf-8 -*-
"""Plot loss curves from a finished run.

    python training/plot_training.py training/runs/qlora-v1/final/log_history.json

Works with no tracker configured and needs nothing but matplotlib, so the curves
are recoverable from a pod that has already been terminated as long as
log_history.json was downloaded. Produces a PNG suitable for the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("history", type=Path, help="log_history.json from a run")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    log = json.loads(args.history.read_text(encoding="utf-8"))
    train = [(h["step"], h["loss"]) for h in log if "loss" in h and "step" in h]
    evals = [(h["step"], h["eval_loss"]) for h in log if "eval_loss" in h]
    if not train:
        raise SystemExit("no training loss found in that history file")

    out = args.out or args.history.with_name("loss_curve.png")
    plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(*zip(*train), color="#2b6cb0", lw=1.3, label="train")
    if evals:
        ax.plot(*zip(*evals), color="#c53030", lw=1.6, marker="o", ms=3.5, label="validation")
        best_step, best_loss = min(evals, key=lambda t: t[1])
        ax.axvline(best_step, color="#c53030", ls=":", lw=1)
        ax.annotate(f"best {best_loss:.3f}\n@ step {best_step}", (best_step, best_loss),
                    textcoords="offset points", xytext=(8, 14), fontsize=7.5, color="#c53030")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (target tokens only)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("QLoRA SFT — J-lens concept reporting", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")

    print(f"\ntrain  {train[0][1]:.4f} -> {train[-1][1]:.4f}   ({len(train)} points)")
    if evals:
        best_step, best_loss = min(evals, key=lambda t: t[1])
        print(f"eval   {evals[0][1]:.4f} -> {evals[-1][1]:.4f}   best {best_loss:.4f} @ step {best_step}")
        if evals[-1][1] > best_loss * 1.02:
            print("\nvalidation ended above its best: the model overfit after that point.")
            print(f"the checkpoint at step {best_step} would likely evaluate better than the final one.")


if __name__ == "__main__":
    main()
