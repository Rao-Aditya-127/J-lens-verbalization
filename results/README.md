# Results

One folder per experiment. Each holds a `README.md` with the full write-up —
what the question was, how it was set up, which code produced it, what the
numbers mean and where they fall short — plus that experiment's own figures.

| experiment | question | headline |
|---|---|---|
| [activation-injection](activation-injection/) | Does the model notice a concept injected into its workspace, and does fine-tuning help? | Both models' reports causally track their activations (median rank 25,942 → 93 and 19,896 → 672, p ≈ 0). Fine-tuning made that sensitivity **all-or-nothing**: it reaches the top-10 more often (9% vs 2%) yet fails outright far more often (24% of trials still above rank 5,000, vs 1%). The base model ranks the concept higher on **63 of 100** rows, Wilcoxon p = 5.2e-07. |

## Conventions

- **Figures live beside the write-up that uses them**, referenced by relative
  path, so a folder can be read or moved on its own.
- **Every number is traceable** to a JSON in the repo root or to the command
  printed in the write-up's *Reproducing* section. The result JSONs themselves
  are gitignored — the repo stays code-only and the data lives on
  [Hugging Face](https://huggingface.co/datasets/RaoAditya/j-lens-verbalization).
- **Failed attempts are written up too.** In this project the dead ends were
  usually more informative than the final run: four injection configurations
  returned zero before the fifth worked, and the reason was the measurement, not
  the model.
