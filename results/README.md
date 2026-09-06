# Results

One folder per experiment. Each holds a `README.md` with the full write-up —
what the question was, how it was set up, which code produced it, what the
numbers mean and where they fall short — plus that experiment's own figures.

| experiment | question | headline |
|---|---|---|
| [concept-swap](concept-swap/) | With the prompt held fixed, does swapping a concept inside the workspace change what the model says is active? | Yes. A target verified absent from the question, the answer and the row's 250-deep readout is named **12 of 25** times at the full dose against **0 of 25** with no swap (Fisher exact p = 8.6e-05). Detection is monotonic in the width of the swapped band (0% → 0% → 8% → 48%) and the reported rank tracks dose (7.0 → 4.8). Leakage into an unrelated answer: 2/25. Base model only — `swapToken` is API-only. |
| [activation-injection](activation-injection/) | Does the model notice a concept injected into its workspace, and does fine-tuning help? | Both models' reports causally track their activations (median rank 25,942 → 93 and 19,896 → 672, p ≈ 0). Fine-tuning made that sensitivity **all-or-nothing**: it reaches the top-10 more often (9% vs 2%) yet fails outright far more often (24% of trials still above rank 5,000, vs 1%). The base model ranks the concept higher on **63 of 100** rows, Wilcoxon p = 5.2e-07. |
| [capability-regression](capability-regression/) | What did two epochs of narrow SFT cost the model outside the trained task? | A matched 2 × 2 of model × `<think>` prefix on 30 held-out questions. Asked an ordinary question — no system prompt, nothing about introspection — the fine-tuned model emits its trained concept list **30 of 30** and answers **0**; the base model answers **30 of 30**. Forcing the prefix recovers question-directed text on 17 of 30, but the fine-tuned model escapes the block on a 16–21 character stub on the other 13 where the base model's shortest is 794 — and the one row re-run at full budget was a repetition loop. |

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
