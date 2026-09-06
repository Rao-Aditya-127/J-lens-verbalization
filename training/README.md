# training

Everything that runs a model. Four groups, because these scripts serve very
different purposes and were getting hard to tell apart in one flat folder.

| folder | what it does | needs a GPU |
|---|---|---|
| [`sft/`](sft/) | build the dataset, fine-tune, evaluate | yes |
| [`lens/`](lens/) | run the Jacobian Lens locally, on the base model or the adapter | yes |
| [`injection/`](injection/) | the activation-injection experiment and its analysis | the experiment does; the stats and figure do not |
| [`analysis/`](analysis/) | analysis of results already collected | **no**, except `probe_thinking.py` |

Write-ups of what these produced live in [`results/`](../results/).
Setup for a fresh GPU box is in [RUNPOD.md](RUNPOD.md).

---

## `sft/` — the fine-tune

Teach Qwen3.6-27B to report the J-lens concepts active while it answers. Data is
`dataset/jlens/collected_answers.jsonl` (3,800 rows), also published at
<https://huggingface.co/datasets/RaoAditya/j-lens-verbalization>.

### Two targets, one model

Every collected row becomes **two** training examples, and the instruction says
which is being asked for:

| mode | asked for | target | note |
|---|---|---|---|
| A | "most active concepts" | `j_lens_top10` | ~55% of it is words already in the text |
| B | "concepts appearing nowhere in the text" | `j_lens_top10_novel` | no copying shortcut exists |

The mode **must** be in the prompt. Without it the same input carries two
different correct answers and the model learns to average them.

### Run order

```bash
python training/sft/build_sft_dataset.py           # 6,020 train / 608 val / 972 test
python training/sft/train_sft.py --smoke           # ~10 min, catches masking bugs
python training/sft/train_sft.py --epochs 2        # ~4.7 h on one 48GB card
python training/sft/eval_sft.py --adapter training/runs/qlora-v1/final_fixed \
    --limit 150 --no-thinking --max-new-tokens 256
```

**Run the smoke test.** It prints one fully-rendered example with its label mask
applied. A masking bug wastes an entire run and is invisible in the loss curve —
in fact it produces a *better*-looking curve, because copying the question back
is easier than reporting concepts. That happened here: the first run trained on
100% of every sequence.

### The three scripts that exist because something went wrong

- `check_template.py` — Qwen3.6 reasons by default. Without `enable_thinking=False`
  the prompt ends at `<think>` and the model spends its whole budget reasoning
  without answering. This prints how the template renders for training vs eval.
- `check_adapter.py` — PEFT matches adapter weights **by module name**. Load a
  different class than training saved against and it loads zero weights behind a
  warning, and generation silently returns base-model output. This counts
  non-zero `lora_B` in the *live* model and refuses to pass if enabling the
  adapter changes nothing.
- `fix_adapter_keys.py` — the rename that fixes exactly that. TRL saved keys
  under `model.language_model.layers`; `Qwen3_5ForCausalLM` wants `model.layers`.

### Sizing

```
sequence length   median 432 tokens, p99 648, max 763   -> max_seq_len 1024 is safe
loss-bearing      target is ~16% of each sequence
memory            ~31 GB at batch 4  -> fits 48GB
throughput        0.71 samples/s     -> 2 epochs ~4.7 h
```

**The vocabulary is the memory risk, not the model.** 248,320 tokens × 5,120
hidden = 1.27B parameters in the embedding alone, and the logits tensor is ~2 GB
at batch 4, doubling in backward. If you OOM, that is why — `--liger` avoids
materialising it, and `--batch-size 2 --grad-accum 8` is the simpler fallback.

---

## `lens/` — running the Jacobian Lens locally

Neuronpedia hosts the base model, not the adapter, so anything about the
fine-tuned model's *internals* has to run locally. These do that.

- `probe_interp_engine.py` — can interp-engine run the lens on a PEFT model?
  (Yes: hand `EagerModel` the inner module, `peft_model.base_model.model`, which
  carries the LoRA layers in place while keeping the original module paths.)
- `jlens_calibrate.py` — **run this first.** The checkpoint does not record
  whether the transport is `resid @ J` or `resid @ J.T`, so it sweeps both
  against readouts collected through the API: `J.T` reproduces them at 0.837
  overlap@15, `J` gives 0.000.
- `jlens_shift.py` — did fine-tuning move the J-space the eval scores against?
  (0.829 against a 0.845 noise floor: no.)

---

## `injection/` — the causal experiments

Change what is inside the model, leave the prompt byte-identical, and see whether
the report follows. The only experiments here that a text-only predictor cannot
explain. There are two, and they are different interventions:

| | write-up | models | dose axis |
|---|---|---|---|
| `swapToken`, via Neuronpedia's API | [`results/concept-swap/`](../results/concept-swap/) | base only | width of the layer band |
| `steer`, local | [`results/activation-injection/`](../results/activation-injection/) | base **and** fine-tuned | strength |

Swap could not be run against the fine-tuned model: it is an API parameter and
the API hosts the base model. Locally it also has no strength knob, which is why
the two-model comparison is built on `steer`.

```bash
python training/injection/inject_sanity.py --rows 3        # does the injection land?
python training/injection/inject.py --base-only --rows 100
python training/injection/inject.py --adapter RaoAditya/j-lens-verbalization-qlora --rows 100
python training/injection/inject_stats.py                  # offline
python training/injection/plot_inject.py                   # offline

python dataset/jlens/analysis/injection_curve.py           # the swap run, API key, no GPU
python training/injection/plot_swap.py                     # offline
```

Run `inject_sanity.py` first. A flat dose-response is ambiguous between "the
model can't report it" and "the intervention did nothing", and only the first is
a result.

---

## `analysis/` — mostly no GPU required

Everything here reads JSON that a GPU run already produced, with one exception:
`probe_thinking.py` generates, so it needs the model.

- `probe_thinking.py` — **needs a GPU.** What did fine-tuning cost outside the
  trained task? Puts an ordinary question to the model with the `<think>` prefix
  off and on. Answer: with it off, the fine-tuned model returns its trained
  concept list on 30 of 30 held-out rows and answers none of them. Write-up:
  [`results/capability-regression/`](../results/capability-regression/).
- `plot_regression.py` — the figure for that, with the counts re-derived from the
  per-row table rather than copied, so the figure and the write-up cannot drift
  apart.
- `compare_eval.py` — before/after with bootstrap CIs, separating accuracy from
  generation failure. An empty generation scores 0 alongside a genuinely wrong
  answer, and a mean cannot tell them apart.
- `show_examples.py` — the model's predictions beside the lens readout, hits
  marked, drawn with a fixed seed so nothing is cherry-picked.
- `text_only_baseline.py` — how much of the target is reachable from the text
  alone, by a bag-of-words nearest-neighbour predictor that never sees an
  activation. Reaches 0.427 of the fine-tuned model's 0.579.
- `ask.py` — interactive: ask your own question and watch the model report on
  itself under both framings.

---

## Reading the eval

`eval_sft.py` scores both lists under **two framings** — introspective (what was
trained) and guessing ("you have NO introspective access").

If the fine-tuned model scores the same under both, it learned to **predict**
J-lens output from text rather than to read its own state. That is a real
finding, and a different one from introspection.

One caveat found the hard way: after fine-tuning that control is close to inert.
The two framings agree with each other at **0.945**, against 0.394 for the base
model — six thousand training examples all used the introspective framing, so
the model learned one behaviour and runs it whatever the prompt claims. A null
under a manipulation that changes 5% of the output constrains very little, which
is why the injection experiment exists.
