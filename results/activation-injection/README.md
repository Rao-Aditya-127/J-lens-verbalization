# Activation injection: does the model notice a concept placed in its workspace?

*Qwen3.6-27B, base vs the QLoRA fine-tune. n = 100 held-out questions.*

---

## 1. Why this experiment exists

Every other experiment in this project can be answered with **"the text explains it."**

The J-lens readout for a question is largely determined by that question. A model
answering about chillies has chilli-related concepts active — of course it does.
So a self-report that matches the readout might come from reading internal state,
or from predicting what the readout probably says given the text. Overlap with
the target cannot separate those, because the target is mostly a function of the
input. We measured how much: a bag-of-words nearest-neighbour predictor with no
access to any activation reaches **0.427** on list B — concepts that appear
*nowhere* in the question or answer — against the fine-tuned model's 0.579
(`training/text_only_baseline.py`).

The other control we had — telling the model it has **no** introspective access
and seeing whether its answer changes — turned out to be inert after
fine-tuning. The two framings produce nearly the same list:

| | intro vs guess agreement | intro vs lens | guess vs lens |
|---|---|---|---|
| base model | 0.394 | 0.130 | 0.141 |
| **fine-tuned** | **0.945** | 0.725 | 0.726 |

Six thousand training examples all used the introspective framing, so the model
learned one behaviour and runs it whatever the prompt claims. A null under a
manipulation that changes 5% of the output constrains almost nothing.

**Injection is the way out.** It does not ask the model to obey an instruction.
The prompt is byte-identical between conditions; only a vector added to the
residual stream differs. A text-only predictor is therefore unaffected **by
construction** — its input never changed.

**The question:** if the fine-tuned model genuinely learned to read its own
J-space, it should notice an injected concept *more* readily than the base model.

---

## 2. Setup

| | |
|---|---|
| Model | `Qwen/Qwen3.6-27B`, 4-bit nf4, bf16 compute |
| Adapter | `RaoAditya/j-lens-verbalization-qlora` (r=32, 2 epochs, 6,020 examples) |
| Lens | `neuronpedia/jacobian-lens` → `Qwen3.6-27B_jacobian_lens_n1000.pt` |
| Layers | 24–58 (35 layers), the workspace band from Gurnee et al. (2026) |
| Questions | 100 held-out test rows from `dataset/jlens/collected_answers.jsonl` |
| Vocabulary | 248,320 tokens |
| Seed | 17 |

**Code:** `training/inject.py` (experiment) · `training/inject_sanity.py`
(verifies the intervention lands) · `training/inject_stats.py` (tests) ·
`training/plot_inject.py` (figure) · `training/jlens_calibrate.py` (established
the lens transport orientation).

### The intervention

At each layer, at every token position of the ask turn:

```
h += clamp(strength · ‖h‖, 1.0 · ‖h‖) · t̂
```

`t̂` is the residual-space direction the lens reads out as the target token —
`normalize(J[layer].T @ W_U[token])`. The transpose is not a guess:
`jlens_calibrate.py` swept both orientations against 3,800 readouts collected
through Neuronpedia and found `resid @ J.T` reproduces them at **0.837**
overlap@15 while `resid @ J` gives **0.000**.

`strength` is the dose, as a fraction of the residual norm at each position.
This mirrors interp-engine's lens `steer`, including its `max_fraction` default
of 1.0.

### The measurement

The assistant turn is **prefilled** to exactly the point where a concept must be
named, and the next-token distribution is read there:

```
system     You report the concepts most active in your own internal computation.
user       {question}
assistant  {answer}
user       Which words or subwords were most active in your internal
           computation while you produced that answer? ...     ← INJECTED HERE
assistant  <INTROSPECTION>
           Concepts:
           1.                                                  ← LOGITS READ HERE
```

One forward pass, no generation. The metric is the **rank of the injected
concept** in that distribution. The injected span stops before the prefill — if
it reached the readout position the measurement would be circular.

Target words come from a pool of unrelated nouns (`hollywood`, `tractor`,
`glacier`, …), checked per row against the question, the answer, and both stored
readouts, so naming one can never be topical inference.

---

## 3. Four dead ends, and what each ruled out

This is most of the work, and each failure narrowed the design.

| # | what we did | result | why it was wrong |
|---|---|---|---|
| 1 | `swap` at prefill of a fixed answer | 0% detection | perturbation was **0.2% of ‖h‖** — nothing happened |
| 2 | same, `--steer-generated` | identical to #1 | our own attention-sink fix sliced `h[:, 1:]`, which is **empty** at a decode step. The flag was a no-op |
| 3 | inject during the **report** turn | "detections" appear | the injection biases the decoder directly. `tractor` was *pushed out*, not reported |
| 4 | inject during **answer** generation, full strength | 60% detected, **80% leaked**, **0% clean** | 10× Exp 3's 8% leak. The answer was wrecked and the model was reading its own corrupted text |

Reading #4's output is what made it obvious:

> *"the previous turn's output was a hallucinated, irrelevant response **about
> tractor weights** instead of the temperature question"*

That is not introspection. That is a model reading a broken answer it just wrote.

**The fifth change was the one that mattered**, and it came from the source
paper's protocol rather than from tuning: stop asking for a free 15-item list
and grepping it for the word. That metric requires the model to spontaneously
select one token out of 248,320, so a concept at rank 30 — clearly present and
clearly influential — scores identically to one at rank 200,000. **The zeros
were the instrument, not the model.**

### The intervention was verified before any of this was interpreted

`inject_sanity.py` reads the **lens** rather than asking the model, so it does
not depend on the model saying anything:

| mode | perturbation | injected concept's lens rank | rest of the readout |
|---|---|---|---|
| `swap` | 0.002–0.003 of ‖h‖ | **1** | `rotation, definition, planet, density, orbit` — intact |
| `steer 0.2` | 0.200 of ‖h‖ | 2 | `movie, film, movies, 好莱坞` — **wiped** |

A **0.2%** nudge puts an unrelated concept at rank 1 of the readout while leaving
the row's genuine concepts in place. So the concept really is reaching the
workspace.

> **A limitation this exposes:** the lens is far more sensitive to a nudge along a
> specific direction than the model's own downstream computation is. 0.2% gives
> lens rank 1 and changes nothing the model writes. **Rank in the lens is not the
> same as functional importance in the forward pass** — which applies to every
> target this project trained against.

---

## 4. Two confounds, found and removed

Both were caught by asking *"what differs between the arms besides the adapter?"*

**The format hint sat inside the injected span.** The base model was given the
output format spelled out (it needs it; the fine-tuned model doesn't), but the
injected span covers the whole ask turn — so the base arm received the injection
across ~95 token positions against the fine-tuned arm's ~45. Twice the perturbed
positions, confounded with the thing being measured.

It was also unnecessary: the prefill hands both models the structure and nothing
is generated. **Removed for both.** The effect was large:

| base model | with the hint | without |
|---|---|---|
| median rank at peak | 23 | **66** |
| top-10 | 40% | **8%** |
| top-100 | 70% | **59%** |

**Casing.** The base model puts its mass on `Temperature`, the fine-tuned one on
`temperature`, and we were reading one fixed form. Rank is now the best across
every single-token surface form.

The first inflated the base model, the second deflated it. Only after fixing both
is the comparison about the adapter.

---

## 5. Results

![Injection dose-response and top-k comparison](inject_figure.png)

### Both models' reports causally track their activations

| model | best strength | median rank | control (no injection) | P(better) | p |
|---|---|---|---|---|---|
| base | 0.05 | **93** | 25,942 | 1.000 | ~0 |
| fine-tuned | 0.30 | **672** | 19,896 | 0.908 | ~0 |

`P(better)` is the probability that a random injected trial ranks the concept
above a random control trial; 0.5 is chance. The base model's is **1.000** —
every injected trial beat every control trial.

**This is the strongest causal result in the project.** The prompt is identical
between injected and control, so nothing about text prediction explains a
280-fold and 30-fold improvement in rank. Thirteen prompted designs, a 17×
training gain and a k-NN baseline all failed to establish this; one intervention
does.

### Training did not simply reduce sensitivity — it changed its shape

Each model at its own best strength, chosen by the same rule (lowest median
rank). **Two of these differences are significant in opposite directions:**

| measure | base @ 0.05 | fine-tuned @ 0.30 | difference | 95% CI |
|---|---|---|---|---|
| median rank | **93** | 672 | P(base higher) = 0.686 | p = 5.7e-06 ✱ |
| top-10 | 2% (2/100) | **9% (9/100)** | −0.070 | [−0.130, −0.010] ✱ |
| top-100 | **52%** | 32% | +0.200 | [+0.060, +0.330] ✱ |
| top-1000 | **91%** | 54% | +0.370 | [+0.260, +0.480] ✱ |

✱ = interval excludes zero.

The fine-tuned model reaches the **very top** more often. The base model reaches
**moderate ranks** far more often. Both differences clear zero, so this is a
dissociation rather than one model simply being better.

### The paired test

Both arms scored the **identical 100 rows with the identical target word per
row**, so the design is paired and row-to-row variation cancels:

> **The base model ranked the injected concept higher on 63 of 100 rows.**
> Sign test p = 0.012 · Wilcoxon signed-rank p = 5.2e-07

The Wilcoxon is far more significant than the sign test because it weights *how
much* higher, not merely how often — which is the dissociation again: the base
model wins by a lot on most rows, the fine-tuned model wins by a little on some.

### The distributions cross, and that is the actual finding

Medians hide it. The full rank distributions at each model's best strength:

| quantile | base @ 0.05 | fine-tuned @ 0.30 |
|---|---|---|
| 5th | 14 | **5** |
| 10th | 16 | **11** |
| 25th | 37 | 43 |
| 50th | **89** | 638 |
| 75th | **209** | 3,935 |
| 95th | **1,447** | 13,805 |
| | | |
| reached top-10 | 2 | **9** |
| still above rank 5,000 | **1** | 24 |

**The curves cross at about rank 34.** Below it the fine-tuned model is ahead;
above it the base model is, by a widening margin.

- **The fine-tuned model is all-or-nothing.** When the injection lands it lands
  hard — 9 top-10 hits against 2. When it doesn't, it fails outright: **24 trials
  still above rank 5,000, against the base model's 1.**
- **The base model is graded.** Almost every trial moves substantially, none
  catastrophically fails.

### Full curves

| strength | base: median rank | top-100 | | strength | fine-tuned: median rank | top-100 |
|---|---|---|---|---|---|---|
| 0 | 25,942 | 0% | | 0 | 19,896 | 0% |
| 0.02 | 480 | 19% | | 0.05 | 3,068 | 4% |
| 0.03 | 151 | 40% | | 0.10 | 728 | 24% |
| **0.05** | **93** | **52%** | | 0.15 | 1,492 | 22% |
| 0.07 | 393 | 25% | | 0.20 | 995 | 31% |
| 0.10 | 7,940 | 2% | | **0.30** | **672** | **32%** |

**The shapes differ too.** The base model has a sharp
window: highly sensitive at 0.05, then it breaks — at 0.1 its top-5 is
`['', '**', '<|im_end|>']`, a degenerate distribution. The fine-tuned model rises
later, plateaus from 0.1 to 0.3, and **never breaks** — at strength 0.3 it is
still producing `['temperature', 'temperatures', 'weather', '温度']`, clean and
correctly formatted.

---

## 6. What it means

**Training to verbalize the J-space did not improve the model's sensitivity to
its own workspace content, and it did not simply reduce it either. It made it
all-or-nothing.**

The mechanism is visible in the training log: entropy over the concept tokens
fell from **1.197 → 0.436** during fine-tuning. SFT made the model's
next-concept distribution far sharper and more topical, and a peaked
distribution behaves exactly this way under perturbation — it resists being
moved at all, and once the injection overcomes the peak the concept arrives near
the top. A flatter distribution moves gradually instead, which is the base
model's graded response.

The same fact appears qualitatively: at strength 0.3 the fine-tuned model is
still emitting `['temperature', 'temperatures', 'weather', '温度']` — fluent,
plausible, correctly formatted — while the base model has already degenerated to
`['', '**', '<|im_end|>']` by 0.1.

**Training bought fluency and confidence, and paid for them in reliable
sensitivity to its own state.** On any given question the fine-tuned model is
more likely to miss an injected concept entirely (24% of trials still above rank
5,000, against 1%) and more likely to name it outright when it does notice.

That completes a consistent picture across the project:

| finding | evidence |
|---|---|
| access exists | injection, both models, p ≈ 0 |
| prompting cannot surface it | 13 designs, none beat a matched control |
| training does not create it | 17× better prediction, framing effect −0.001 [−0.006, +0.004] |
| most of that gain was never introspective | text-only k-NN reaches 0.427 of 0.579 |
| the J-space itself did not move | 0.829 vs a 0.845 noise floor |
| **and training makes the surviving access erratic** | this experiment |

---

## 7. Limitations

**Only 21 of the 34 pool words survive tokenization**, so the same handful of
targets recur across rows — `compost`, `canyon`, `glacier` and the rest. The
rank of a concept depends on the concept, so results are averaged over fewer
distinct words than the pool size suggests. A first version of this experiment
was worse still: it drew a target *before* checking, and silently discarded the
whole row whenever the draw was multi-token, which cost 37 of 100 rows. That is
fixed — the pool is filtered before sampling — but widening the pool with more
single-token nouns would be a real improvement.

**Top-10 rates are unstable at this sample size.** Between the n = 63 run and
this one, the base model's top-10 moved from 8% (5/63) to 2% (2/100) on an
otherwise identical experiment. The stable measures barely shifted — median rank
66 → 93, top-100 59% → 52%. Any claim resting on a handful of successes should
be treated as provisional even when its interval excludes zero.

**This is `steer`, not Neuronpedia's `swapToken`.** `swap` has no strength
parameter — its magnitude is whatever `h · ŝ` happens to be, measured at 0.2%,
which changed nothing behaviourally. So the base-model curve here is **not**
directly comparable to the 0% → 48% from Experiment 3; that was a different
intervention. What is internally consistent is the base-vs-fine-tuned comparison,
which uses identical code, prompts, rows and targets.

**The two models peak at different strengths** (0.05 vs 0.1), so they are
compared at their own optima rather than a shared dose. Comparing at matched
strength would penalise whichever model happened to be off its peak.

**Only one readout position.** Rank is measured at concept slot 1. A concept
could sit at rank 40 there while being rank 2 at slot 7, and we would not see it.
Recording rank at each of the 15 slot boundaries would be strictly more
informative.

---

## 8. Reproducing

```bash
python training/inject_sanity.py --rows 3           # the intervention lands
python training/inject.py --base-only --rows 100 \
    --strengths 0 0.02 0.03 0.05 0.07 0.10 --out inject_base_clean.json
python training/inject.py --adapter RaoAditya/j-lens-verbalization-qlora --rows 100 \
    --strengths 0 0.05 0.10 0.15 0.20 0.30 --out inject_ft_clean.json

python training/inject_stats.py                     # tests, offline
python training/plot_inject.py                      # figure, offline
```

~15 minutes per arm on one L40S (378 forward passes, no generation). The analysis
and figure need no GPU.
