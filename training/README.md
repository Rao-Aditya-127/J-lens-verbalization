# training

QLoRA SFT: teach Qwen3.6-27B to report the J-lens concepts active while it answers.

Data comes from `dataset/jlens/collected_answers.jsonl` (3,800 rows), also
published at <https://huggingface.co/datasets/RaoAditya/j-lens-verbalization>.

## Two targets, one model

Every collected row becomes **two** training examples, and the instruction says
which is being asked for:

| mode | asked for | target | note |
|---|---|---|---|
| A | "most active concepts" | `j_lens_top10` | ~58% of it is words already in the text |
| B | "concepts appearing nowhere in the text" | `j_lens_top10_novel` | no copying shortcut exists |

The mode **must** be in the prompt. Without it the same input carries two
different correct answers and the model learns to average them.

## Run order

```bash
python training/build_sft_dataset.py           # 6,020 train / 608 val / 972 test
python training/train_sft.py --smoke           # ~10 min, catches masking bugs
python training/train_sft.py                   # ~2-6 h on one 48GB card
python training/eval_sft.py --adapter training/runs/qlora-v1/final
```

Run the smoke test. It prints the fully-rendered example with its label mask
applied, which is the only reliable way to catch a chat-template or masking bug —
those waste an entire run and are invisible in the loss curve.

## Setup (RunPod / any single GPU)

```bash
pip install "transformers>=4.46" "trl>=0.12" peft bitsandbytes accelerate datasets
# strongly recommended: fused cross-entropy, see the vocab note below
pip install liger-kernel
```

Use a **persistent volume** — the checkpoint is 55.6 GB and re-downloading it
each session is billed time.

## Sizing

Measured on the actual dataset:

```
sequence length   median 432 tokens, p99 648, max 763   -> max_seq_len 1024 is safe
loss-bearing      target is ~14% of each sequence
tokens/epoch      ~2.6M      3 epochs ~7.8M
memory            ~31 GB at batch 4  -> fits 48GB; 80GB gives headroom
```

**The vocabulary is the memory risk, not the model.** 248,320 tokens × 5,120
hidden = 1.27B parameters in the embedding alone, and the logits tensor is
~2 GB at batch 4 (doubling in backward). If you OOM, that is why. A fused
cross-entropy (Liger, Unsloth) avoids materialising it and is the difference
between comfortable and tight on a 48 GB card.

Qwen3.6-27B is multimodal; loading the CausalLM class reads only `text_config`,
so the ~0.5B vision tower is never materialised.

## Reading the eval

`eval_sft.py` scores both lists under **two framings** — introspective (what was
trained) and guessing ("you have NO introspective access").

Prompting baselines for the untrained model:

```
list A    ICL 0.247 | zero-shot 0.192 | guessing control 0.170
list B    ~0.06 when asked for it directly
```

If the fine-tuned model scores the same under both framings, it learned to
**predict** J-lens output from text rather than to read its own state. That is a
real finding, and a different one from introspection — the headline number alone
cannot tell them apart, which is why the control ships with the eval rather than
being optional.
