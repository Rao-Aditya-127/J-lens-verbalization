# Running the SFT on RunPod — from zero

Assumes nothing. Every command is here in order.

---

## 1. Create the pod

On <https://runpod.io> → **Pods** → **Deploy**.

| setting | value | why |
|---|---|---|
| GPU | **1 × L40S (48 GB)** | ~$0.99/hr. A100 80GB also fine if you want headroom |
| Template | **RunPod PyTorch 2.6 or newer** | torch < 2.5 breaks transformers — see step 2 |
| Container Disk | **60 GB** | OS, python packages |
| **Volume Disk** | **150 GB** | **persistent** — mounted at `/workspace`, survives stop/start |
| Volume Mount Path | `/workspace` | the default |

**The volume disk is the important one.** The model checkpoint is 55.6 GB. If you
put it on container disk it is re-downloaded every time you restart the pod, and
you pay for that download time. On the volume it persists.

Deploy, wait for **Running**, then **Connect → Start Web Terminal** (or use SSH).

---

## 2. Check what you actually got

```bash
nvidia-smi
```

Confirm you see the GPU and its memory (should say ~46-49 GB for an L40S). Then:

```bash
df -h /workspace          # should show ~150G
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda.is_available()` prints `False`, the pod has no GPU attached — destroy it
and redeploy. Nothing below will work.

### torch must be >= 2.5

Check the version the template gave you:

```bash
python -c "import torch; print(torch.__version__)"
```

**If it starts with `2.4` or lower, upgrade now**, before installing anything
else:

```bash
pip install --upgrade torch torchvision
pip install --upgrade --force-reinstall bitsandbytes
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Qwen3.6's `qwen3_5` architecture only exists in recent `transformers`, and recent
`transformers` requires torch >= 2.5. On an older torch it disables its PyTorch
backend and then fails with a confusing `NameError: name 'nn' is not defined`
somewhere deep in an import — see the troubleshooting entry. Pinning
`transformers` back is not a workaround; the model would not load at all.

---

## 3. Point Hugging Face at the persistent volume

**Do this before anything downloads.** By default HF caches to `~/.cache`, which
is on container disk and dies with the pod.

```bash
export HF_HOME=/workspace/hf
mkdir -p /workspace/hf

# make it stick for future shells
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
```

---

## 4. Get the code

```bash
cd /workspace
git clone https://github.com/Rao-Aditya-127/J-lens-verbalization.git
cd J-lens-verbalization
```

---

## 5. Install dependencies

```bash
pip install --upgrade pip

pip install "transformers>=4.46" "trl>=0.12" peft bitsandbytes accelerate datasets
pip install matplotlib              # for plot_training.py
pip install huggingface_hub
```

Verify the ones that actually matter:

```bash
python -c "
import torch, transformers, trl, peft, bitsandbytes
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('trl         ', trl.__version__)
print('peft        ', peft.__version__)
print('bitsandbytes', bitsandbytes.__version__)
print('gpu         ', torch.cuda.get_device_name(0))
"
```

`bitsandbytes` is the one that most often fails to import on a mismatched CUDA
build. If it errors, `pip install -U bitsandbytes` usually fixes it.

---

## 6. Get the dataset

The data is **not** in the git repo — it lives on Hugging Face.

```bash
mkdir -p dataset/jlens
python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('RaoAditya/j-lens-verbalization',
                    'collected_answers.jsonl', repo_type='dataset')
shutil.copy(p, 'dataset/jlens/collected_answers.jsonl')
print('downloaded ->', 'dataset/jlens/collected_answers.jsonl')
"

wc -l dataset/jlens/collected_answers.jsonl      # expect 3800
```

No HF token needed — the dataset and the model are both public.

---

## 7. Build the training examples

```bash
python training/build_sft_dataset.py
```

Expect:

```
read 3800 collected rows
  sft_test.jsonl           972 examples   A=486 B=486
  sft_train.jsonl         6020 examples   A=3010 B=3010
  sft_validation.jsonl     608 examples   A=304 B=304
```

---

## 8. Optional: Weights & Biases

Skip this and everything still works — `log_history.json` is written either way.
But the pod is ephemeral, and W&B is the easiest way to keep the curves.

```bash
pip install wandb
export WANDB_API_KEY=<your key from wandb.ai/authorize>
echo 'export WANDB_API_KEY=<your key>' >> ~/.bashrc
```

---

## 9. Baseline: score the UNTRAINED model first

```bash
python training/eval_sft.py --base-only --adapter none --limit 150
```

This gives you a before/after under *identical* prompts and rows. The 0.247
figure from earlier used a different prompt on different rows, so it is not a
clean comparison — this is.

**Budget for it.** The first invocation downloads the 55.6 GB model (15-30 min,
one time, onto the volume). Then generation is the cost: 4-bit 27B runs at
roughly 10-20 tok/s, and each row needs **4 generations** (2 lists x 2 framings)
of up to 160 tokens.

| `--limit` | generations | rough wall time |
|---|---|---|
| 50 | 200 | ~40 min |
| **150** | **600** | **~2 hr** |
| all (486) | 1944 | ~6.5 hr — $6+ of GPU |

`--limit 150` is the right call: it takes the first 150 test rows
deterministically, so the baseline and the post-training eval score the *same*
rows, and n=150 resolves a paired difference of ~0.03 in overlap. Running the
full 486 costs 4x more for a marginal gain in precision.

---

## 10. Smoke test — do not skip

```bash
python training/train_sft.py --smoke
```

~10 minutes. It prints a block like:

```
======================================================================
MASKING CHECK -- only this text should contribute to the loss:
======================================================================
<INTROSPECTION>
Concepts:
1. county
...
</INTROSPECTION>
======================================================================
94 of 512 tokens are loss-bearing (18%) -- expect roughly 10-20%
```

**Read it.** If the question or the answer appears in that block, the mask is
wrong and the full run would be wasted. If it says 0 loss-bearing tokens the
script stops by itself.

---

## 11. Full run

```bash
nohup python training/train_sft.py > /workspace/train.log 2>&1 &

tail -f /workspace/train.log        # ctrl-C stops watching, not training
```

`nohup ... &` means it survives your terminal disconnecting — a browser tab
closing will otherwise kill it.

Expect **2-4 hours**. Watch that loss starts around 2-4 and falls.

---

## 12. Evaluate

```bash
python training/eval_sft.py --adapter training/runs/qlora-v1/final --limit 150     --out training/eval_after.json
python training/plot_training.py training/runs/qlora-v1/final/log_history.json
```

**Use the same `--limit` as step 9**, or you are comparing different rows. Note
the separate `--out`: the default path is the same file the baseline wrote, and
without this the baseline numbers are overwritten.

The table to read is `list A introspective` vs `list A guessing`. If they are
equal, the model learned to *predict* J-lens output from text rather than to read
its own state — which is the same conclusion the thirteen prompted designs
reached, now with training thrown at it.

---

## 13. Download EVERYTHING before you terminate

Local files do not survive pod termination.

```bash
cd /workspace/J-lens-verbalization
tar czf /workspace/results.tar.gz \
    training/runs/qlora-v1/final \
    training/eval_results.json \
    training/runs/qlora-v1/final/loss_curve.png \
    /workspace/train.log
ls -lh /workspace/results.tar.gz
```

Then in RunPod: **Connect → HTTP/Jupyter**, or use `runpodctl send`, or
`scp` if you set up SSH. Download `results.tar.gz` before hitting terminate.

**Stop** the pod (keeps the volume, stops billing GPU) rather than **Terminate**
(destroys everything) if you plan to run again.

---

## Troubleshooting

**CUDA out of memory.** It is the 248,320-token vocabulary, not the model. The
logits tensor is ~2 GB at batch 4 and doubles in the backward pass. In order:

```bash
python training/train_sft.py --batch-size 2 --grad-accum 8   # same effective batch

pip install liger-kernel                                     # then, if still OOM
python training/train_sft.py --liger
```

`--liger` is off by default because liger-kernel has to support this exact
architecture and fails loudly if it does not. Try the batch size first.

**`NameError: name 'nn' is not defined`** from inside a `transformers` import,
usually preceded by `Disabling PyTorch because PyTorch >= 2.5 is required but
found 2.4.x`. The torch version is too old for `transformers`; the `NameError` is
a downstream symptom. Fix with the torch upgrade in step 2.

**`bitsandbytes` import error** — `pip install -U bitsandbytes`. After any torch
upgrade, use `pip install --upgrade --force-reinstall bitsandbytes` so it rebinds
to the new torch.

**Loss starts near 0.1** — the mask is letting the model see the answer. Stop and
send me the masking block.

**Loss is flat** — learning rate problem. Try `--lr 1e-4` or `--lr 3e-4`.

**Validation loss rising while train loss falls** — overfitting. The run prints
the best step at the end; evaluate that checkpoint instead of `final/`, or rerun
with `--epochs 2`.

**Disconnected and the run is gone** — you forgot `nohup`. Check with
`ps aux | grep train_sft`.
