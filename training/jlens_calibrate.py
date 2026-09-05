# -*- coding: utf-8 -*-
"""Reproduce Neuronpedia's J-lens readout locally, then check it actually matches.

    python training/jlens_calibrate.py --rows 5

Everything downstream -- the J-space of the fine-tuned model, injection on the
fine-tuned model -- rests on the local readout agreeing with the API that
produced all 3,800 collected rows. If it does not, new numbers are not
comparable with old ones and the difference has to be known now rather than
discovered in a result.

Two things are genuinely unknown and are resolved by measurement, not assumption:

  * transport orientation. J[layer] is [d_model, d_model]; whether the fitted
    map is applied as `resid @ J` or `resid @ J.T` is not recorded in the
    checkpoint.
  * whether the API's generation prompt had thinking enabled. The stored answers
    carry no <think> block, but that could mean it was disabled or that it was
    stripped, and the two prefixes put the answer at different positions.

Both are swept, and agreement with the stored `j_lens_top10` decides. Run with
the BASE model (no --adapter): that is what the stored readouts came from, so it
is the only configuration where agreement is the right expectation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]
COLLECTED = REPO / "dataset" / "jlens" / "collected_answers.jsonl"
MODEL_ID = "Qwen/Qwen3.6-27B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
LAYER_MIN, LAYER_MAX, TOP_N, TOP_K = 24, 58, 8, 15


def is_word_token(text: str) -> bool:
    """Approximates the API's filterNonWordTokens=True.

    Every one of the 114,000 collected concepts is a single space-free run of
    word characters, so anything else cannot match a stored target regardless.
    """
    s = text.strip()
    return bool(s) and " " not in s and any(ch.isalnum() for ch in s)


def build_transport(J: dict, transpose: bool):
    def transport(residual: torch.Tensor, layer: int) -> torch.Tensor:
        mat = J[layer].to(device=residual.device, dtype=residual.dtype)
        return residual @ (mat.T if transpose else mat)
    return transport


def readout(model, tok, lens_J, prompt_text: str, answer_text: str,
            transpose: bool, top_n: int = TOP_N, top_k: int = TOP_K) -> list[str]:
    """Top-k concepts by frequency over answer-token positions, layers 24-58."""
    from interp_engine.lens import layer_logits

    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"][0]
    answer_ids = tok(answer_text, add_special_tokens=False,
                     return_tensors="pt")["input_ids"][0]
    ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(model.device)

    # The API reads at generated positions. Teacher-forcing the stored answer puts
    # the same tokens at the same places, and greedy decoding made them
    # deterministic, so the residuals are the ones the readout was taken from.
    first, last = len(prompt_ids), ids.shape[1]
    layers = list(range(LAYER_MIN, LAYER_MAX + 1))

    out = layer_logits(model, ids, {"jacobian_lens": layers},
                       transport=build_transport(lens_J, transpose))
    per_layer = out["jacobian_lens"]

    counts: Counter = Counter()
    first_pos: dict[str, int] = {}
    for layer in layers:
        logits = per_layer[layer]
        if logits.dim() == 3:
            logits = logits[0]
        for pos in range(first, min(last, logits.shape[0])):
            for tid in logits[pos].topk(top_n).indices.tolist():
                text = tok.decode([tid])
                if not is_word_token(text):
                    continue
                token = text.strip().lower()
                counts[token] += 1
                first_pos.setdefault(token, pos)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], first_pos[kv[0]], kv[0]))
    return [t for t, _ in ordered[:top_k]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--adapter", type=str, default=None,
                   help="omit for the base model -- that is what the stored readouts came from")
    p.add_argument("--collected", type=Path, default=COLLECTED)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from interp_engine import EagerModel

    rows = [json.loads(l) for l in args.collected.open(encoding="utf-8") if l.strip()][: args.rows]
    print(f"{len(rows)} rows from {args.collected.name}")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    inner = hf
    if args.adapter:
        from peft import PeftModel
        inner = PeftModel.from_pretrained(hf, args.adapter).base_model.model
        print(f"adapter: {args.adapter}")
    model = EagerModel(MODEL_ID, hf_model=inner, tokenizer=tok)
    print(f"model on {model.device}, {model.n_layers} layers\n")

    blob = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu",
                      weights_only=False)
    J = {int(k): v for k, v in blob["J"].items()}
    print(f"lens: {len(J)} layers, J[24] {tuple(J[24].shape)} {J[24].dtype}\n")

    print("=" * 78)
    print("SWEEP: transport orientation x thinking prefix, scored against the API readout")
    print("=" * 78)
    print(f"  {'orientation':<14}{'thinking':<12}{'mean overlap@15 vs stored':>28}")
    best = None
    for transpose in (False, True):
        for thinking in (False, True):
            scores = []
            for row in rows:
                chat = [{"role": "user", "content": row["question"]}]
                prompt = tok.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                    **({} if thinking else {"enable_thinking": False}))
                pred = readout(model, tok, J, prompt, row["answer"], transpose)
                truth = {c["concept"].strip().lower() for c in row["j_lens_top10"]}
                scores.append(len(set(pred) & truth) / TOP_K)
            mean_score = sum(scores) / len(scores)
            label = "resid @ J.T" if transpose else "resid @ J"
            print(f"  {label:<14}{str(thinking):<12}{mean_score:>28.3f}")
            if best is None or mean_score > best[0]:
                best = (mean_score, transpose, thinking)

    score, transpose, thinking = best
    print("\n" + "=" * 78)
    print(f"BEST: {'resid @ J.T' if transpose else 'resid @ J'}, "
          f"thinking={thinking}, overlap {score:.3f}")
    print("=" * 78)
    if score < 0.6:
        print("  Below 0.6 -- the local readout is NOT reproducing the API. Do not build")
        print("  on it. Likely causes: wrong prompt prefix, a different fitted lens, or")
        print("  a filter the API applies that is_word_token does not.")
    else:
        print("  Reproduces the API well enough to compare new readouts against the")
        print("  3,800 collected rows.")

    # WHERE does the disagreement live? If it is the fragile low-count tail, overlap
    # should be near-perfect at top-5 and fall toward the top-15 figure, and the API
    # concepts we miss should sit well below rank 8 (the uniform expectation for a
    # 15-item list). If instead it is flat across k and the missed ranks average ~8,
    # the disagreement is everywhere and the local readout differs structurally.
    print("\nWHERE THE DISAGREEMENT LIVES")
    print(f"  {'k':<6}{'mean overlap@k':>18}")
    missed_ranks: list[int] = []
    for k in (5, 10, 15):
        scores = []
        for row in rows:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": row["question"]}], tokenize=False,
                add_generation_prompt=True,
                **({} if thinking else {"enable_thinking": False}))
            pred = readout(model, tok, J, prompt, row["answer"], transpose, top_k=k)
            truth = [c["concept"].strip().lower() for c in row["j_lens_top10"]][:k]
            scores.append(len(set(pred) & set(truth)) / k)
            if k == TOP_K:
                got = set(pred)
                missed_ranks += [i for i, t in enumerate(truth, start=1) if t not in got]
        print(f"  {k:<6}{sum(scores) / len(scores):>18.3f}")

    if missed_ranks:
        mean_rank = sum(missed_ranks) / len(missed_ranks)
        print(f"\n  API concepts the local readout missed: {len(missed_ranks)} "
              f"of {len(rows) * TOP_K}")
        print(f"  their mean rank in the API list: {mean_rank:.1f} of 15 "
              f"(uniform would be 8.0)")
        if mean_rank > 9.5:
            print("  -> concentrated in the low-count tail: the readout agrees where the")
            print("     signal is strong and reshuffles where counts are 1-2.")
        else:
            print("  -> spread across the ranking, not just the tail. The local readout")
            print("     differs structurally and local/API numbers should not be mixed.")

    row = rows[0]
    pred = readout(model, tok, J, tok.apply_chat_template(
        [{"role": "user", "content": row["question"]}], tokenize=False,
        add_generation_prompt=True,
        **({} if thinking else {"enable_thinking": False})),
        row["answer"], transpose)
    truth = [c["concept"].strip().lower() for c in row["j_lens_top10"]]
    print(f"\n{row['example_id']}")
    print("  API   :", ", ".join(truth))
    print("  local :", ", ".join(pred))
    print("  shared:", ", ".join(sorted(set(pred) & set(truth))) or "(none)")


if __name__ == "__main__":
    main()
