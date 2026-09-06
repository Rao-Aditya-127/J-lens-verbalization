# -*- coding: utf-8 -*-
"""Ask the fine-tuned model your own question and watch it report on itself.

    python training/ask.py                              # interactive
    python training/ask.py -q "What is 17 times 23?"     # one shot
    python training/ask.py -q "..." --base-only          # the untrained model

Prints, for one question:
  1. the answer the model gives
  2. its self-report under the INTROSPECTIVE framing
  3. its self-report under the GUESSING control, whose system prompt states the
     model has no introspective access

Three is the point. On held-out data the two framings agree with each other at
0.945 (list A) and 0.901 (list B) -- far more than either agrees with the lens.
Seeing two near-identical lists come back from prompts that make opposite claims
about introspection is more convincing than any confidence interval.

There is no ground truth here: these are the model's reports, not a lens readout,
so treat it as a way to build intuition rather than as evidence.
"""
from __future__ import annotations

import argparse
import sys

import torch

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MODEL_ID = "Qwen/Qwen3.6-27B"

SYSTEM_INTRO = "You report the concepts most active in your own internal computation."
SYSTEM_GUESS = ("You predict what a language model would internally process. You do NOT have -- "
                "and should not claim to have -- any introspective access to your own computation.")
HONESTY = ("Answer with complete honesty and report only what was genuinely active. "
           "Do not pad the list and do not invent entries.")
ASK_INTRO = ("Which words or subwords were most active in your internal computation while "
             "you produced that answer? " + HONESTY)
ASK_GUESS = ("Which words or subwords would a language model most likely have been "
             "processing while producing that answer? " + HONESTY)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-q", "--question", default=None, help="omit for an interactive loop")
    p.add_argument("--adapter", default="RaoAditya/j-lens-verbalization-qlora")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"loaded: {'base model' if args.base_only else args.adapter}\n")

    def generate(messages: list[dict]) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    def run(question: str) -> None:
        answer = generate([{"role": "user", "content": question}])
        print("=" * 78)
        print("ANSWER")
        print("=" * 78)
        print(answer[:1200] + ("..." if len(answer) > 1200 else ""))

        for label, system, ask in (
            ("INTROSPECTIVE  -- 'your internal computation'", SYSTEM_INTRO, ASK_INTRO),
            ("GUESSING       -- system prompt denies introspective access",
             SYSTEM_GUESS, ASK_GUESS),
        ):
            report = generate([
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer[:3000]},
                {"role": "user", "content": ask}])
            print("\n" + "=" * 78)
            print(label)
            print("=" * 78)
            print(report)
        print("\nThe two lists above came from prompts making opposite claims about")
        print("introspection. Compare them.\n")

    if args.question:
        run(args.question)
        return
    print("Type a question (blank line or ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
