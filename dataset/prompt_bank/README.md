# Dataset

`build_prompt_bank.py` is the single script that builds the prompt-only
prompt bank. It draws from five sources:

- GSM8K's official test split (free-response arithmetic word problems);
- BBH reasoning prompts, split evenly across five named subtasks
  (`boolean_expressions`, `causal_judgement`, `date_understanding`,
  `logical_deduction_five_objects`, `tracking_shuffled_objects_five_objects`);
- TruthfulQA's official CSV;
- ARC-Challenge science/general-knowledge multiple-choice questions; and
- HotpotQA `bridge`-type multi-hop questions (distractor config, validation
  split). Only `bridge` type is used, not `comparison`: a bridge question
  requires passing through an intermediate entity that is never named in the
  question itself, so a plausible-sounding self-report inferred purely from
  the question/answer text is more likely to diverge from the model's actual
  internal concepts than on the other four sources.

Every record identifies its source dataset and, for BBH, its subtask; HotpotQA
records also carry `source_type` (always `bridge`) and `source_level`. ARC
records preserve their answer choices but omit the `answerKey`. No record
from any source contains an `answer`, `solution`, `rationale`, `target`,
`answerKey`, `supporting_facts`, or `context` field.

## Regenerating the prompt bank

```powershell
python dataset/prompt_bank/build_prompt_bank.py
```

With no arguments this reproduces
`prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_225.jsonl` (50 GSM8K + 50 BBH +
50 TruthfulQA + 25 ARC-Challenge + 50 HotpotQA).

Each source's count is configurable independently:

```powershell
python dataset/prompt_bank/build_prompt_bank.py --gsm8k-count 50 --bbh-count 50 --truthfulqa-count 50 --arc-count 50 --hotpotqa-count 50
```

- `--gsm8k-count`, `--truthfulqa-count`, `--arc-count`, `--hotpotqa-count`
  each take a plain total.
- `--bbh-count` is a total split evenly across the five fixed subtasks, so it
  must be divisible by 5.
- `--output` overrides the output path; otherwise the filename encodes the
  resulting total, e.g.
  `prompt_bank_gsm8k_bbh_truthfulqa_arc_hotpotqa_<total>.jsonl`, written next
  to the script.

The current sample is a small schema/collection pilot; its `split` is
`unassigned` until the combined bank is stratified into train, validation,
and test partitions.
