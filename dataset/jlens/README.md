# jlens

Client-side aggregation for Neuronpedia's `POST /api/lens/prompt`
(Jacobian Lens) responses. See `think/understanding.md` for the full
collection plan this implements the aggregation step of.

The API has no layer-range or frequency-sort option: a `JACOBIAN_LENS`
response always contains every fitted layer's top-N read-out tokens for
every position (prompt and generated). Restricting to a later-layer window
and ranking by frequency happens entirely on our side, after the call.

`aggregate.py` provides `aggregate_top_k(response, config)`:

- keeps only `is_generated: true` positions (the answer, not the question);
- keeps only read-out layers inside `[config.layer_min, config.layer_max]`;
- normalizes each token (`strip().lower()`) and counts frequency across the
  kept layer slices;
- returns the top `config.top_k` tokens by count, with ties broken by
  earliest generated position, then lexically, for reproducibility.

It is a pure function of `(response, config)`: given the same buffered
(`stream: false`) response and the same `JLensConfig`, it always returns the
same result. This is why the collection plan calls for saving the full raw
API response per row (`j_lens_raw_response`) rather than only the derived
top-10 -- the window can be changed later (e.g. 22-63 vs 30-63) by re-running
this function, with no new API call.

```python
from aggregate import aggregate_top_k, JLensConfig

config = JLensConfig(layer_min=22, layer_max=63, top_k=10)
j_lens_top10 = aggregate_top_k(response, config)
```

`response` is the buffered `{meta, tokens, done}` JSON object returned when
`stream: false` is passed to `/api/lens/prompt`.

## Collection pipeline

`config.py` locks the Phase-0 experimental configuration (model id, lens
type, layer window, decoding params, the two fixed ICL demo `prompt_id`s).
Requires `NEURONPEDIA_API_KEY` in a `.env` file at the repo root (see
`.env.example`); `client.py` reads it via `python-dotenv` and sends it as the
`x-api-key` header. `client.py` also paces calls to stay under Neuronpedia's
240-requests/hour limit and retries with backoff on 429/5xx.

Two resumable phases, run in order (Phase 2 depends on Phase 1 having already
run for both the target row and the two demo rows):

```powershell
python dataset/jlens/collect.py answers [--limit N]
python dataset/jlens/collect.py introspection [--limit N] [--condition zero_shot|few_shot_icl|text_only_control|all]
python dataset/jlens/score.py
```

- **`answers`**: for each prompt-bank row (plus the 2 demo rows), calls
  `/api/lens/prompt` with just the question, extracts the answer from
  `done.completion`, aggregates `j_lens_top10` via `aggregate.py`, and
  appends one row to `collected_answers.jsonl`. Already-collected
  `prompt_id`s are skipped on re-run.
- **`introspection`**: for each condition, builds the introspection prompt
  (`prompts.py`) -- for `few_shot_icl`, prepending the 2 fixed demonstrations
  (their `<INTROSPECTION>` answer turn is synthesized from their real,
  already-collected `j_lens_top10`, not model-generated) -- calls the API
  again, parses the model's own `<INTROSPECTION>` block into
  `predicted_top10`, and appends to `collected_introspection.jsonl`.
  Already-collected `(prompt_id, condition)` pairs are skipped on re-run.
- **`score.py`**: reports mean overlap@10 / precision / recall / F1 per
  condition -- the project's first concrete result (does few-shot ICL beat
  zero-shot?).

Every raw API response is saved individually (gzip-compressed) under `raw/`
(git-ignored), named `{prompt_id}__answer.json.gz` or
`{prompt_id}__introspect_{condition}.json.gz`, so the layer window or parsing
rule can be revisited later without re-calling the API.

Recommended rollout: run both subcommands with `--limit 20` first, sanity
check `collected_answers.jsonl` and `collected_introspection.jsonl` by eye,
then drop `--limit` for the full 225-row run (expect several hours given the
rate limit).

`viewer/index.html` is a self-contained, offline HTML page for browsing
`collected_introspection.jsonl` -- drag the file onto the page, no server
needed.

**Phase 1 baseline is complete.** For the locked results, the full ICL demo
redesign history, bugs found and fixed, and what the numbers actually
establish, see `think/decision_log.md`.
