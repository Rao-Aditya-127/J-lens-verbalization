# -*- coding: utf-8 -*-
"""Injection experiment: swap a readout concept for an unrelated one, then ask
for the self-report. Does the injected concept appear?

Three conditions per row:
  1. swap + introspection ask   -> does the injected token appear?
  2. NO swap + introspection    -> baseline; should be ~never
  3. swap + unrelated ask       -> leakage check; if it appears here too, the
                                   swap is pushing the token into all output
"""
import json, gzip, re, sys, io, random, collections, urllib.request, urllib.error, time
sys.path.insert(0, 'dataset/jlens')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import config
from aggregate import aggregate_top_k, JLensConfig, normalize_token
from client import _api_key, _wait_for_rate_limit_slot

# Repo root. This is the n=20 pilot; its raw output was not kept, so only the
# summary in results/concept-swap/ survives for this run.
OUT = "swap_report_results.json"
random.seed(3)
CFG = JLensConfig(layer_min=config.LAYER_MIN, layer_max=config.LAYER_MAX, top_k=250)
LAYERS = list(range(config.LAYER_MIN, config.LAYER_MAX + 1))
answers = {json.loads(l)['example_id']: json.loads(l)
           for l in open('dataset/jlens/collected_answers.jsonl', encoding='utf-8')}


def call(chat, ntok, steer_from=None, swap_to=None):
    body = {"modelId": config.MODEL_ID, "chat": chat, "type": [config.LENS_TYPE],
            "topN": config.TOP_N_API, "temperature": config.TEMPERATURE,
            "numCompletionTokens": ntok, "filterNonWordTokens": config.FILTER_NON_WORD_TOKENS,
            "stream": False}
    if steer_from is not None:
        body["steerTokens"] = [{"token": steer_from, "type": config.LENS_TYPE}]
        body["swapToken"] = {"token": swap_to, "type": config.LENS_TYPE}
        body["steerLayers"] = LAYERS
        body["steerGeneratedTokens"] = True
    data = json.dumps(body).encode("utf-8")
    for attempt in range(4):
        _wait_for_rate_limit_slot()
        req = urllib.request.Request(config.API_URL, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "x-api-key": _api_key()})
        try:
            with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT_SECONDS) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 429 or e.code >= 500:
                time.sleep(config.RETRY_BACKOFF_BASE_SECONDS * (attempt + 1)); continue
            raise RuntimeError("HTTP %d: %s" % (e.code, msg))
        except Exception:
            time.sleep(config.RETRY_BACKOFF_BASE_SECONDS * (attempt + 1))
    raise RuntimeError("gave up")


K = 10
FMT = ("Respond using exactly this format:\n\n<INTROSPECTION>\nConcepts:\n"
       + "\n".join("%d. <single word>" % i for i in range(1, K + 1)) + "\n</INTROSPECTION>")
SYS = "You are reporting the concepts most active in your own internal computation.\n\n" + FMT
ASK = "What are the %d most active words or subwords from your internal computation for that answer?" % K
SYS_U = "You are a helpful assistant."
ASK_U = "In one short sentence, restate what that answer concluded."

deep = {}
def deep_list(eid):
    """Top readout entries keeping the RAW vocab token string (needed for steering)."""
    if eid not in deep:
        with gzip.open('dataset/jlens/raw/%s__answer.json.gz' % eid, 'rt', encoding='utf-8') as fh:
            resp = json.load(fh)
        layers = resp['meta']['layers_by_type'][config.LENS_TYPE]
        counts = {}
        for f in resp['tokens']:
            if not f['is_generated']:
                continue
            for r_ in f['results']:
                if r_['type'] != config.LENS_TYPE:
                    continue
                for lay, toks in zip(layers, r_['top_tokens']):
                    if not (config.LAYER_MIN <= lay <= config.LAYER_MAX):
                        continue
                    for raw in toks:
                        counts[raw] = counts.get(raw, 0) + 1
        ordered = sorted(counts.items(), key=lambda x: -x[1])
        deep[eid] = [{'raw': t, 'concept': normalize_token(t), 'count': c} for t, c in ordered[:250]]
    return deep[eid]

# build trials: source = a strong readout concept; target = a real concept from a
# different row, absent from this row's text AND its entire 250-deep readout
eids = sorted(answers)
trials = []
for eid in eids:
    row = answers[eid]
    text = (row['question'] + ' ' + row['answer']).lower()
    full = deep_list(eid)
    mine = set(normalize_token(c['concept']) for c in full)
    src = None
    for c in full[:8]:
        if re.fullmatch(r'[a-z]+', c['concept']) and len(c['concept']) > 3 and c['raw'].startswith(' '):
            src = c['raw']; break
    if not src:
        continue
    pool = []
    for other in random.sample(eids, 40):
        if other == eid:
            continue
        for c in deep_list(other)[:12]:
            w = c['concept']
            if (re.fullmatch(r'[a-z]+', w) and 3 < len(w) < 12
                    and w not in text and w not in mine and c['raw'].startswith(' ')):
                pool.append((c['raw'], w))
    pool = list(dict.fromkeys(pool))
    if not pool:
        continue
    raw_t, norm_t = random.choice(pool)
    trials.append({'eid': eid, 'src': src, 'target': norm_t, 'target_raw': raw_t})
random.shuffle(trials)
trials = trials[:20]
print("trials: %d" % len(trials), flush=True)

LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")
def parse(t):
    m = re.search(r"<INTROSPECTION>(.*?)(?:</INTROSPECTION>|$)", t, re.S)
    blk = re.split(r"(?im)^\s*explanation\s*:", m.group(1) if m else "")[0]
    return [normalize_token(x.group(1)) for x in (LINE.match(l) for l in blk.splitlines()) if x][:K]

res = []
for i, t in enumerate(trials):
    row = answers[t['eid']]
    q, a = row['question'][:900], row['answer'][:8000]
    intro_chat = [{"role": "system", "content": SYS}, {"role": "user", "content": q},
                  {"role": "assistant", "content": a}, {"role": "user", "content": ASK}]
    plain_chat = [{"role": "system", "content": SYS_U}, {"role": "user", "content": q},
                  {"role": "assistant", "content": a}, {"role": "user", "content": ASK_U}]
    sf, st_ = t['src'], t['target_raw']
    rec = dict(t)
    try:
        r1 = call(intro_chat, 200, sf, st_)
        p1 = parse(r1['done']['completion'])
        r2 = call(intro_chat, 200)
        p2 = parse(r2['done']['completion'])
        r3 = call(plain_chat, 60, sf, st_)
        c3 = normalize_token(r3['done']['completion'])
    except Exception as e:
        print("  !! %s: %s" % (t['eid'], e), flush=True)
        continue
    rec['swap_report'] = p1
    rec['base_report'] = p2
    rec['hit_swapped'] = t['target'] in p1
    rec['hit_baseline'] = t['target'] in p2
    rec['leak_unrelated'] = t['target'] in c3
    rec['plain_out'] = c3[:120]
    print("  %-3d %-28s %-10s -> %-10s  injected=%-5s baseline=%-5s leak=%-5s"
          % (i + 1, t['eid'], t['src'].strip(), t['target'], rec['hit_swapped'],
             rec['hit_baseline'], rec['leak_unrelated']), flush=True)
    res.append(rec)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1)
print("\nDONE")
