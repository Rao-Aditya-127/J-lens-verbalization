# -*- coding: utf-8 -*-
"""Injection detection curve.

25 rows x 4 doses (swap applied to progressively wider layer bands) + a no-swap
baseline per row + a leakage probe at the strongest dose.

Records whether the injected concept is reported and at what rank, so we get a
graded detection curve rather than a single rate.
"""
import json, gzip, re, sys, io, random, urllib.request, urllib.error, time
sys.path.insert(0, 'dataset/jlens')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import config
from aggregate import normalize_token
from client import _api_key, _wait_for_rate_limit_slot

# Written to the repo root, which is where training/injection/plot_swap.py
# looks for it. Gitignored -- the repo stays code-only.
OUT = "injection_curve_results.json"
random.seed(23)
K = 10
N_ROWS = 25
# dose = width of the swapped layer band, centred on the workspace midpoint
DOSES = [("narrow", list(range(39, 44))),
         ("small",  list(range(35, 48))),
         ("wide",   list(range(30, 53))),
         ("full",   list(range(config.LAYER_MIN, config.LAYER_MAX + 1)))]

answers = {json.loads(l)['example_id']: json.loads(l)
           for l in open('dataset/jlens/collected_answers.jsonl', encoding='utf-8')}


def call(chat, ntok, steer_from=None, swap_to=None, layers=None):
    body = {"modelId": config.MODEL_ID, "chat": chat, "type": [config.LENS_TYPE],
            "topN": config.TOP_N_API, "temperature": config.TEMPERATURE,
            "numCompletionTokens": ntok, "filterNonWordTokens": config.FILTER_NON_WORD_TOKENS,
            "stream": False}
    if steer_from is not None:
        body["steerTokens"] = [{"token": steer_from, "type": config.LENS_TYPE}]
        body["swapToken"] = {"token": swap_to, "type": config.LENS_TYPE}
        body["steerLayers"] = layers
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


FMT = ("Respond using exactly this format:\n\n<INTROSPECTION>\nConcepts:\n"
       + "\n".join("%d. <single word>" % i for i in range(1, K + 1)) + "\n</INTROSPECTION>")
SYS = "You are reporting the concepts most active in your own internal computation.\n\n" + FMT
ASK = "What are the %d most active words or subwords from your internal computation for that answer?" % K
SYS_U = "You are a helpful assistant."
ASK_U = "In one short sentence, restate what that answer concluded."

deep = {}
def deep_list(eid):
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
        deep[eid] = [{'raw': t, 'concept': normalize_token(t), 'count': c}
                     for t, c in sorted(counts.items(), key=lambda x: -x[1])[:250]]
    return deep[eid]


eids = sorted(answers)
trials = []
for eid in eids:
    row = answers[eid]
    text = (row['question'] + ' ' + row['answer']).lower()
    full = deep_list(eid)
    mine = set(c['concept'] for c in full)
    src = next((c['raw'] for c in full[:8]
                if re.fullmatch(r'[a-z]+', c['concept']) and len(c['concept']) > 3
                and c['raw'].startswith(' ')), None)
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
trials = trials[:N_ROWS]
print("rows: %d   doses: %s   calls: %d"
      % (len(trials), [d[0] for d in DOSES], len(trials) * (len(DOSES) + 2)), flush=True)

LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")
def parse(t):
    m = re.search(r"<INTROSPECTION>(.*?)(?:</INTROSPECTION>|$)", t, re.S)
    blk = re.split(r"(?im)^\s*explanation\s*:", m.group(1) if m else "")[0]
    return [normalize_token(x.group(1)) for x in (LINE.match(l) for l in blk.splitlines()) if x][:K]

res = []
for i, t in enumerate(trials):
    row = answers[t['eid']]
    q, a = row['question'][:900], row['answer'][:8000]
    intro = [{"role": "system", "content": SYS}, {"role": "user", "content": q},
             {"role": "assistant", "content": a}, {"role": "user", "content": ASK}]
    plain = [{"role": "system", "content": SYS_U}, {"role": "user", "content": q},
             {"role": "assistant", "content": a}, {"role": "user", "content": ASK_U}]
    rec = dict(t); rec['doses'] = {}
    try:
        for name, layers in DOSES:
            p = parse(call(intro, 200, t['src'], t['target_raw'], layers)['done']['completion'])
            rec['doses'][name] = {'hit': t['target'] in p,
                                  'rank': p.index(t['target']) + 1 if t['target'] in p else None,
                                  'n_layers': len(layers)}
        pb = parse(call(intro, 200)['done']['completion'])
        rec['baseline_hit'] = t['target'] in pb
        cl = normalize_token(call(plain, 60, t['src'], t['target_raw'], DOSES[-1][1])['done']['completion'])
        rec['leak'] = t['target'] in cl
    except Exception as e:
        print("  !! %s: %s" % (t['eid'], e), flush=True)
        continue
    marks = " ".join("%s=%s" % (n, "Y" if rec['doses'][n]['hit'] else ".") for n, _ in DOSES)
    print("  %-3d %-28s ->%-11s %s  base=%s leak=%s"
          % (i + 1, t['eid'][:28], t['target'], marks,
             "Y" if rec['baseline_hit'] else ".", "Y" if rec['leak'] else "."), flush=True)
    res.append(rec)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1)
print("\nDONE")
