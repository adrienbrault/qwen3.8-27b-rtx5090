#!/usr/bin/env python3
"""Token-level fidelity of a served checkpoint on a FIXED corpus (R95 quant-quality yardstick).

One prefill pass per model: POST /v1/completions with prompt_logprobs=K, max_tokens=1 → per-token
logprob of the actual next token + the model's top-K.  Saved per chunk to a JSONL; `compare` mode
then scores a model against a reference run (near-lossless FP8): mean NLL, top-1 agreement,
truncated KL(ref||model) over the reference's top-K, per corpus bucket.

  fidelity.py build   --out corpus.jsonl --chunks 200 --tokens 2048 --tokenizer /model/dir  <files…>
  fidelity.py run     --url http://localhost:8029 --model qwen3.8-27b --corpus corpus.jsonl --out run-X.jsonl
  fidelity.py compare --ref run-fp8.jsonl run-A.jsonl run-B.jsonl …
"""
import argparse, json, math, sys, time, urllib.request, concurrent.futures as cf

def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out, n = open(a.out, "w"), 0
    per_file = max(1, a.chunks // max(1, len(a.files)))
    for path in a.files:
        bucket = "code" if path.endswith((".py", ".sh", ".ts", ".go", ".rs", ".c", ".cpp")) else "prose"
        if "tooleval" in path or "traj" in path or "transcript" in path: bucket = "agent"
        ids = tok(open(path, errors="ignore").read(), add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - a.tokens, a.tokens):
            if n >= a.chunks: break
            out.write(json.dumps({"id": n, "bucket": bucket, "src": path, "text": tok.decode(ids[i:i + a.tokens])}) + "\n"); n += 1
            if n % per_file == 0: break
    print(f"built {n} chunks x {a.tokens} tokens -> {a.out}")

def _post(url, body, timeout=600):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def run(a):
    chunks = [json.loads(l) for l in open(a.corpus)]
    def one(c):
        body = {"model": a.model, "prompt": c["text"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": a.topk, "logprobs": 1}
        for attempt in range(3):
            try:
                res = _post(a.url + "/v1/completions", body); break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(5)
        pl = res["choices"][0]["prompt_logprobs"]  # list (None for first token) of {tok_id: {logprob, rank, decoded_token}}
        toks = []
        for d in pl:
            if not d: continue
            # vLLM V1 puts the prompt token itself at position 0 of each dict (then the top-K by rank)
            k0 = next(iter(d)); actual = (int(k0), d[k0]["logprob"])
            top = sorted(((int(k), v["logprob"]) for k, v in d.items() if v.get("rank") is not None and v["rank"] <= a.topk), key=lambda x: -x[1])[: a.topk]
            toks.append({"t": actual[0], "lp": actual[1], "top": top})
        return {"id": c["id"], "bucket": c["bucket"], "toks": toks}
    with open(a.out, "w") as out, cf.ThreadPoolExecutor(a.conc) as ex:
        for i, r in enumerate(ex.map(one, chunks)):
            out.write(json.dumps(r) + "\n")
            if i % 20 == 0: print(f"{i+1}/{len(chunks)}", file=sys.stderr)
    print(f"ran {len(chunks)} chunks -> {a.out}")

def _load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p))}

def compare(a):
    ref = _load(a.ref)
    print(f"{'run':<28}{'bucket':<8}{'tokens':>8}{'NLL':>8}{'ΔNLL%':>8}{'top1agree':>11}{'KL(ref||m)':>12}")
    for p in a.runs:
        m = _load(p); agg = {}
        for cid, rr in ref.items():
            mm = m.get(cid)
            if not mm: continue
            for rt, mt in zip(rr["toks"], mm["toks"]):
                b = agg.setdefault(rr["bucket"], [0, 0.0, 0.0, 0, 0.0]); ba = agg.setdefault("ALL", [0, 0.0, 0.0, 0, 0.0])
                for x in (b, ba):
                    x[0] += 1; x[1] += -mt["lp"]; x[2] += -rt["lp"]
                    x[3] += int(rt["top"][0][0] == mt["top"][0][0]) if rt["top"] and mt["top"] else 0
                    # truncated KL over ref top-K: sum p_ref (log p_ref - log p_m), p_m from model's top-K (floor if absent)
                    mtop = dict(mt["top"]); floor = min(mtop.values()) - 2.0 if mtop else -20.0
                    x[4] += sum(math.exp(lp) * (lp - mtop.get(k, floor)) for k, lp in rt["top"])
        for bkt in sorted(agg, key=lambda k: (k != "ALL", k)):
            n, nll_m, nll_r, ag, kl = agg[bkt]
            print(f"{p.split('/')[-1][:27]:<28}{bkt:<8}{n:>8}{nll_m/n:>8.4f}{100*(nll_m-nll_r)/nll_r:>8.2f}{ag/n:>11.4f}{kl/n:>12.5f}")

ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
b = sp.add_parser("build"); b.add_argument("--out", required=True); b.add_argument("--chunks", type=int, default=200); b.add_argument("--tokens", type=int, default=2048); b.add_argument("--tokenizer", required=True); b.add_argument("files", nargs="+")
r = sp.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--model", required=True); r.add_argument("--corpus", required=True); r.add_argument("--out", required=True); r.add_argument("--topk", type=int, default=10); r.add_argument("--conc", type=int, default=4)
c = sp.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("runs", nargs="+")
a = ap.parse_args(); {"build": build, "run": run, "compare": compare}[a.cmd](a)
