#!/usr/bin/env python3
"""R156 agentic-regime fidelity: bf16 generates the reference, quant arms are teacher-forced on it.

Why this design (see r156-REVIEW.md §8): scoring our own transcripts would favour whichever
model authored them. Here the TRUE model (bf16) generates greedy responses to pinned,
chat-templated prompts (tools + thinking = the deployed regime), and every quantized arm is
scored on bf16's exact TOKEN IDS. Alignment is exact by construction (ids, not re-tokenised
text), and the question answered is: how faithfully does each quant reproduce the true
model's own trajectory on the work this box actually does?

  gen   (bf16 arm):   agentic_ref.py gen   --url U --prompts P.jsonl --tokenizer DIR --out-ref ref.jsonl --out-ids ids.jsonl
  score (quant arm):  agentic_ref.py score --url U --ids ids.jsonl --out dump.jsonl

Dumps use the fidelity_ladder.py record format, so fidelity_compare.py works unchanged:
  {"doc","pos","tok","tok_lp","argmax","p1","top"}  with tok/argmax as "token_id:N" strings.
Only RESPONSE positions are written (pos >= prompt_len), so the (doc,pos) join is the response.
"""
import argparse, json, math, sys, threading, time, urllib.request

def post(url, payload, timeout=1800, retries=3):
    body = json.dumps(payload).encode(); last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(3 * (i + 1))
    raise RuntimeError(f"request failed: {last}")

def tid(s):
    """'token_id:123' -> 123 (fail loud if the server did not honour return_tokens_as_token_ids)."""
    if not (isinstance(s, str) and s.startswith("token_id:")):
        raise RuntimeError(f"expected token_id:N string, got {s!r} — return_tokens_as_token_ids not honoured")
    return int(s[9:])

def records_from_logprobs(doc, base_pos, toks, tlps, tops, start):
    """Emit dump records for positions >= start (index within the scored sequence)."""
    out = []
    for i in range(start, len(toks)):
        if i >= len(tlps) or tlps[i] is None: continue
        top = tops[i] if i < len(tops) and tops[i] else None
        if not top: continue
        items = sorted(top.items(), key=lambda kv: -kv[1]); argmax, best = items[0]
        out.append({"doc": doc, "pos": base_pos + i, "tok": toks[i], "tok_lp": tlps[i],
                    "argmax": argmax, "p1": math.exp(best), "top": [[t, lp] for t, lp in items]})
    return out

def smoke(url, model):
    r = post(f"{url}/v1/completions", {"model": model, "prompt": [785, 3974, 13876], "max_tokens": 2, "temperature": 0,
                                        "logprobs": 3, "echo": True, "return_tokens_as_token_ids": True})
    if "error" in r: raise RuntimeError(f"smoke: {str(r['error'])[:200]}")
    lp = r["choices"][0]["logprobs"]; tid(lp["tokens"][0]); tid(next(iter(lp["top_logprobs"][1])))
    print(f"[smoke] ok: id-list prompt + return_tokens_as_token_ids honoured (usage={r.get('usage')})", flush=True)

def run_gen(a):
    import warnings; warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
    smoke(a.url, a.model)
    lock = threading.Lock(); done = [0]; t0 = time.time()
    fref = open(a.out_ref, "w"); fids = open(a.out_ids, "w")
    def work(p):
        text = tok.apply_chat_template(p["messages"], tools=p.get("tools"), tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        r = post(f"{a.url}/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0,
                                              "logprobs": 20, "return_tokens_as_token_ids": True})
        if "error" in r: raise RuntimeError(f"doc {p['id']}: {str(r['error'])[:200]}")
        ch = r["choices"][0]; lp = ch["logprobs"]
        gen_ids = [tid(t) for t in lp["tokens"]]
        recs = records_from_logprobs(p["id"], len(ids), lp["tokens"], lp["token_logprobs"], lp["top_logprobs"], 0)
        with lock:
            for rec in recs: fref.write(json.dumps(rec) + "\n")
            fids.write(json.dumps({"id": p["id"], "kind": p["kind"], "prompt_len": len(ids), "prompt_ids": ids, "gen_ids": gen_ids,
                                   "finish_reason": ch.get("finish_reason"), "text": ch.get("text", "")}) + "\n")
            fref.flush(); fids.flush(); done[0] += 1
            print(f"[gen] {done[0]}/{len(prompts)} doc={p['id']} kind={p['kind']} prompt={len(ids)} gen={len(gen_ids)} "
                  f"finish={ch.get('finish_reason')} ({time.time()-t0:.0f}s)", flush=True)
    # bounded concurrency: reference generation is not batch-sensitive at the 0.003% floor
    sem = threading.Semaphore(a.conc); errs = []
    def wrap(p):
        with sem:
            try: work(p)
            except Exception as e:  # noqa: BLE001
                errs.append((p["id"], str(e)[:160])); print(f"[gen] FAIL doc={p['id']}: {str(e)[:160]}", flush=True)
    ths = [threading.Thread(target=wrap, args=(p,)) for p in prompts]
    for t in ths: t.start()
    for t in ths: t.join()
    fref.close(); fids.close()
    print(f"DONE gen docs={done[0]} failed={len(errs)} elapsed={time.time()-t0:.0f}s", flush=True)
    return 1 if done[0] == 0 or len(errs) > len(prompts) // 10 else 0

def run_score(a):
    docs = [json.loads(l) for l in open(a.ids) if l.strip()]
    smoke(a.url, a.model); t0 = time.time(); n = 0; fails = 0
    with open(a.out, "w") as out:
        for d in docs:
            seq = d["prompt_ids"] + d["gen_ids"]
            if not d["gen_ids"]: continue
            try:
                r = post(f"{a.url}/v1/completions", {"model": a.model, "prompt": seq, "max_tokens": 0, "temperature": 0,
                                                      "logprobs": 20, "echo": True, "return_tokens_as_token_ids": True})
                if "error" in r: raise RuntimeError(str(r["error"])[:200])
                lp = r["choices"][0]["logprobs"]
                # guard: the echoed ids must be exactly what we sent
                got = [tid(t) for t in lp["tokens"]]
                if got != seq: raise RuntimeError(f"echo mismatch (sent {len(seq)} ids, got {len(got)}, first diff at "
                                                  f"{next((i for i,(x,y) in enumerate(zip(seq,got)) if x!=y), 'len')})")
                recs = records_from_logprobs(d["id"], 0, lp["tokens"], lp["token_logprobs"], lp["top_logprobs"], d["prompt_len"])
                for rec in recs: out.write(json.dumps(rec) + "\n")
                out.flush(); n += len(recs)
                print(f"[score] doc={d['id']} kind={d['kind']} positions={len(recs)} total={n} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"[score] FAIL doc={d['id']}: {str(e)[:160]}", flush=True)
    print(f"DONE score docs={len(docs)-fails} failed={fails} positions={n} elapsed={time.time()-t0:.0f}s", flush=True)
    return 1 if n == 0 or fails > len(docs) // 10 else 0

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen"); g.add_argument("--url", required=True); g.add_argument("--model", default="qwen3.8-27b")
    g.add_argument("--prompts", required=True); g.add_argument("--tokenizer", required=True)
    g.add_argument("--out-ref", required=True); g.add_argument("--out-ids", required=True)
    g.add_argument("--max-tokens", type=int, default=2048); g.add_argument("--conc", type=int, default=4)
    s = sub.add_parser("score"); s.add_argument("--url", required=True); s.add_argument("--model", default="qwen3.8-27b")
    s.add_argument("--ids", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    return run_gen(a) if a.mode == "gen" else run_score(a)

if __name__ == "__main__":
    sys.exit(main())
