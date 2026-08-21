#!/usr/bin/env python3
"""Depth-needle retrieval probe (cold + warm, optional concurrent loaders).

The failure mode this exists for is FLUENT-BUT-WRONG recall (a 4-bit KV layout/scale bug
makes the model answer confidently with the wrong secret) — so it scores EXACT secret match,
not "looks fine". Each (depth, sample) gets its own random secret and a fresh random filler
seed (cold = no prefix reuse); `--warm` re-asks the same prompt right after (prefix-cache /
L1 path). `--loaders N` keeps N background 20K-token chats running during the probe so the
needle is measured under concurrency (the "sean gate": cold + cached multi-request 32K+
recall under MTP on the exact selected backend).

usage: needle_depth.py --url http://localhost:8029/v1 --model qwen3.8-27b \
         --depths 9000 20000 40000 60000 100000 --samples 3 --loaders 4 --warm --out needles.jsonl
exit code 0 iff every sample hit.
"""
import argparse, concurrent.futures as cf, json, random, string, sys, threading, time, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu river mountain "
         "harbor lantern copper meadow orbit velvet canyon ember glacier marble nectar ozone pepper").split()


def filler(n_tokens: int, rng: random.Random) -> str:
    # ~1.3 tokens/word for this tokenizer family -> aim low, length is measured from usage anyway
    n_words = int(n_tokens / 1.3)
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 17 == 16:
            out.append(".")
    return " ".join(out)


def secret(rng: random.Random) -> str:
    return "-".join(["".join(rng.choices(string.ascii_uppercase, k=6)), "".join(rng.choices(string.digits, k=4))])


def chat(url, model, messages, max_tokens, timeout, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if extra:
        body.update(extra)
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "") + " " + (msg.get("reasoning") or msg.get("reasoning_content") or "")
    return content, dt, d.get("usage", {})


def one_probe(args, depth, sample, warm_pass=False):
    rng = random.Random(f"{args.seed}-{depth}-{sample}")
    sec = secret(rng)
    body = filler(depth, rng)
    cut = int(len(body) * args.position)
    doc = body[:cut] + f"\n\nIMPORTANT RECORD: the vault access code is {sec}. Memorize it.\n\n" + body[cut:]
    messages = [
        {"role": "system", "content": "You are a precise assistant. Answer with the exact code only."},
        {"role": "user", "content": doc + "\n\nQuestion: what is the vault access code stated in the document above? Reply with the code only."},
    ]
    extra = {"chat_template_kwargs": {"reasoning_effort": args.effort}} if args.effort else None
    res = {"depth": depth, "sample": sample, "secret": sec}
    try:
        out, dt, usage = chat(args.url, args.model, messages, 64, args.timeout, extra)
        res.update(cold_s=round(dt, 2), prompt_tokens=usage.get("prompt_tokens"), cold_hit=sec in out,
                   cold_answer=out.strip()[:120])
        if args.warm:
            out2, dt2, _ = chat(args.url, args.model, messages, 64, args.timeout, extra)
            res.update(warm_s=round(dt2, 2), warm_hit=sec in out2, warm_answer=out2.strip()[:120])
    except Exception as e:  # noqa: BLE001
        res.update(error=repr(e)[:200], cold_hit=False)
    return res


def loader(args, stop: threading.Event, idx: int):
    rng = random.Random(f"loader-{args.seed}-{idx}")
    while not stop.is_set():
        body = filler(args.loader_tokens, rng)
        try:
            chat(args.url, args.model, [{"role": "user", "content": body + "\n\nSummarize the above in one sentence."}],
                 200, args.timeout)
        except Exception:  # noqa: BLE001
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8029/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--depths", type=int, nargs="+", default=[9000, 20000, 40000, 60000])
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--position", type=float, default=0.6, help="needle position as fraction of the filler")
    ap.add_argument("--loaders", type=int, default=0)
    ap.add_argument("--loader-tokens", type=int, default=20000)
    ap.add_argument("--parallel", type=int, default=1, help="probe requests in flight at once")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--effort", default=None, help="reasoning_effort template kwarg (low|medium|xhigh)")
    ap.add_argument("--seed", default="nvfp4kv")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stop = threading.Event()
    threads = [threading.Thread(target=loader, args=(args, stop, i), daemon=True) for i in range(args.loaders)]
    for t in threads:
        t.start()
    if threads:
        time.sleep(5)

    jobs = [(d, s) for d in args.depths for s in range(args.samples)]
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for res in ex.map(lambda j: one_probe(args, *j), jobs):
            results.append(res)
            flag = "HIT " if res.get("cold_hit") else "MISS"
            wflag = "" if not args.warm else (" warm=HIT" if res.get("warm_hit") else " warm=MISS")
            print(f"[{flag}] depth={res['depth']:>7} s{res['sample']} cold={res.get('cold_s')}s "
                  f"ptok={res.get('prompt_tokens')}{wflag} ans={res.get('cold_answer', res.get('error'))!r}", flush=True)
    stop.set()

    hits = sum(1 for r in results if r.get("cold_hit"))
    whits = sum(1 for r in results if r.get("warm_hit")) if args.warm else None
    summary = {"total": len(results), "cold_hits": hits, "warm_hits": whits, "loaders": args.loaders,
               "depths": args.depths, "samples": args.samples, "url": args.url, "model": args.model,
               "effort": args.effort, "seed": args.seed}
    print("SUMMARY", json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
    ok = hits == len(results) and (whits is None or whits == len(results))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
