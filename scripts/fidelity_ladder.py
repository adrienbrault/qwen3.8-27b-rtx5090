#!/usr/bin/env python3
"""R156 fidelity scorer — teacher-forced per-position logprobs for one ladder arm.

Scores a FIXED corpus against a running vLLM arm using the OpenAI-compat echo path
(verified on our v0.28 build): POST /v1/completions with
    {"max_tokens": 0, "echo": true, "logprobs": K, "temperature": 0}
which returns, for every prompt position: the forced token, its logprob, and the
top-K distribution. Teacher forcing is essential — free generation diverges at the
first disagreement and per-position comparison becomes meaningless.

Emits one JSONL record per scored position (position 0 has no prediction, skipped):

    {"doc": int, "pos": int, "tok": str, "tok_lp": float,
     "argmax": str, "p1": float, "top": [[tok, logprob], ...]}

Run at concurrency 1: vLLM logprobs are not batch-invariant, and at the 0.1% event
rates this campaign resolves, batching noise can masquerade as quantization error.

Usage:
  fidelity_ladder.py --url http://127.0.0.1:8029 --model qwen3.8-27b \
      --corpus corpus.jsonl --out dump-C.jsonl [--logprobs 20] [--limit N] [--resume]
"""
import argparse, json, math, os, sys, time, urllib.error, urllib.request


def post(url, payload, timeout, retries=3):
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - transient server/network errors
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def score_doc(url, model, text, logprobs, timeout):
    r = post(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": logprobs,
            "temperature": 0,
        },
        timeout,
    )
    if "error" in r:
        raise RuntimeError(f"server error: {str(r['error'])[:300]}")
    lp = r["choices"][0].get("logprobs")
    if not lp:
        raise RuntimeError("no logprobs in response (echo path unsupported?)")
    return lp["tokens"], lp["token_logprobs"], lp.get("top_logprobs") or []


def next_token(url, model, prefix, logprobs, timeout):
    """Sparse depth probe: score ONE next-token prediction after a long prefix.

    Only one position's logits materialize, so memory is trivial and the prefix may be
    arbitrarily long — unlike echo/prompt_logprobs, whose cost is N_tokens x vocab x 4
    bytes (7K tokens x 152K vocab = 4.2 GB, which OOM'd an engine during R156 setup).
    """
    r = post(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": prefix,
            "max_tokens": 1,
            "echo": False,
            "logprobs": logprobs,
            "temperature": 0,
        },
        timeout,
    )
    if "error" in r:
        raise RuntimeError(f"server error: {str(r['error'])[:300]}")
    lp = r["choices"][0].get("logprobs")
    if not lp or not lp.get("top_logprobs"):
        raise RuntimeError("no top_logprobs in generation response")
    return lp["top_logprobs"][0]


def run_depth2_mode(a, docs, out):
    """POWERED depth probe via shared prefixes + prefix caching.

    The naive design (one long prefill per probe) costs depth x samples tokens and is why the
    first depth run was stuck at 40 samples/bin with unusable CIs. Here, for each (depth, base)
    we prefill ONE long context and then probe many short continuations off it. Prefix caching
    is exact -- identical KV, identical logits -- so this costs nothing in rigor while cutting
    cost by ~samples-per-base (~200x), which is what makes 200K depths and ~250 samples/bin
    affordable.

    Depth labels are the MEASURED prompt_tokens from the server (not a chars/token estimate),
    and prefixes are built to undershoot so nothing clips against max-len. All arms build byte
    identical prefixes from the pinned corpus, so their token counts match exactly.
    """
    depths = [int(x) for x in a.depths.split(",")]
    corpus_text = [d["text"] for d in docs]
    n = 0
    for depth_tok in depths:
        for base in range(a.depth_bases):
            need = int(depth_tok * 3.0)  # undershoot: never clip against max-len
            buf, i = "", base * 13       # stride 13 => distinct content per base
            while len(buf) < need:
                buf += corpus_text[i % len(corpus_text)] + "\n\n"
                i += 1
            prefix = buf[:need]
            try:  # warm the prefix cache once; result discarded
                next_token(a.url, a.model, prefix, a.logprobs, a.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] depth={depth_tok} base={base} warm failed: {str(e)[:120]}", flush=True)
                continue
            for c in range(a.depth_conts):
                cont = corpus_text[(base * 13 + 7 + c) % len(corpus_text)][:160]
                try:
                    top, ptok = next_token_with_usage(
                        a.url, a.model, prefix + cont, a.logprobs, a.timeout
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] depth={depth_tok} base={base} cont={c}: {str(e)[:110]}", flush=True)
                    continue
                items = sorted(top.items(), key=lambda kv: -kv[1])
                argmax, best_lp = items[0]
                out.write(json.dumps({
                    "doc": base * 10000 + c,
                    "pos": depth_tok,            # nominal bin
                    "actual_tokens": ptok,       # measured prefix length
                    "tok": f"d{depth_tok}b{base}c{c}",   # alignment fingerprint
                    "tok_lp": None,
                    "argmax": argmax, "p1": math.exp(best_lp),
                    "top": [[t, lp] for t, lp in items],
                }) + "\n")
                n += 1
            out.flush()
            print(f"[depth2] depth={depth_tok} base={base+1}/{a.depth_bases} records={n}", flush=True)
    return n


def next_token_with_usage(url, model, prefix, logprobs, timeout):
    r = post(f"{url}/v1/completions", {
        "model": model, "prompt": prefix, "max_tokens": 1,
        "echo": False, "logprobs": logprobs, "temperature": 0,
    }, timeout)
    if "error" in r:
        raise RuntimeError(f"server error: {str(r['error'])[:300]}")
    lp = r["choices"][0].get("logprobs")
    if not lp or not lp.get("top_logprobs"):
        raise RuntimeError("no top_logprobs in generation response")
    ptok = (r.get("usage") or {}).get("prompt_tokens")
    return lp["top_logprobs"][0], ptok


def run_depth_mode(a, docs, out):
    """Build deterministic long prefixes from the corpus, probe one token at each depth.

    Records are written as doc=sample, pos=depth so the comparator's existing
    depth-binning and (doc,pos) join work unchanged. `tok` carries a prefix
    fingerprint so the alignment guard still verifies both arms saw identical text.
    """
    depths = [int(x) for x in a.depths.split(",")]
    corpus_text = [d["text"] for d in docs]
    n_pos = 0
    for sample in range(a.depth_samples):
        for depth_tok in depths:
            # ~3.5 chars/token; deterministic start offset per sample
            need = int(depth_tok * 3.5)
            buf, i = "", sample * 7  # stride 7 => different content per sample
            while len(buf) < need:
                buf += corpus_text[i % len(corpus_text)] + "\n\n"
                i += 1
            prefix = buf[:need]
            fp = f"len{len(prefix)}:{prefix[-24:]!r}"
            try:
                top = next_token(a.url, a.model, prefix, a.logprobs, a.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] depth={depth_tok} sample={sample} failed: {str(e)[:140]}", flush=True)
                continue
            items = sorted(top.items(), key=lambda kv: -kv[1])
            argmax, best_lp = items[0]
            out.write(json.dumps({
                "doc": sample, "pos": depth_tok, "tok": fp, "tok_lp": None,
                "argmax": argmax, "p1": math.exp(best_lp),
                "top": [[t, lp] for t, lp in items],
            }) + "\n")
            n_pos += 1
        out.flush()
        print(f"[depth] sample {sample+1}/{a.depth_samples} done, records={n_pos}", flush=True)
    return n_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--corpus", required=True, help="JSONL of {id, text, domain}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logprobs", type=int, default=20, help="top-K (vLLM default max 20)")
    ap.add_argument("--limit", type=int, default=0, help="score only first N docs (smoke)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--resume", action="store_true", help="skip docs already in --out")
    ap.add_argument("--mode", choices=["dense", "depth", "depth2"], default="dense",
                    help="dense = teacher-forced over corpus docs; depth = sparse "
                         "single-token probe after long prefixes")
    ap.add_argument("--depths", default="1000,4000,16000,64000,128000")
    ap.add_argument("--depth-samples", type=int, default=40)
    ap.add_argument("--depth-bases", type=int, default=5, help="depth2: distinct long prefixes per depth")
    ap.add_argument("--depth-conts", type=int, default=50, help="depth2: probes sharing each prefix")
    a = ap.parse_args()

    docs = []
    with open(a.corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if a.limit:
        docs = docs[: a.limit]

    done = set()
    if a.resume and os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc"])
                except Exception:  # noqa: BLE001 - tolerate a torn final line
                    pass
        print(f"[resume] {len(done)} docs already scored", flush=True)

    t0 = time.time()
    n_pos = n_docs = n_fail = 0

    if a.mode in ("depth", "depth2"):
        runner = run_depth2_mode if a.mode == "depth2" else run_depth_mode
        with open(a.out, "a" if a.resume else "w") as out:
            n_pos = runner(a, docs, out)
        print(f"DONE mode={a.mode} records={n_pos:,} elapsed={time.time()-t0:.0f}s -> {a.out}", flush=True)
        return 0 if n_pos else 1

    with open(a.out, "a" if a.resume else "w") as out:
        for i, d in enumerate(docs):
            did = d["id"]
            if did in done:
                continue
            try:
                toks, tlps, tops = score_doc(a.url, a.model, d["text"], a.logprobs, a.timeout)
            except Exception as e:  # noqa: BLE001 - log and continue; one bad doc != lost run
                n_fail += 1
                print(f"[warn] doc {did} failed: {str(e)[:160]}", flush=True)
                continue
            for p in range(len(toks)):
                # position 0 has no preceding context -> no prediction
                if p >= len(tlps) or tlps[p] is None:
                    continue
                top = tops[p] if p < len(tops) and tops[p] else None
                if not top:
                    continue
                items = sorted(top.items(), key=lambda kv: -kv[1])
                argmax, best_lp = items[0]
                out.write(
                    json.dumps(
                        {
                            "doc": did,
                            "pos": p,
                            "tok": toks[p],
                            "tok_lp": tlps[p],
                            "argmax": argmax,
                            "p1": math.exp(best_lp),
                            "top": [[t, lp] for t, lp in items],
                        }
                    )
                    + "\n"
                )
                n_pos += 1
            n_docs += 1
            out.flush()
            if n_docs % 20 == 0:
                el = time.time() - t0
                print(
                    f"[{n_docs}/{len(docs)}] positions={n_pos:,} "
                    f"({n_pos/max(el,1e-9):.0f} pos/s, {el:.0f}s elapsed)",
                    flush=True,
                )

    el = time.time() - t0
    print(
        f"DONE docs={n_docs} failed={n_fail} positions={n_pos:,} "
        f"elapsed={el:.0f}s -> {a.out}",
        flush=True,
    )
    return 1 if (n_docs == 0 or n_fail > len(docs) // 10) else 0


if __name__ == "__main__":
    sys.exit(main())
