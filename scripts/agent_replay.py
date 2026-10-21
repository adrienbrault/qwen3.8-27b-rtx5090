#!/usr/bin/env python3
"""Replay the SHAPE of real agent traffic, so a decode number means something for the way the box is used.

Every probe in this repo so far measures one of two regimes: a single stream that owns the box, or N streams
started together with unique salted prompts. Production is neither. R585 showed the gap is arrival shape --
prefill interleaving with in-flight decode -- and R586 showed what the real shape is, over 9,926 requests and
7.03 hours:

    prompts      median 26,032 tokens, p90 64,648        97 % prefix-cached
    concurrency  4.21 mean streams decoding
    aggregate    306.3 t/s          per stream 72.8 t/s
    acceptance   72 % MTP

The 97 % is the part no existing probe reproduces, and it is not an incidental detail: it is the difference
between a prompt costing a full prefill and costing a few hundred new tokens. It happens because an agent
session resends a GROWING conversation -- step i's prompt is step i-1's prompt plus the model's reply plus the
tool output. Every request shares a long prefix with the previous one from the same session. A probe that
salts its prompts to be unique, which every probe here does deliberately so that pool tests measure footprint
rather than reuse, produces 0 % cached and therefore the wrong prefill cost, the wrong interleaving, and the
wrong decode rate.

So this replays sessions, not requests:

    session s:  base context (a synthetic repo blob, ~BASE tokens, unique per session)
                step 0: prompt = base + task          -> force G tokens out
                step 1: prompt = base + task + reply0 + tool_output0   -> force G tokens
                step k: ... and so on, the prompt growing by (G + TOOL) tokens per step

    sessions start staggered, and between steps a session sleeps THINK seconds to stand in for tool execution,
    which is what makes concurrency fluctuate instead of sitting at a constant.

This file only GENERATES the traffic. It deliberately does not compute the headline throughput: that is done
server-side by probes/tabby_log_agg.py, reading the container log, which is the same instrument that produced
the production numbers above. Measuring both sides with one instrument is the whole point -- a replay scored
by a different method could not be compared to the thing it is replaying.

  agent_replay.py --url http://127.0.0.1:8022/v1 --model <m> --sessions 6 --steps 12 --out replay.jsonl
"""
import argparse, itertools, json, math, random, sys, threading, time, urllib.request

PROSE = ("The maintainers agreed that the loader should verify every shard before the model is placed, because a "
         "silent mismatch costs more than a refused boot. The change touches the manifest reader, the placement "
         "planner and the two tests that cover them. ")
CODE = ("def resolve(self, name, *, strict=True):\n    entry = self._index.get(name)\n    if entry is None:\n"
        "        if strict:\n            raise KeyError(name)\n        return None\n    return entry.load()\n\n")


# Tokens per whitespace-separated word, MEASURED from R597's records rather than assumed. The old code used
# fn_bench's 1/1.6 for both kinds and was wrong by 30 % in opposite directions: R597's ten sessions realized
# exactly 0.703x the requested base on even session ids and 1.13x on odd ones, because `kind` is chosen by
# `sid % 2` and the two corpora tokenize differently -- CODE is punctuation-dense, PROSE is ordinary English.
# The error was deterministic, not random, so it did not average out; it split every round into two populations
# and made `--base` mean two different things depending on which session you looked at.
TOK_PER_WORD = {"prose": 1.125, "code": 1.80}


def filler(tokens, kind, rng):
    """Roughly `tokens` tokens of padding, drawn from rng so each session's base is its own token sequence."""
    unit = (CODE if kind == "code" else PROSE).split()
    if tokens <= 0:
        return ""
    return " ".join(rng.choice(unit) for _ in range(int(tokens / TOK_PER_WORD[kind]))) + " "


def post(url, model, prompt, ntok, timeout, temp):
    """One forced-length completion. Returns (ok, ttft, wall, gen_tokens, prompt_tokens, text, sse_frames, timing). SSE gaps are frame gaps, not token gaps under MTP."""
    body = {"model": model, "prompt": prompt, "max_tokens": ntok, "min_tokens": ntok,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; n = 0; text = []; usage = None; last_piece = None; gaps = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    c = json.loads(d)
                except Exception:
                    continue
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    piece = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.time()
                        if tf is None:
                            tf = now
                        if last_piece is not None:
                            gaps.append(now - last_piece)
                        last_piece = now
                        n += 1
                        text.append(piece)
    except Exception as e:
        return False, None, time.time() - t0, 0, None, f"ERROR {e}", 0, {}
    pt = (usage or {}).get("prompt_tokens")
    # Count generated tokens from the usage block, NOT from SSE frames. With MTP the server emits several
    # accepted tokens in one frame, so frames undercount by the acceptance factor: R593's client reported a
    # median of 202 where the server's own log said 608, almost exactly 3x, and a reader comparing the client
    # record against the forced length would conclude min_tokens was being ignored when it was being honoured.
    ct = (usage or {}).get("completion_tokens")
    timing = {"last_text_time": last_piece,
              "sse_gap_p95_s": sorted(gaps)[int((len(gaps)-1)*0.95)] if gaps else None,
              "sse_gap_max_s": max(gaps) if gaps else None,
              "decode_tps": (ct-1)/(last_piece-tf) if ct and tf and last_piece > tf else None,
              "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
    return True, (tf - t0 if tf else None), time.time() - t0, (ct if ct else n), pt, "".join(text), n, timing


def draw(sid, a, rng):
    """(base tokens, step count) for one session instance.

    WITH `--ladder` THIS IS DETERMINISTIC, AND THAT IS THE POINT. Two rounds were spent tuning a random draw
    against a single 10-session sample, which cannot work: resampling R597's own generator 400 times at n=10
    gives medians spanning 29.4k-35.1k and p90s spanning 50.6k-60.5k (p10-p90), and R597's actual draw --
    27.1k / 43.3k -- fell below the p10 of BOTH. Its parameters were not wrong; its sample was. Chasing that
    with a new distribution is the same error as R588's single-boot A/B, one level up.

    A fixed ladder removes the draw entirely. The enumerated set {base_i + k*G} over the ladder has whatever
    median and p90 you solve it for -- the shipped default matches production to 0.04 % and 0.01 % -- and it is
    IDENTICAL in every arm of every round, so a prompt-distribution difference can no longer masquerade as an
    effect. The random draw is kept behind --dist for the record, not because it should be used.
    """
    if a.ladder:
        return a.ladder[sid % len(a.ladder)]
    spread = max(0.0, a.spread)
    if a.dist == "lognormal":
        return (max(4000, int(a.base * math.exp(rng.gauss(0.0, spread)))),
                max(2, int(round(a.steps * math.exp(rng.gauss(0.0, spread * 0.6))))))
    return (max(4000, int(a.base * rng.uniform(1 - spread, 1 + spread))),
            max(2, int(round(a.steps * rng.uniform(1 - spread, 1 + spread)))))


def session(sid, a, rng, sink, lock, stop, inst=0):
    """One agent session: a growing conversation, with think time between steps."""
    base_t, nsteps = draw(sid, a, rng)
    base = filler(base_t, "code" if sid % 2 else "prose", rng)
    convo = base + "\n\nTask: find why the loader refuses the shard and propose a patch.\n"
    print(f"[s{sid}#{inst}] base ~{base_t} tokens, {nsteps} steps", flush=True)
    for step in range(nsteps):
        if stop.is_set():
            return
        gen = max(64, int(rng.gauss(a.gen, a.gen * 0.5)))
        t_sub = time.time()
        ok, ttft, wall, n, pt, text, frames, timing = post(a.url, a.model, convo, gen, a.timeout, a.temp)
        rec = {"session": sid, "inst": inst, "step": step, "steps_asked": nsteps, "base_asked": base_t,
               "ok": ok, "t_submit": t_sub, "ttft_s": ttft,
               "wall_s": wall, "gen_tokens": n, "asked_tokens": gen, "prompt_tokens": pt,
               "sse_frames": frames, "prompt_chars": len(convo), **timing}
        with lock:
            sink.write(json.dumps(rec) + "\n"); sink.flush()
            if ok:
                a.done_steps[sid] = a.done_steps.get(sid, 0) + 1
        if not ok:
            print(f"[s{sid}#{inst} step{step}] {text[:120]}", flush=True)
            return
        print(f"[s{sid}#{inst} step{step}/{nsteps}] prompt~{pt or '?'} gen {n} "
              f"ttft {ttft if ttft is None else round(ttft,2)} wall {wall:.1f}s", flush=True)
        # The reply and a synthetic tool result both become part of the next prompt. This is what makes the
        # next request share a long prefix with this one, and it is the whole reason this probe exists.
        convo += text + "\n\n[tool output]\n" + filler(a.tool, "code", rng) + "\n"
        if stop.wait(max(0.0, rng.gauss(a.think, a.think * 0.4))):
            return


# Solved against production (median 26,032, p90 64,648, 97 % cached) using the per-step growth MEASURED from
# R597's records: a step adds 2,703 prompt tokens (median over 140 consecutive pairs, mean 2,688, range
# 2,045-3,534). Under --respawn this ladder gives median 26,015, p90 64,560 and 93 % cached.
#
# IT IS SOLVED FOR --respawn, WHICH WEIGHTS SLOTS DIFFERENTLY FROM ONE PASS. Left to run once each, a slot
# contributes n_i requests, so deep sessions dominate; restarted on completion, every slot delivers requests at
# about the same RATE, so each contributes equally and the deep sessions lose weight. The same ladder read the
# wrong way is 15 % low on p90 -- solving for one regime and running the other is its own calibration bug.
#
# It is solved for G = 2703, so it is only valid while a step actually adds that much: --gen 700 --tool 2020
# reproduce it with the calibrated filler above. The 93 % against production's 97 % is the one target that does
# not fit: a slot spends one uncached prefill per pass, so the cached share is bounded by the session depths
# that the median and the p90 already fix. It is reported, not hidden -- the replay generates somewhat more
# fresh prefill work than production does, which makes it conservative for anything prefill-side.
LADDER_PROD = "3500:9,4000:9,5000:28,6000:23,7000:16,9000:28,9500:7,10500:23,12500:30,20500:19"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--base", type=int, default=22000, help="starting context per session, tokens")
    p.add_argument("--gen", type=int, default=700, help="mean forced generation length")
    p.add_argument("--tool", type=int, default=900, help="synthetic tool output appended per step, tokens")
    p.add_argument("--think", type=float, default=6.0, help="mean seconds between a session's steps")
    p.add_argument("--dist", choices=("lognormal","uniform"), default="lognormal",
                   help="shape of the per-session draw. lognormal is the default because production's prompt "
                        "distribution is skewed (median 26k, p90 65k) and a symmetric draw cannot hit both.")
    p.add_argument("--spread", type=float, default=0.5,
                   help="per-session variation in base size and step count, as a fraction (0 = every session "
                        "identical, which is what made R593's prompt p90 far too low)")
    p.add_argument("--ladder", default="",
                   help="fixed session shapes, 'base:steps,base:steps,...'. Overrides --base/--steps/--dist/"
                        "--spread and makes the prompt distribution identical in every arm. The default in "
                        "LADDER_PROD matches production's median and p90 to better than 0.1 %%.")
    p.add_argument("--prod-ladder", action="store_true", help="use the solved production ladder")
    p.add_argument("--max-fail-streak", type=int, default=3,
                   help="consecutive no-progress sessions on one slot before the whole run aborts. Guards "
                        "against respawning into a dead server, which cost R600 an arm and 14,245 requests.")
    p.add_argument("--drain", action="store_true",
                   help="at --max-seconds, stop STARTING sessions and let in-flight ones finish, instead of "
                        "cutting them. Without it the last instance of every slot is truncated by definition, "
                        "which removes exactly the deep-session tail the ladder was solved for: R599 lost 10 of "
                        "32 instances that way and its realized p90 landed 7 %% under the designed 64,560.")
    p.add_argument("--drain-grace", type=float, default=420,
                   help="hard cap on the drain, so a stuck session cannot hold the GPU past the round")
    p.add_argument("--respawn", action="store_true",
                   help="restart a session slot when it finishes, so concurrency does not decay as "
                        "the shallow ladder entries retire")
    p.add_argument("--stagger", type=float, default=4.0, help="seconds between session starts")
    p.add_argument("--temp", type=float, default=0.6, help="real clients sample; 0 would be the probe regime")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=900)
    a = p.parse_args()

    if a.prod_ladder and not a.ladder:
        a.ladder = LADDER_PROD
    if a.ladder:
        a.ladder = [tuple(int(x) for x in e.split(":")) for e in a.ladder.split(",") if e.strip()]
        a.sessions = len(a.ladder)

    # TWO SEEDS, DELIBERATELY. The ladder fixes the SHAPE of every round; the filler CONTENT must still be
    # unique per invocation or the NVMe prefix tier turns a fixed shape into a fixed cache hit. That is the
    # failure benchmark-seed-uniqueness records: a re-run served from the previous run's pages reads as a
    # speedup. So structure is deterministic and content is nonced, and the acceptance check is that step-0
    # prompts show ~0 % cached in both arms.
    seed = a.seed if a.seed is not None else int(time.time())
    nonce = int(time.time() * 1000) % 100000 if a.seed is None else seed
    shape = ("ladder " + ",".join(f"{b}:{s}" for b, s in a.ladder)) if a.ladder else \
            f"{a.sessions} sessions x {a.steps} steps, base {a.base}, {a.dist} spread {a.spread}"
    print(f"agent_replay: {shape}, gen ~{a.gen}, tool {a.tool}, think ~{a.think}s, stagger {a.stagger}s, "
          f"temp {a.temp}, seed {seed}, content nonce {nonce}", flush=True)
    sink = open(a.out, "a"); lock = threading.Lock(); stop = threading.Event(); abort = threading.Event()
    done_steps = {}
    a.done_steps = done_steps   # session() books completed steps here so worker() can tell progress from failure
    t0 = time.time(); deadline = t0 + a.max_seconds

    # A worker owns one ladder slot and RESTARTS it when it finishes. Without this the round decays: the solved
    # ladder has six sessions of <=9 steps and three of >=22, so the shallow ones retire in the first minutes
    # and the expensive tail prompts -- the whole object of study -- would arrive beside two streams instead of
    # production's four. Restarting keeps mean concurrency flat AND keeps the distribution exactly the ladder's,
    # because a restart replays the same slot.
    # A FAILED SESSION MUST NOT BE RESPAWNED INSTANTLY. R600 learned this the expensive way: the server OOM'd
    # in CUDA graph capture 15 minutes into an arm, every subsequent request returned "connection reset", each
    # session therefore returned immediately, and this loop started a new instance each time -- 14,245 instances
    # and 14,216 errors against 398 real steps, hammering a dead server for eight minutes while the round
    # believed it was measuring. The GPU is the scarcest resource here; a harness that cannot tell "finished"
    # from "the server is gone" wastes it at full speed.
    fail_streak = {}
    def worker(sid):
        for inst in itertools.count():
            if stop.is_set() or time.time() > deadline:
                return
            t_inst = time.time()
            before = done_steps.get(sid, 0)
            session(sid, a, random.Random(seed + sid * 7919 + inst * 104729 + nonce), sink, lock, stop, inst)
            if not a.respawn:
                return
            # "Completed no step at all" is the signature of a dead server, not of a short session: every ladder
            # entry has at least three steps.
            progressed = done_steps.get(sid, 0) > before
            fail_streak[sid] = 0 if progressed else fail_streak.get(sid, 0) + 1
            if fail_streak[sid] >= a.max_fail_streak:
                print(f"[s{sid}] {fail_streak[sid]} consecutive sessions made no progress -- stopping the run",
                      flush=True)
                abort.set(); stop.set()
                return
            if not progressed:
                # back off rather than spin, so a transient blip costs seconds instead of thousands of requests
                if stop.wait(min(30.0, 2.0 * fail_streak[sid])):
                    return
            elif time.time() - t_inst < 1.0:
                if stop.wait(1.0):
                    return

    threads = []
    for s in range(a.sessions):
        th = threading.Thread(target=worker, args=(s,), daemon=True)
        threads.append(th); th.start()
        if stop.wait(a.stagger):
            break
    while any(t.is_alive() for t in threads):
        if time.time() > deadline:
            if a.drain:
                # The deadline stops NEW sessions -- the worker loop already checks it before starting the next
                # instance -- and the in-flight ones are left to finish. Cutting them is what truncated a third
                # of R599's instances and pulled the realized p90 below the ladder's designed value, and the
                # loss is not random: it is always the deep slots, which are the only source of the tail.
                print(f"max-seconds reached, draining in-flight sessions (grace {a.drain_grace:.0f}s)", flush=True)
                hard = time.time() + a.drain_grace
                while any(t.is_alive() for t in threads) and time.time() < hard:
                    time.sleep(1)
                if any(t.is_alive() for t in threads):
                    print("drain grace expired, cutting what is left", flush=True)
                    stop.set()
            else:
                print("max-seconds reached, stopping sessions", flush=True)
                stop.set()
            break
        time.sleep(1)
    for t in threads:
        t.join(timeout=120)
    stop.set()
    print(f"agent_replay: {time.time()-t0:.0f} s wall. Score this with probes/tabby_log_agg.py over the "
          f"container log for this window -- not from this file.", flush=True)
    if abort.is_set():
        print("agent_replay: ABORTED -- the server stopped answering; this arm did not measure anything",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
