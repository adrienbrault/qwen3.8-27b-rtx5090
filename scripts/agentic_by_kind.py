#!/usr/bin/env python3
"""R156 agentic regime — per-kind breakdown of the bf16-reference comparison.

The agentic corpus (build_agentic_prompts.py) pins doc ids by kind:
    tool 0-31 | code 100-115 | reason 200-215 | prose 300-307
This splits the fidelity dumps by kind and runs fidelity_compare.py on each slice,
so a checkpoint's fidelity on tool-call turns is not averaged away by long code/prose
generations (code+prose are ~80% of scored positions by token count).

Usage: agentic_by_kind.py --ref dump-a1-agentic.jsonl --arm dump-c1-agentic.jsonl --label c [--arm2 dump-h1-agentic.jsonl --label2 h]
Prints one summary row per (kind, arm): positions, top-1 agreement, PPL delta, certain/near-tie flip rates.
"""
import argparse, json, os, re, subprocess, sys, tempfile

KINDS = {"tool": range(0, 100), "code": range(100, 200), "reason": range(200, 300), "prose": range(300, 400)}
HERE = os.path.dirname(os.path.abspath(__file__))


def split(path, tmpdir, tag):
    outs = {k: open(os.path.join(tmpdir, f"{tag}-{k}.jsonl"), "w") for k in KINDS}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)["doc"]
            except (json.JSONDecodeError, KeyError):
                continue
            for k, rng in KINDS.items():
                if d in rng:
                    outs[k].write(line)
                    break
    for o in outs.values():
        o.close()
    return {k: os.path.join(tmpdir, f"{tag}-{k}.jsonl") for k in KINDS}


def compare(ref, arm, label):
    out = subprocess.run([sys.executable, os.path.join(HERE, "fidelity_compare.py"),
                          "--ref", ref, "--arm", arm, "--label", label],
                         capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out) or [None, None])[1]
    return {
        "n": g(r"scored positions: ([\d,]+)"),
        "agree": g(r"top-1 agreement: ([\d.]+)%"),
        "ppl": g(r"delta ([+-][\d.]+)%"),
        "certain": g(r"certain\s+[\d,]+\s+[\d,]+\s+([\d.]+)%"),
        "confident": g(r"confident\s+[\d,]+\s+[\d,]+\s+([\d.]+)%"),
        "moderate": g(r"moderate\s+[\d,]+\s+[\d,]+\s+([\d.]+)%"),
        "neartie": g(r"near-tie\s+[\d,]+\s+[\d,]+\s+([\d.]+)%"),
        "raw": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--arm2")
    ap.add_argument("--label2", default="arm2")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = [(a.label, a.arm)] + ([(a.label2, a.arm2)] if a.arm2 else [])
    res = {}
    with tempfile.TemporaryDirectory() as td:
        refs = split(a.ref, td, "ref")
        print(f"{'kind':<7} {'arm':<4} {'positions':>10} {'top1-agree':>11} {'PPL delta':>10} {'certain':>8} {'confid':>8} {'moder':>8} {'neartie':>8}")
        for k in KINDS:
            for lab, path in arms:
                arm_split = split(path, td, f"arm-{lab}")
                r = compare(refs[k], arm_split[k], f"{k}:{lab}")
                res[f"{k}:{lab}"] = r
                print(f"{k:<7} {lab:<4} {r['n'] or '-':>10} {(r['agree'] or '-')+'%':>11} {(r['ppl'] or '-')+'%':>10} "
                      f"{(r['certain'] or '-')+'%':>8} {(r['confident'] or '-')+'%':>8} {(r['moderate'] or '-')+'%':>8} {(r['neartie'] or '-')+'%':>8}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
