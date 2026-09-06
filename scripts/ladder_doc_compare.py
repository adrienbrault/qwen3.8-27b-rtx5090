#!/usr/bin/env python3
"""R203: per-document comparison of two fidelity_ladder dense dumps (same corpus, same model, two engine configs).

fidelity_compare.py reports corpus-level PPL / top-1 / KL; that hides the failure mode vllm#53488 describes
(prompt_logprobs "wildly wrong for a subset of requests" under speculative decoding). This script scores each doc
in both dumps (mean NLL over the doc's positions, from tok_lp), prints the distribution of per-doc PPL deltas,
lists the worst docs, and counts positions whose token logprob moved by more than --pos-thresh nats.

Usage: ladder_doc_compare.py --a dump-ON-dense.jsonl --b dump-OFF-dense.jsonl [--doc-thresh 2.0] [--pos-thresh 1.0]
Exit 0 always; the verdict line is what the unit greps ("DOC-COMPARE").
"""
import argparse, json, math, sys
from collections import defaultdict


def load(path):
    docs = defaultdict(dict)  # doc -> pos -> tok_lp
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "doc" not in r or "pos" not in r or "tok_lp" not in r:
                continue
            docs[r["doc"]][r["pos"]] = r["tok_lp"]
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dump A (e.g. spec ON)")
    ap.add_argument("--b", required=True, help="dump B (e.g. spec OFF)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--doc-thresh", type=float, default=2.0, help="per-doc |PPL delta| %% flagged as an outlier")
    ap.add_argument("--pos-thresh", type=float, default=1.0, help="per-position |tok_lp delta| in nats flagged")
    ap.add_argument("--worst", type=int, default=10)
    a, b = load(ap.parse_args().a), load(ap.parse_args().b)
    args = ap.parse_args()
    common = sorted(set(a) & set(b))
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    rows, big_pos, n_pos, nll_a, nll_b = [], 0, 0, 0.0, 0.0
    for d in common:
        pa, pb = a[d], b[d]
        pos = sorted(set(pa) & set(pb))
        if not pos:
            continue
        sa = sum(pa[p] for p in pos); sb = sum(pb[p] for p in pos)
        nll_a -= sa; nll_b -= sb; n_pos += len(pos)
        big = sum(1 for p in pos if abs(pa[p] - pb[p]) > args.pos_thresh)
        big_pos += big
        ppl_a, ppl_b = math.exp(-sa / len(pos)), math.exp(-sb / len(pos))
        rows.append((100.0 * (ppl_b - ppl_a) / ppl_a, d, len(pos), ppl_a, ppl_b, big))
    if not rows:
        print("DOC-COMPARE: no common docs"); return
    deltas = sorted(r[0] for r in rows)
    n = len(deltas)
    q = lambda f: deltas[min(n - 1, int(f * n))]
    out_docs = [r for r in rows if abs(r[0]) > args.doc_thresh]
    print(f"docs common={n} only_{args.label_a}={only_a} only_{args.label_b}={only_b} positions={n_pos}")
    print(f"corpus PPL {args.label_a} {math.exp(nll_a / n_pos):.4f}  {args.label_b} {math.exp(nll_b / n_pos):.4f}  "
          f"delta({args.label_b} vs {args.label_a}) {100.0 * (math.exp(nll_b / n_pos) / math.exp(nll_a / n_pos) - 1):+.3f}%")
    print(f"per-doc PPL delta % ({args.label_b} vs {args.label_a}): min {deltas[0]:+.3f} p05 {q(0.05):+.3f} median {q(0.5):+.3f} "
          f"p95 {q(0.95):+.3f} max {deltas[-1]:+.3f}  mean {sum(deltas) / n:+.3f}")
    print(f"positions with |delta tok_lp| > {args.pos_thresh} nats: {big_pos} of {n_pos} ({100.0 * big_pos / n_pos:.4f}%)")
    print(f"worst {args.worst} docs by |delta|:")
    for r in sorted(rows, key=lambda r: -abs(r[0]))[:args.worst]:
        print(f"  doc {r[1]:>5} n={r[2]:>5} ppl {args.label_a} {r[3]:.4f} {args.label_b} {r[4]:.4f} delta {r[0]:+.3f}% big_pos {r[5]}")
    verdict = "OUTLIERS" if out_docs else "CLEAN"
    print(f"DOC-COMPARE {verdict}: {len(out_docs)} of {n} docs beyond ±{args.doc_thresh}% per-doc PPL; "
          f"{big_pos} positions beyond {args.pos_thresh} nats; median doc delta {q(0.5):+.3f}%")


if __name__ == "__main__":
    main()
