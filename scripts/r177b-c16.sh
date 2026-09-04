#!/usr/bin/env bash
# R177b (2026-09-04): addendum to r177-matrix.sh on the live daily (:8020, SEQS 16, traffic only, no GPU lock).
# (1) code-c16 re-run with 1024 tokens per stream: at 512 tokens decode_ss.py found no steady-state window
#     ("samples=15, ss=0"), the same 512-token setting r142c used on the MTP shape. (2) three more code-c1 runs
#     to bound the run-to-run spread behind the 323.8 read (SEQS-8 reads on this image ranged 193–265 median).
set -u
R=/srv/qwen5090/results/2026-09-04-r177-matrix
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020
p1(){ local tag=$1 name=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log" || echo "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)" | tee -a "$R/audit.log"
}
log "=== R177b start (c16 at 1024 tokens; code-c1 repeat) ==="
curl -sf -m 5 $U/health >/dev/null || { log "daily not healthy"; exit 1; }
p1 daily16b code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
p1 daily16b code-c1  --conc 1 --tokens 1024 --runs 3 --kind code
log "=== R177b DONE ==="
