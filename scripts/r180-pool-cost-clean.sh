#!/usr/bin/env bash
# R180 (2026-09-04): R179 re-run with the eval-l2 tier WIPED before every arm (R179's MB4 arm served its 30K/100K prompts from the
# tier at 83% external hit, so its usage numbers are not a prefill measurement), plus the fidelity gate for the SSM-bf16 arm.
# Arms (daily image, :8029, SEQS 16): BASE (daily flags, pin 13.98 GB) | MB4 (--mamba-block-size 11776) | BF16 (--mamba-ssm-cache-dtype
# bfloat16, pin 13.5 GB so the pool stays inside the EXP band: R179 read 1,020,596 at 13.98 GB, +12.9%).
# Per arm: pool/layout; kv_capacity_probe short x1, short x4, 30K, 100K, 100K revisit, 5 x 100K ignore_eos; decode_ss code c8 x2, prose c1 x2.
# BASE + BF16: bf16 rulers (fidelity_ladder dense 724,781 positions ~3 min; agentic_ref 57,972 positions ~1 min) vs the R156 bf16 dumps,
# and BF16: decode_fidelity greedy 20 x 256 at ctx0/ctx30000 vs the r173c bf16 DECODE reference (NS9A on the S image reads 0.0052 at 30K).
# Unit: sudo systemd-run --unit=r180-pool-cost-clean --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r180-pool-cost-clean bash /srv/qwen5090/r180-pool-cost-clean.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r180-pool-cost-clean; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] || { log "ABORT: references missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R180 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R180 pool-cost clean re-run start (lock held): arms BASE/MB4/BF16, tier wiped per arm ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 wiped"; }
boot_arm(){ local tag=$1 xargs=$2 kv rc; shift 2
  for kv in "$@"; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv EXTRA_ARGS_APPEND="$xargs" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv args='$xargs' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB"
      log "[$tag] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|mamba_block_size="[0-9]+"|mamba_ssm_cache_dtype="[a-z0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ')"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"; teardown
  done; return 1; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
rulers(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  log "[$T bf16 agentic] $(grep -aE "^DONE|FAIL" "$R/score-$T-agentic.out" | tail -1 | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
arm(){ local tag=$1 xargs=$2 fid=$3; shift 3
  wipe_l2
  boot_arm $tag "$xargs" "$@" || { log "[$tag] BOOT FAILED on every pin"; return 1; }
  sleep 20
  cap $tag short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 21
  cap $tag short4 --ctx 0 --conc 4 --tokens 400 --ignore-eos --seed 22
  cap $tag ctx30k --ctx 36000 --conc 1 --tokens 200 --ignore-eos --seed 23
  cap $tag ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 24
  cap $tag revisit100k --ctx 120000 --conc 1 --tokens 32 --seed 24
  cap $tag five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 25
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  if [ "$fid" = 1 ]; then rulers $tag; dfid $tag 0; dfid $tag 30000; fi
  log "[$tag engine error-lines] $(sudo docker logs vllm-exp 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
  teardown; }
teardown
arm BASE "" 1 13980000000 13500000000
arm MB4  "--mamba-block-size 11776" 0 13980000000 13500000000
arm BF16 "--mamba-ssm-cache-dtype bfloat16" 1 13500000000 13000000000
grep -aE "BOOT OK|block_size|cap |RESULT|FAILED|error-lines|vs bf16|decode ctx" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
