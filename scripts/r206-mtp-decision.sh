#!/usr/bin/env bash
# R206 (2026-09-06, after R205d PASS): the MTP decision run — DFlash ns7 daily route vs MTP ns3 on the 0158 (eagle-shift) image,
# PAIRED in one unit, same hour, same probes, so the sheet can price the MTP route against the served daily:
#   DF arm  = served daily image, default route (DFlash ns7, SEQS 16, PCIE_IPC=1, BSS=1, pin 13.98):
#             layout, warm revisit 32K (the per-revisit re-prefill tax: DFlash re-prefills ~685 of 53,453 tokens on a revisit, R204),
#             decode_ss code-c1 runs 3 / prose-c1 runs 2 / code-c8 runs 2 / code-c16 / prose-c1 at 30K filler, tool-eval-bench 69x4.
#   M3 arm  = <daily>-mtppcie-mtpcache-eagleshift, SPEC_METHOD=mtp SPEC_NS=3, VLLM_SM12X_PCIE_IPC_MTP=1 (the R205d route):
#             the same rows (revisit tax on MTP was 1,936 of 53,456 in R205d), PLUS the needle gate with a 16x90K flood (1.44M tokens):
#             R205d's 12x90K flood (1.08M) cannot evict a needle from the 1.31M pool when the flood is sequential and the pool is LRU
#             (needle + flood = 1.21-1.30M < 1.31M), so tier_served stayed 0 and the tier path was only proven on the 32K row.
#   M5 arm  = the same image at SPEC_NS=5 (the 1-layer MTP head is applied recursively): boot + layout (pool per draft position, R200b
#             priced one GDN state copy per position per request) + decode_ss code-c8 / code-c16 / prose-c1. Decode only: no rulers.
# Fidelity: NOT re-run here — 0158 changes prefix hashing only (R205d warm_equal 3/3 with hits), and the MTP ns3 rulers exist from R205.
# Boot-pin tally: R205, R205c and R205d each booted this chain at 13.98 first try = 3 of 3, recorded, not re-run.
# Verdicts go on the sheet; promotion is the user's call.
#   unit: sudo systemd-run --unit=r206-mtp-decision --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r206-mtp-decision bash /srv/qwen5090/r206-mtp-decision.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r206-mtp-decision; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); MTP=$DAILY-mtppcie-mtpcache-eagleshift
PIN=13980000000; PIN_FALLBACK=13500000000
PROOF='SM12X eagle-drop replay boundary retained'
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
for i in "$DAILY" "$MTP"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
sudo docker run --rm --entrypoint cat "$MTP" /opt/prs-markers/0158 2>/dev/null | grep -q 0158 || { log "ABORT: marker 0158 missing in $MTP"; exit 3; }
grep -q "R197 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R197 SPEC_METHOD passthrough"; exit 3; }
for t in warm-revisit.py needle_gate.sh needle_depth.py decode_ss.py tooleval_summary.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
command -v tool-eval-bench >/dev/null || { log "ABORT: tool-eval-bench not on PATH"; exit 3; }

. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R206 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R206 start (lock held): DF=$DAILY (dflash ns7) vs M3/M5=$MTP (mtp ns3 / ns5), EXP=1 SEQS=16 PCIE_IPC=1 BSS=1 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_once TAG IMG PIN [SPEC_METHOD SPEC_NS] → 0 = up
boot_once(){ local tag=$1 img=$2 kv=$3 method=${4:-dflash} ns=${5:-7} rc extra="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1"
  [ "$method" = mtp ] && extra="$extra -e VLLM_SM12X_PCIE_IPC_MTP=1"
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$img SPEC_METHOD=$method SPEC_NS=$ns EXTRA_ENV_APPEND="$extra" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    log "[$tag] BOOT OK pin=$kv spec=$method/ns$ns pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) proof_0158=$(grep -ac "$PROOF" "$R/engine-boot-$tag.log") min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB aot_loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log") compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception|headroom" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
boot_arm(){ local tag=$1; shift 1; boot_once "$tag" "$@" || boot_once "${tag}b" "$@" ; }   # positional: IMG PIN [method ns]; retry once at the same pin
layout(){ log "[$1 layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
revisit(){ local tag=$1 name=$2; shift 2
  python3 $PR/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 "$@" > "$R/revisit-$tag-$name.log" 2>&1
  log "[$tag revisit $name] $(grep -a RESULT "$R/revisit-$tag-$name.log" | cut -c1-300)"
  log "[$tag revisit-tax] $(python3 -c "
import json,re; s=open('$R/revisit-$tag-$name.log').read(); m=re.search(r'RESULT (\{.*\})', s); r=json.loads(m.group(1))['send2']
print('re-prefilled', int(r['queries_delta']-r['hits_delta']), 'of', int(r['queries_delta']), 'tokens on the warm revisit, ttft', r['ttft_s'], 's')" 2>&1 | cut -c1-200)"; }
tooleval(){ local tag=$1
  log "[$tag tool-eval] start 69x4 (temp 0.6 top-p 0.95 top-k 20 parallel 8)"
  ( cd "$HOME" && timeout 5400 tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$tag.json" > "$R/tooleval-$tag.log" 2>&1 ); local rc=$?
  curl -sf -m 5 $U/health >/dev/null || { log "[$tag tool-eval] ENGINE DIED (rc=$rc)"; return 1; }
  python3 $PR/tooleval_summary.py "$R/tooleval-$tag.json" "$tag" 2>&1 | cut -c1-300 | sed "s/^/[$tag tool-eval] /" | tee -a "$R/audit.log"; }
decode_rows(){ local tag=$1
  p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
  p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $tag code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  p1 $tag prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000; }
arm_report(){ local tag=$1
  log "[$tag] engine error lines: $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')  prefix: $(curl -s -m 5 $U/metrics | awk '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} END {printf "queries=%d hits=%d", q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"]}')"
  ELOG > "$R/engine-$tag-final.log"; }

### DF arm — the served daily route
teardown; wipe_l2
if boot_arm DF "$DAILY" $PIN dflash 7; then
  sleep 20; layout DF
  revisit DF gpu-warm
  decode_rows DF
  tooleval DF
  arm_report DF
else log "[DF] ARM FAILED: could not boot the daily image at the pin (twice)"; fi

### M3 arm — MTP ns3 on the 0158 image
teardown; wipe_l2
if boot_arm M3 "$MTP" $PIN mtp 3; then
  sleep 20; layout M3
  grep -aq "$PROOF" "$R/engine-boot-M3.log" "$R/engine-boot-M3b.log" 2>/dev/null && log "[M3] 0158 proof line present" || log "[M3] WARNING: 0158 proof line MISSING (retention/eagle-drop route changed?)"
  revisit M3 gpu-warm
  decode_rows M3
  log "[M3 needle gate] start: 131K + 220K cold, 16x90K flood (1.44M > 1.31M pool), two re-asks each through the tier"
  U=$U EVICT=16 bash $PR/needle_gate.sh m3 "$R" > "$R/needle-gate-M3.log" 2>&1; rc=$?
  log "[M3 needle gate] rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate-M3.log" | tail -1 | cut -c1-330)"
  grep -aE "^\[needle-gate" "$R/needle-gate-M3.log" | grep -aE "depth=" | cut -c1-200 | sed "s/^/[M3 needle] /" | tee -a "$R/audit.log"
  tooleval M3
  arm_report M3
else log "[M3] ARM FAILED: could not boot $MTP at the pin (twice)"; fi

### M5 arm — MTP ns5, decode + pool only
teardown; wipe_l2
if boot_arm M5 "$MTP" $PIN mtp 5; then
  sleep 20; layout M5
  p1 M5 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 M5 code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  p1 M5 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  arm_report M5
else log "[M5] ARM FAILED: could not boot $MTP at ns5 (twice) — ns5 row empty"; fi

{ echo "R206 MTP decision run — DF=$DAILY (dflash ns7) vs M3/M5=$MTP (mtp ns3/ns5); boot-pin tally for the MTP chain: R205/R205c/R205d 3 of 3 at 13.98 first try (not re-run)"
  grep -aE "BOOT OK|FAILED|layout|revisit|RESULT|PROBE FAILED|tool-eval|needle|error lines|proof|ENGINE DIED|ARM FAILED" "$R/audit.log" | cut -c1-330; } > "$R/sheet.txt"
finish DONE
