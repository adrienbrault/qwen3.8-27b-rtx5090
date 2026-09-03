#!/usr/bin/env bash
# R170 (2026-09-03): live validation of 0137 — LRU eviction inside vLLM's fs tier (codex NOTES20; the user asked for it after
# the eval tier hit 100% under the SWE-bench campaign, and the daily was stranded the same way on 09-01). Two arms, each on
# a wiped eval-l2 with TIER_CAP_GB=24 (evict_scope root, min_free 5 GB); a 12 x 90K flood writes far more than 24 GB, so
# eviction MUST fire on both:
#   N = the rc2 prs nvfp4 candidate (launch-daily-v0290-candidate.sh EXP=1; the 0.29 target)
#   D = the v0.28 daily image + the 0137 layer only (vllm-qwen38:v0280-nvfp4kv-0137), fp8 daily shape — daily-adoption evidence
# Per arm: (1) FLOOD: needles 131K x2 --evict 12 --evict-reasks 3 (15 s apart). Hits must be 100% — an evicted prefix simply
#     recomputes; a wrong answer, a "block I/O failed" line, or a dead engine is the failure. Then /metrics: evicted_files > 0,
#     bytes_used <= cap; df(eval-l2) <= cap + 8 GB (in-flight stores); (2) CONCURRENCY: two needle_depth processes in parallel
#     (different seeds, --evict 6 each) so lookups, loads, stores and evictions interleave — the pin logic is what keeps a file
#     from being unlinked between lookup and load (the offloading worker's `assert success` would kill the engine);
#     (3) RESTART: docker restart → the startup scan repopulates the LRU (bytes_used ≈ du) → the (1) seed again with
#     --evict-reasks 2 → hits; (4) plain 9K/131K needles as the sanity tail. PASS per arm = all hits, evicted_files > 0,
#     usage bounded, 0 error lines, engine alive at the end.
# Unit: sudo systemd-run --unit=r170-tier-evict --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r170-tier-evict bash /srv/qwen5090/r170-tier-evict.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r170-tier-evict; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
RC2_IMG=${RC2_IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}
V28_IMG=${V28_IMG:-vllm-qwen38:v0280-nvfp4kv-0137}
CAP=${CAP:-24}; MINFREE=${MINFREE:-5}
for i in "$RC2_IMG" "$V28_IMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
for i in "$RC2_IMG" "$V28_IMG"; do sudo docker run --rm --entrypoint python3 "$i" -c "import vllm.v1.kv_offload.tiering.fs.manager as m; assert hasattr(m.FileSystemTierManager,'_evict_to_limits'), '0137 missing'" >/dev/null 2>&1 || { log "ABORT: $i does not carry 0137"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R170 tier eviction validation start (lock held): N=$RC2_IMG D=$V28_IMG cap=${CAP}G min_free=${MINFREE}G ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
L2=/srv/qwen5090/eval-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R170 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 wiped (engine down; cold tier): $(df -h "$L2" | tail -1 | awk '{print $3" used of "$2}')"; }
TIER_ENV="TIER_CAP_GB=$CAP TIER_EVICT_SCOPE=root TIER_MIN_FREE_GB=$MINFREE L2MNT=$L2"
COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
boot_N(){ local kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 $TIER_ENV ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-N.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[N] BOOT OK ${kv:+(retry KV_BYTES=$kv) }$(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-N.log" | tail -1)"; return 0; fi
    log "[N] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-N.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-N.log" || break
    teardown
  done; return 1; }
boot_D(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON $TIER_ENV IMAGE="$V28_IMG" $FP8_ENV EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="$FP8_X" bash $LAUNCH > "$R/boot-D.log" 2>&1; rc=$?
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[D] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-D.log" | tail -1 | cut -c1-200)"; return 1; fi
  log "[D] BOOT OK pool=$(sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9) (v0.28 daily image + 0137, fp8 daily shape)"; return 0; }
tier_cfg(){ sudo docker logs vllm-exp 2>&1 | grep -aoE 'max_capacity_gb[^,}]*|evict_scope[^,}]*|min_free_gb[^,}]*' | sort -u | tr '\n' ' '; }
tmetrics(){ curl -s -m 5 $U/metrics | grep -a "^vllm:kv_offload_tiering_fs_" | sed 's/{[^}]*}//' | sed "s/^/[$1 tier-metrics] /" | tee -a "$R/audit.log"; }
tm_val(){ curl -s -m 5 $U/metrics | grep -a "^vllm:kv_offload_tiering_fs_$1" | awk '{s+=$2} END{printf "%d", s}'; }
usage(){ log "[$1 usage] df=$(df -B1 --output=used "$L2" | tail -1 | awk '{printf "%.1fG",$1/1e9}') du=$(sudo du -sb "$L2" 2>/dev/null | awk '{printf "%.1fG",$1/1e9}') namespaces=$(ls "$L2" | tr '\n' ' ')"; }
errlines(){ local n; n=$(sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|block I/O failed\|AssertionError\|Failed to evict\|Failed to account"); echo "[$1 engine error-lines] $n" | tee -a "$R/audit.log"; [ "$n" = 0 ]; }
nd(){ # $1 tag, $2 seed, rest = args
  local tag=$1 seed=$2; shift 2
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --seed "$seed" "$@" --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1; local rc=$?
  grep -a "^\[" "$R/needles-$tag.out" | cut -c1-260 | sed "s/^/[$tag] /" >> "$R/audit.log"
  log "[$tag] hits=$(grep -ac '^\[HIT ' "$R/needles-$tag.out") miss=$(grep -ac MISS "$R/needles-$tag.out") rc=$rc $(grep -a SUMMARY "$R/needles-$tag.out" | cut -c1-200)"; return $rc; }
arm(){ local A=$1 fail=0 ev bu
  log "[$A] tier config seen by the engine: $(tier_cfg)"
  usage "$A start"; tmetrics "$A start"
  # (1) flood
  nd "$A-flood" r170 --depths 131000 --samples 2 --evict 12 --evict-ctx 90000 --evict-reasks 3 --evict-reask-gap 15 || fail=1
  tmetrics "$A after-flood"; usage "$A after-flood"; errlines "$A after-flood" || fail=1
  ev=$(tm_val evicted_files); bu=$(tm_val bytes_used)
  [ "${ev:-0}" -gt 0 ] || { log "[$A] FAIL: evicted_files=$ev (no eviction fired under a flood that exceeds the cap)"; fail=1; }
  [ "${bu:-0}" -le $((CAP*1000000000)) ] || { log "[$A] FAIL: bytes_used $bu > cap"; fail=1; }
  dfb=$(df -B1 --output=used "$L2" | tail -1); [ "$dfb" -le $(( (CAP+8)*1000000000 )) ] || { log "[$A] FAIL: df used $dfb > cap+8G — accounting and disk disagree"; fail=1; }
  # (2) concurrency: two floods + re-asks interleaved
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --seed r170a --depths 131000 --samples 1 --evict 6 --evict-ctx 90000 --evict-reasks 2 --evict-reask-gap 5 --out "$R/needles-$A-conc-a.jsonl" > "$R/needles-$A-conc-a.out" 2>&1 & pa=$!
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --seed r170b --depths 131000 --samples 1 --evict 6 --evict-ctx 90000 --evict-reasks 2 --evict-reask-gap 5 --out "$R/needles-$A-conc-b.jsonl" > "$R/needles-$A-conc-b.out" 2>&1 & pb=$!
  wait $pa; ra=$?; wait $pb; rb=$?
  for x in a b; do grep -a "^\[" "$R/needles-$A-conc-$x.out" | cut -c1-260 | sed "s/^/[$A-conc-$x] /" >> "$R/audit.log"; done
  log "[$A-conc] a: hits=$(grep -ac '^\[HIT ' "$R/needles-$A-conc-a.out") miss=$(grep -ac MISS "$R/needles-$A-conc-a.out") rc=$ra | b: hits=$(grep -ac '^\[HIT ' "$R/needles-$A-conc-b.out") miss=$(grep -ac MISS "$R/needles-$A-conc-b.out") rc=$rb"
  [ $ra = 0 ] && [ $rb = 0 ] || fail=1
  curl -sf -m 5 $U/health >/dev/null || { log "[$A] FAIL: engine dead after the concurrent floods"; fail=1; }
  tmetrics "$A after-conc"; usage "$A after-conc"; errlines "$A after-conc" || fail=1
  # (3) restart revisit: startup scan must repopulate the accounting
  sudo docker logs vllm-exp > "$R/engine-$A-prerestart.log" 2>&1
  sudo docker restart vllm-exp >/dev/null 2>&1; ok=0
  for i in $(seq 120); do sleep 5; curl -sf -m 3 $U/health >/dev/null && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    log "[$A] restarted; engine up after $((i*5)) s"
    tmetrics "$A after-restart"; usage "$A after-restart"
    nd "$A-restart" r170 --depths 131000 --samples 2 --evict 12 --evict-ctx 90000 --evict-reasks 2 --evict-reask-gap 15 || fail=1
    tmetrics "$A after-restart-flood"; usage "$A after-restart-flood"; errlines "$A after-restart" || fail=1
    nd "$A-tail" r170t --depths 9000 131000 --samples 2 --warm || fail=1
  else log "[$A] FAIL: engine did not come back after docker restart"; fail=1; fi
  curl -sf -m 5 $U/health >/dev/null || { log "[$A] FAIL: engine dead at the end"; fail=1; }
  sudo docker logs vllm-exp 2>&1 | grep -ai "evict\|tier\|ENOSPC\|No space" | grep -av "^$" | cut -c1-200 | tail -20 | sed "s/^/[$A tier-log] /" >> "$R/audit.log"
  log "[$A] VERDICT: $([ $fail = 0 ] && echo PASS || echo FAIL)"; echo "$A $([ $fail = 0 ] && echo PASS || echo FAIL)" >> "$R/verdicts.txt"; }
teardown
wipe_l2; if boot_N; then arm N; else echo "N BOOT-FAILED" >> "$R/verdicts.txt"; fi; teardown
wipe_l2; if boot_D; then arm D; else echo "D BOOT-FAILED" >> "$R/verdicts.txt"; fi; teardown
log "verdicts: $(tr '\n' ' ' < "$R/verdicts.txt")"
finish DONE
