#!/usr/bin/env bash
# R169 (2026-09-03, "focus on .29"): the v0.29.0rc2 audition — the promotion sheet inputs for BOTH 0.29 routes, plus the
# tier half of G12 on the target chain (replaces the cancelled r166d):
#   N = nvfp4 candidate via launch-daily-v0290-candidate.sh (EXP=1, SEQS 8): validates the PROVISIONAL pinned pool
#       (retries with KV_BYTES −0.5 / −1.0 GB if the 384 MiB free guard trips, and says so), needles 9K/131K, fidelity vs
#       the FP8 reference (+ direct vs the fp8 arm), benchy c1/c8, decode_ss code c1/c8, deep30k (R167: 29 tok/s on rc1
#       — r168 isolates it; this arm records where rc2 stands), then the TIER CHECK: needles 131K x2 --evict 12 (flood
#       12 x 90K > pool) --evict-reasks 3 (15 s apart: the fs lookup promotes asynchronously) → docker restart → same seed again (restart revisit). Pass = every hit AND
#       tier_served > 0 on at least one pass (rc1 served a fresh-boot first touch in R167; v0.28 never did, R166c).
#   F = fp8 daily shape on the rc2 prs image + embed offload (0134 + 0135): the shortest path to ANY 0.29 daily
#       (R165b: rc1 fp8 at parity; R167: +8.4% pool). Same measurements incl. the tier check; fidelity vs the v0.28 fp8 run.
#   D = v0.28 fp8 daily shape (the current daily's image/flags on :8029), same hour, for the paired sheet (SKIP_D=1 skips).
# eval-l2 is wiped before each arm (cold tier; the campaign filled it to 100% — R168 note). Queue-registered; restores the
# daily at the end unless another unit is queued.
# Unit: sudo systemd-run --unit=r169-rc2 --collect -p User=adrienbrault -p RuntimeMaxSec=14400 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r169-rc2 bash /srv/qwen5090/r169-rc2-audition.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-03-r169-rc2; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
RC2_IMG=${RC2_IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs}
V28_IMG=vllm-qwen38:v0280-nvfp4kv
sudo docker image inspect "$RC2_IMG" >/dev/null 2>&1 || { log "ABORT: image $RC2_IMG missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R169 rc2 audition start (lock held): $RC2_IMG; SKIP_D=${SKIP_D:-0} ==="
U=http://127.0.0.1:8029
LAUNCH=/srv/qwen5090/launch-daily-v0280.sh
CAND=/srv/qwen5090/launch-daily-v0290-candidate.sh
MODEL=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
EB='--extra-body {"temperature":0.6}'
L2=/srv/qwen5090/eval-l2
FD=/srv/qwen5090/results/2026-08-23-fidelity
FP8_RUN=/srv/qwen5090/results/2026-09-03-r165c-audition/run-v0280-redhat-fp8kv.jsonl
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R169 rc2 audition $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; log "eval-l2 wiped (engine down; cold tier): $(df -h "$L2" | tail -1 | awk '{print $3" used of "$2}')"; }

COMMON="MODEL_DIR=$MODEL TP=2 PORT=8029 NAME=vllm-exp BIND_ADDR=127.0.0.1 MAXLEN=262144 NO_TIER=0 L2MNT=$L2 PREFIX_CACHE=1 POOL_MIN=1 POOL_MAX=9999999 MNBT=8192 SEQS=8"
FP8_ENV="KVD_OVERRIDE=fp8_e4m3 FIWS=268435456 UTIL=0.92"
FP8_SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":9}'
FP8_X="-e NCCL_P2P_LEVEL=SYS"
EMBED_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
bootfacts(){ sudo docker logs vllm-exp 2>&1 | grep -aE "Offloader set to|CPU offloaded parameters|matched no parameters|Available KV cache memory|GPU KV cache size|Graph capturing finished|Capturing dflash|running the draft eagerly|decode_backend=|kv_cache_memory_bytes" | cut -c1-220 > "$R/bootfacts-$1.txt"; }
boot_N(){ local kv rc
  for kv in "" 14000000000 13500000000; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=8 ${kv:+KV_BYTES=$kv} bash $CAND > "$R/boot-N.log" 2>&1; rc=$?
    bootfacts N
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then log "[N] BOOT OK ${kv:+(retry KV_BYTES=$kv) }$(grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB' "$R/boot-N.log" | tail -1) $(grep -a 'CPU offloaded parameters' "$R/bootfacts-N.txt" | tail -1 | sed 's/.*INFO[^]]*] //' | cut -c1-60)"; return 0; fi
    log "[N] boot attempt ${kv:-default pin} FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-N.log" | tail -1 | cut -c1-200)"
    grep -aq "Bug C headroom missing" "$R/boot-N.log" || break   # only the free-VRAM guard is worth a lower pin
    teardown
  done; return 1; }
boot_F(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$RC2_IMG" $FP8_ENV EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="$FP8_X" EXTRA_ARGS="$EMBED_ARGS" bash $LAUNCH > "$R/boot-F.log" 2>&1; rc=$?
  bootfacts F
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[F] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-F.log" | tail -1 | cut -c1-200)"; return 1; fi
  local bad=0 BL; BL=$(sudo docker logs vllm-exp 2>&1)
  [ "$(echo "$BL" | grep -ac 'Offloader set to UVAOffloader')" -ge 1 ] || { log "[F] ASSERT FAILED: UVAOffloader not selected"; bad=1; }
  [ "$(echo "$BL" | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || { log "[F] ASSERT FAILED: embed shard not offloaded"; bad=1; }
  [ "$(echo "$BL" | grep -ac 'matched no parameters')" -eq 0 ] || { log "[F] ASSERT FAILED: selector matched no parameters"; bad=1; }
  [ $bad = 0 ] || { log "[F] offload asserts failed — arm NOT measured"; return 1; }
  log "[F] BOOT OK pool=$(echo "$BL" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9) min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB vllm=$(sudo docker exec vllm-exp python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"; return 0; }
boot_D(){ local rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" $COMMON IMAGE="$V28_IMG" $FP8_ENV EXTRA_MOUNT="-v $DRAFT:/draft:ro" SPEC_JSON="$FP8_SPEC" EXTRA_ENV="$FP8_X" bash $LAUNCH > "$R/boot-D.log" 2>&1; rc=$?
  bootfacts D
  if [ $rc -ne 0 ] || ! curl -sf -m 5 $U/health >/dev/null; then log "[D] BOOT FAILED rc=$rc: $(grep -aE 'FAILED|Error' "$R/boot-D.log" | tail -1 | cut -c1-200)"; return 1; fi
  log "[D] BOOT OK pool=$(sudo docker logs vllm-exp 2>&1 | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'size: [0-9,]+' | tr -dc 0-9) (v0.28 fp8 daily shape)"; return 0; }
needles(){ # $1 tag, $2 depths, rest = extra args
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths $2 --samples 2 --warm "${@:3}" --out "$R/needles-$1.jsonl" > "$R/needles-$1.out" 2>&1
  log "[$1 needles $2] $(grep -a SUMMARY "$R/needles-$1.out" | cut -c1-120)"; }
nd_tier(){ # $1 tag — fixed seed → identical prompts across invocations (restart revisit)
  local tag=$1; shift
  python3 /srv/qwen5090/probes/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths 131000 --samples 2 --seed r169 "$@" --out "$R/needles-$tag.jsonl" > "$R/needles-$tag.out" 2>&1; local rc=$?
  grep -a "^\[" "$R/needles-$tag.out" | cut -c1-300 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  log "[$tag] $(grep -a SUMMARY "$R/needles-$tag.out" | cut -c1-220) rc=$rc"; }
served(){ python3 -c "import json,sys; s=json.loads(open('$1').read().split('SUMMARY',1)[1].strip().splitlines()[0]) if 'SUMMARY' in open('$1').read() else {}; print(int(bool(s.get('tier_served'))), int(s.get('tier_served_hits',0)), int(s.get('total',0)), int(s.get('cold_hits',0)), int(s.get('evict_hits',0) or 0))" 2>/dev/null || echo "0 0 0 0 0"; }
tier_check(){ # $1 arm: in-session eviction + restart revisit on this engine
  local A=$1 a b
  nd_tier "$A-evict" --evict 12 --evict-ctx 90000 --evict-reasks 3 --evict-reask-gap 15
  log "[$A] docker restart"; sudo docker restart vllm-exp >/dev/null 2>&1; sleep 20
  for i in $(seq 90); do curl -sf -m 5 $U/health >/dev/null && break; sleep 10; done
  curl -sf -m 5 $U/health >/dev/null || { log "[$A] TIER FAILED: engine did not come back after docker restart"; return 1; }
  nd_tier "$A-restart" --evict-reasks 3 --evict-reask-gap 15
  a=$(served "$R/needles-$A-evict.out"); b=$(served "$R/needles-$A-restart.out")
  log "[$A] TIER SHEET evict(served,served_hits,total,cold_hits,evict_hits)=$a restart=$b"
  set -- $a; local s1=$1 h1=$2; set -- $b; local s2=$1 h2=$2 t2=$3 c2=$4
  if { [ "$s1" = 1 ] || [ "$s2" = 1 ]; } && grep -aq "MISS" "$R/needles-$A-evict.out" "$R/needles-$A-restart.out"; then log "[$A] TIER FAIL: tier-served pass with a MISS"
  elif [ "$s1" = 1 ] || [ "$s2" = 1 ]; then log "[$A] TIER PASS: tier served blocks and every pass hit (in-session served=$s1, restart served=$s2)"
  else log "[$A] TIER INCONCLUSIVE: all hits but no pass was tier-served (recomputed every time)"; fi; }
benchy(){ timeout 3000 llama-benchy --base-url $U/v1 --model qwen3.8-27b --tokenizer "$MODEL" --pp 2048 --tg 256 --concurrency $2 --runs $3 --no-cache --latency-mode api --format json $EB --save-result "$R/benchy-$1-c$2.json" > "$R/benchy-$1-c$2.out" 2>&1
  python3 /srv/qwen5090/probes/benchy_runs.py "$R/benchy-$1-c$2.json" 2>&1 | sed "s/^/[$1 benchy c$2] /" | cut -c1-200 | tee -a "$R/audit.log"; }
dss(){ local tag=$1 kind=$2 conc=$3; shift 3
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc $conc --tokens 512 --runs 3 --kind $kind "$@" --out "$R/decode-$tag-$kind-c$conc.json" > "$R/decode-$tag-$kind-c$conc.out" 2>&1
  grep -a RESULT "$R/decode-$tag-$kind-c$conc.out" | sed "s/^/[$tag decode_ss $kind c$conc] /" | cut -c1-230 | tee -a "$R/audit.log"; }
deep30k(){ python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000 --out "$R/decode-$1-deep30k.json" > "$R/decode-$1-deep30k.out" 2>&1
  grep -a RESULT "$R/decode-$1-deep30k.out" | sed "s/^/[$1 decode_ss deep30k] /" | cut -c1-230 | tee -a "$R/audit.log"; }
fidelity(){ python3 /srv/qwen5090/probes/fidelity.py run --url $U --model qwen3.8-27b --corpus "$FD/corpus.jsonl" --out "$R/run-$1.jsonl" --conc 1 > "$R/fidelity-run-$1.log" 2>&1 || log "[$1] ruler FAILED rc=$?"
  python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FD/run-fp8ref.jsonl" "$R/run-$1.jsonl" 2>&1 | tee "$R/fidelity-$1-vs-fp8ref.txt" | cut -c1-200 | sed "s/^/[$1 fidelity vs fp8ref] /" | tee -a "$R/audit.log"; }
metrics_line(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:(num_preemptions_total|spec_decode_num_(accepted|draft)_tokens_total)" | sed "s/^/[$1] /" | cut -c1-160 | tee -a "$R/audit.log"; }
errlines(){ sudo docker logs vllm-exp 2>&1 | grep -ac "illegal memory\|CUDA error\|Traceback\|OutOfMemoryError" | sed "s/^/[$1 engine error-lines] /" | tee -a "$R/audit.log"; }
measure(){ local T=$1
  needles $T "9000 131000"
  fidelity $T
  benchy $T 1 3; benchy $T 8 2
  dss $T code 1; dss $T code 8; dss $T prose 8
  deep30k $T
  metrics_line $T; errlines $T
  tier_check $T; errlines "$T post-tier"; }

log "taking the daily down"; teardown
wipe_l2; if boot_N; then measure N; fi; teardown
wipe_l2; if boot_F; then measure F
  [ -f "$FP8_RUN" ] && python3 /srv/qwen5090/probes/fidelity.py compare --ref "$FP8_RUN" "$R/run-F.jsonl" 2>&1 | tee "$R/fidelity-F-vs-v0280fp8.txt" | cut -c1-200 | sed "s/^/[F fidelity vs v0.28 fp8 daily shape] /" | tee -a "$R/audit.log"
  [ -f "$R/run-N.jsonl" ] && python3 /srv/qwen5090/probes/fidelity.py compare --ref "$R/run-F.jsonl" "$R/run-N.jsonl" 2>&1 | tee "$R/fidelity-N-vs-F.txt" | cut -c1-200 | sed "s/^/[N fidelity vs F (direct, same hour)] /" | tee -a "$R/audit.log"
fi; teardown
if [ "${SKIP_D:-0}" != 1 ]; then wipe_l2; if boot_D; then measure D; fi; teardown; fi
log "SHEET:"; grep -aE "BOOT OK|benchy c[18]\]|decode_ss|fidelity|needles|TIER (PASS|FAIL|INCONCLUSIVE|SHEET)|error-lines" "$R/audit.log" | grep -av "^.*SHEET:$" | cut -c1-200 > "$R/sheet.txt"; wc -l "$R/sheet.txt" | tee -a "$R/audit.log"
finish DONE
