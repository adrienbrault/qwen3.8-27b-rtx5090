#!/usr/bin/env bash
# r27-l1a-seq64-curves (2026-10-17): R794's solid decode_ss + dashed ShareGPT V3 / Spec-Bench lines re-measured on the
#   L1a daily for the 27B public README's seq-64 decode figure (docs/img/decode-scaling-64.svg). Measurement protocol,
#   instruments, shapes, phases, A/B passes and seed scheme are unchanged from R794 (2026-09-28). Both lines use one boot
#   configuration: the served L1a launcher with the sequence limit raised to 64.
# LINEAGE: r793-27b-std-bench.sh (protocol, integrity, traffic, restore; Env-sorted config hash 642ba83) for phase 2;
#   r675-27b-curves.sh (decode_ss curve: 1,024 forced tokens, 3 runs per shape, code and prose) for phase 1;
#   r206c-mtp-c32c64.sh for the SEQS 64 boot (capture cap 320, eval-l2 wipe, max Running / Waiting read-outs).
#   r27-l1a-curves.sh for GLM stop/queue-aware restore and shm sweep after 27B teardown.
# WHAT IS MEASURED: /srv/qwen5090/launch-daily.sh (md5 6347110bbd23a1ecc00a414b9d627fce, asserted before the lock, under it and at every
#   boot attempt) with EXP=1 SEQS=64 -> :8029, container vllm-exp, 127.0.0.1 bind, tier on /srv/qwen5090/eval-l2. The EXP path
#   differs from the daily path in knobs that default OFF, so the unit sets each one to the daily's value:
#     KV_BYTES=14860000000  the served pin. The launcher's pinned-budget table has no SEQS 64 case (8/16/32 only) and its md5 is
#                           gated, so the budget is passed, not added to the table. The pool follows the pin, not SEQS (R206c:
#                           1,309,368 at SEQS 16 and 64 on 13.98), so 1,391,795 is expected (R675 / R793 at SEQS 16).
#     PCIE_IPC=1 BSS=1      forced on the daily port, default OFF on EXP (launch-daily.sh lines 202 / 212).
#     CC_EXTRA max_cudagraph_capture_size 320: the pcie_ipc slab holds 320 rows and vLLM sizes the capture list at
#                           min(SEQS x 4 x 2, 512) = 512 at SEQS 64; 0148 then refuses the MTP drafter (R206c v1 died there).
#                           Decode batches stay inside it (64 x 4 = 256). launch-daily.sh line 355 names this cap.
#     POOL_MIN/POOL_MAX 1360000/1425000 and MIN_FREE_MIB 512: the daily's band and floor. The EXP default ceiling
#                           (1,400,000) sits only 8K above the expected pool; POOL_*/MIN_FREE_MIB are read before the
#                           launcher's unset block (lines 145-148), so they reach it (read at line 121).
#   Everything else is the EXP default = the daily: DAILY_IMG, nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV, MTP ns3, bf16 SSM,
#   MNBT 8192, embed offload, split_kv 1, 16 GiB CPU tier, temperature 0.6 default, reasoning effort medium. L1a's tag selects
#   V2 runner, triton mamba, FlashInfer GDN prefill and draft UB plan=1 on EXP too. What stays
#   different from the served :8020 engine by construction: max-num-seqs 64, the capture cap, and the disk tier (eval-l2,
#   no cap / scope / min-free), plus stock power (the daily caps to 400 W after boot; every published 27B number is at stock).
# PHASE 1 (one boot): probes/decode_ss.py, code then prose, c = 1 2 4 8 16 24 32 48 64, 3 runs, 1,024 forced tokens,
#   --seed-prefix r794-<nonce>- (unchanged for comparability; fresh prompts per invocation). One invocation per (kind, c), all appending to
#   decode-<kind>.jsonl: the seeds (f"{prefix}{c}-{i}") and the file layout are those of one R675-style sweep, and the unit
#   snapshots /metrics around every shape (preemptions, MTP accepted / drafted per shape; decode_ss records its own
#   accept_per_draft from the same counters, as R675). decode_ss's steady-state window only counts samples with
#   num_requests_running == c, so a RESULT at c64 is itself the proof that 64 requests ran together. R206c's own points were
#   code c16 / c32 / c64 and prose c32 / c64 (plus a code c32 re-run after c64); these nine are a superset.
# PHASE 2 (R793's protocol unchanged, [R794] marks the changes): vLLM v0.30.0 client + probes/vllm_bench_tabby.py, ShareGPT V3
#   SG_N 400 --seed 7310 (the R793 sample, kept for comparability: at c64 that is 6.25 waves, logged per cell), Spec-Bench
#   all 480 x 256 tokens (7.5 waves at c64), openai-chat, stream, --temperature 0, template-default thinking, closed loop at
#   [R794] c = 1 4 8 16 32 48 64, passes A and B, ShareGPT then Spec-Bench within each pass, a FRESH boot per cell (28 boots),
#   `conc` 64-token out-of-set warm-ups with the metrics self-test (VOID on the first cell), /metrics deltas per cell.
#   [R794] eval-l2's _model_* namespaces are wiped before EVERY boot (as R206c): no cell can hit the tier from an earlier cell
#   or from phase 1. R793 could not do that on the production native-l2 (it wrote 57.6 MB in one cell, 0 hits anywhere).
#   [R794] preemptions per cell from the same deltas: > 0 demotes the cell (a preempted request is re-prefilled, so the cell
#   no longer measures what the pool admits). Max Running / Waiting / GPU KV cache usage from vLLM's 10 s log lines, per cell.
# BOOT IDENTITY (every boot, phase 1's is the reference): launcher md5, `0.29 nvfp4 EXP UP` (literal deliberately retained
#   by the L1a launcher; version field is v0.31), image name == DAILY_IMG and image
#   id, sha256 of the container Args (in order) + Env (SORTED; docker's Env order is not stable, R793 -1552 VOID), pool,
#   SEQS 64, Args carry --max-num-seqs 64 / the 14.86 GB pin / the 320 capture cap, stock power readback, core 0 / memory
#   +4500 readback (the launcher writes them at every boot), AOT compile hash set (torch_aot_compile/<12 hex>; the capture
#   config is a compile-key input, so boot 1 probably compiles and every later boot loads the same artifact). Any drift: VOID.
# DECISION (pre-registered): the summary's run-level R731 rule is logged unchanged as RUN DECISION. [R794] The same rule is
#   then applied PER CELL (dataset x concurrency): both passes complete; no integrity / cache flag naming either pass's tag;
#   A/B spread on output tok/s <= 3 % except c1 (exempt, published with its spread); foreign 0 and preemptions 0 in both
#   passes. cells-verdict.tsv holds one row per cell. Phase 1 is OK when every (kind, c) shape has 3 good runs, 0 preemptions
#   and the engine logged no error. Last audit line: `DECISION: VOID <why>` | `PUBLISHABLE ...` (phase 1 OK and 14/14 cells)
#   | `NOT-PUBLISHABLE <n>/14 cells publishable; <which fail and why>; phase 1 <state>`.
# TRAFFIC: Olla drained for the unit's lifetime; hermes, hermes-webui, owui-proxy stopped (v0280 restarts owui-proxy at the
#   end of every boot, so it is stopped again after each). :8029 binds 127.0.0.1, so no container can reach it anyway.
# RESTORE (end, signal, or a signal while queued): archive glm53's logs and remove glm53 / glm53-preflight before EXP
#   claims the SAME :8029 port. After every vllm-exp teardown run shm-sweep.sh quiet, then power stock and GLM restore via
#   r858/restart-glm-daily.sh (skipped if another unit is queued). Restart only previously running direct clients with their
#   configs untouched. A queued signal restores only with the lock free and :8029 down; no Hermes repoint.
# GPU BUDGET ~3 h (historical r794-IMPL-NOTES.md estimate); allow up to 23 min for GLM restore plus unmeasured L1a compile.
#   RuntimeMaxSec 43200.
# RUN (on flan after installing this file; .new + mv, never over a running file):
#   sudo systemd-run --unit=r27-l1a-seq64-curves --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r27-l1a-seq64-curves.sh
#   Stop: sudo systemctl stop r27-l1a-seq64-curves
set -uo pipefail
export HOME=${HOME:-$HOME}
export PATH="$HOME/.local/bin:$PATH"   # launch-daily-v0280.sh pre-warms with llama-benchy (~/.local/bin)
UNIT=${UNIT:-r27-l1a-seq64-curves}   # also the `unit=` metadata of every result JSON, which bench/plot.py checks
D=/srv/qwen5090
R=${R:-$D/results/$(date -u +%F)-$UNIT-$(date -u +%H%M)}
[ -e "$R/audit.log" ] && { echo "ABORT: $R already holds a run"; exit 3; }
mkdir -p "$R/results" "$R/probes" "$R/cells" "$R/server" "$R/phase1"
cp "$0" "$R/" 2>/dev/null
C=$R/cells
LAUNCH27=$D/launch-daily.sh
LAUNCH27_MD5=${LAUNCH27_MD5:-6347110bbd23a1ecc00a414b9d627fce}
WANT_IMG=${WANT_IMG:-vllm-qwen38:v0310-r27c-A_S-L1a-ubplan}
LIVE=$D/r858/restart-glm-daily.sh
GLM_U=http://127.0.0.1:8029
NAMEX=vllm-exp
PORTX=8029
U=http://127.0.0.1:$PORTX
MODEL=qwen3.8-27b
MDIR=$D/models/qwen3.8-27b-nvidia-nvfp4   # launch-daily.sh MODEL=; the client tokenizes with the served checkpoint's tokenizer
L2X=$D/eval-l2
# the boot: see the header for why each knob is set
SEQS_X=64
KV_X=${KV_X:-14860000000}
CCX='"max_cudagraph_capture_size":320'
POOL_LO=1360000 POOL_HI=1425000 FREE_MIN=512
R675_POOL=1391795            # the served pin's pool at SEQS 16 (R675 / R793): logged against, not gated
DS=$D/datasets/std-bench
SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json; SG_SHA=35f0e213
SB=$DS/spec_bench_question.jsonl;                 SB_SHA=4b6d33e7
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.30.0}
PROBES=$D/probes
SHIM=$PROBES/vllm_bench_tabby.py
PARSE=$PROBES/parse_container.py   # imported by the summary (unused for these cells)
SUMMARY=$PROBES/std_bench_summary.py
DSS=$PROBES/decode_ss.py
PHASES=${PHASES:-"1 2"}
DEC_KINDS=${DEC_KINDS:-"code prose"}
DEC_CONCS=${DEC_CONCS:-"1 2 4 8 16 24 32 48 64"}
DEC_RUNS=${DEC_RUNS:-3}
DEC_TOK=1024
PASSES=${PASSES:-"A B"}
DATASETS=${DATASETS:-"sharegpt specbench"}
CONCS=${CONCS:-"1 4 8 16 32 48 64"}
SG_N=${SG_N:-400}
SB_N=${SB_N:-480}
SEED=${SEED:-7310}
SB_OUT=${SB_OUT:-256}
WARM_TOK=64
WANT_GPC=${WANT_GPC:-"0 0"}
WANT_MEM=${WANT_MEM:-"4500 4500"}
WANT_PWR=${WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')}
SPREAD_MAX=${SPREAD_MAX:-3}
SPREAD_EXEMPT=${SPREAD_EXEMPT-1}
CACHE_MAX=${CACHE_MAX:-0.01}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-2400}   # R793: Spec-Bench c1 took 549 s
CLIENT_TRC=${CLIENT_TRC:-0}
QUIESCE=${QUIESCE-"hermes hermes-webui owui-proxy"}
CLIENT_NAME=$UNIT-client
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
md5f(){ md5sum < "$1" | cut -c1-32; }

# ---------------- checks before the lock (nothing touched) ----------------
for f in "$LAUNCH27" "$D/launch-daily-v0280.sh" "$LIVE" "$D/daily-power.sh" "$D/shm-sweep.sh" "$D/eval-l2-dio.sh" "$D/lib/gpu-queue.sh" \
         "$D/lib/gateway-drain.sh" "$SHIM" "$PARSE" "$SUMMARY" "$DSS" "$SG" "$SB" \
         "$MDIR/tokenizer.json" "$MDIR/tokenizer_config.json"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ "$(md5f "$LAUNCH27")" = "$LAUNCH27_MD5" ] || { log "ABORT: $LAUNCH27 md5 $(md5f "$LAUNCH27") != $LAUNCH27_MD5 (the 27B launcher changed since this unit was written)"; exit 3; }
grep -q "^DAILY_IMG=$WANT_IMG " "$LAUNCH27" || { log "ABORT: launch-daily.sh DAILY_IMG is not $WANT_IMG"; exit 3; }
sudo docker image inspect "$WANT_IMG" >/dev/null 2>&1 || { log "ABORT: daily image $WANT_IMG not on flan"; exit 3; }
# snapshot the inputs: a probe edited while the unit is queued must not change what this run measures
cp "$SHIM" "$PARSE" "$SUMMARY" "$DSS" "$R/probes/"
[ -e "$PROBES/pp_tg_depth.py" ] && cp "$PROBES/pp_tg_depth.py" "$R/probes/"
sha(){ sha256sum "$1" | cut -c1-8; }
[ "$(sha "$SG")" = "$SG_SHA" ] || { log "ABORT: ShareGPT sha256 $(sha "$SG") != $SG_SHA"; exit 3; }
[ "$(sha "$SB")" = "$SB_SHA" ] || { log "ABORT: Spec-Bench sha256 $(sha "$SB") != $SB_SHA"; exit 3; }
SB_ROWS=$(grep -c . "$SB")
[ "$SB_N" -le "$SB_ROWS" ] || { log "ABORT: SB_N $SB_N > $SB_ROWS Spec-Bench rows"; exit 3; }
sudo docker image inspect "$CLIENT_IMG" >/dev/null 2>&1 || { log "ABORT: $CLIENT_IMG not on flan (the unit never pulls)"; exit 3; }
CENV=(-e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=1)
TRC_ARGS=(); TRC_PY=False; [ "$CLIENT_TRC" = 1 ] && { TRC_ARGS=(--trust-remote-code); TRC_PY=True; }

# preflight (CPU only, --network none), unchanged from R793: the shim patches this image, the 27B tokenizer loads offline,
# Spec-Bench loads through the shim's reader; report-only: templated Spec-Bench prompts filling >= 1 / 2 tier blocks
sudo docker run --rm --network none "${CENV[@]}" -e TABBY_SAMPLES_OUT=/tmp/preflight.samples.tsv -e TRC=$TRC_PY \
  -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro --entrypoint python3 "$CLIENT_IMG" -c '
import json, os, sys; sys.path.insert(0, "/probes")
import vllm_bench_tabby as s; s.install()
import vllm.benchmarks.serve as S
assert S.get_samples.__name__ == "get_samples_recorded", "manifest recorder not installed"
from vllm.tokenizers import get_tokenizer
t = get_tokenizer("/model", trust_remote_code=os.environ["TRC"] == "True")
m = t.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
print("preflight: tokenizer", type(t).__name__, "vocab", len(t), "| template tail", repr(m[-48:]))
from vllm.benchmarks.datasets import datasets as D
d = D.SpecBench(dataset_path="/data/spec_bench_question.jsonl")
print("preflight: specbench rows", len(d.data))
try:
    L = [len(t.encode(t.apply_chat_template([{"role": "user", "content": json.loads(ln)["turns"][0]}],
                                            add_generation_prompt=True, tokenize=False), add_special_tokens=False))
         for ln in open("/data/spec_bench_question.jsonl") if ln.strip()]
    print("preflight: specbench templated prompt tokens max", max(L), "| >= 1472:", sum(x >= 1472 for x in L),
          "| >= 2944:", sum(x >= 2944 for x in L))
except Exception as e:
    print("preflight: specbench block count unavailable:", e.__class__.__name__, str(e)[:120])' > "$R/preflight.log" 2>&1
grep -q '^shim: patched.*sample manifest ON' "$R/preflight.log" && grep -q '^preflight: tokenizer' "$R/preflight.log" \
  && grep -qE "^preflight: specbench rows $SB_ROWS\$" "$R/preflight.log" \
  || { log "ABORT: preflight failed: $(grep -avE '^\s*$' "$R/preflight.log" | tail -2 | cut -c1-200)"; exit 3; }
grep -aE '^(shim|preflight):' "$R/preflight.log" | sed 's/^/  /' | tee -a "$R/audit.log"

# ---------------- queue + lock ----------------
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/gateway-drain.sh
glm_up(){ curl -fsS -m 5 "$GLM_U/health" >/dev/null; }
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' '; }
# a signal while queued: the previous lock-holder skipped its restore because this unit was registered, so boot the
# GLM daily only if the lock is now free, :8029 is down and nobody else is queued.
queued_signal(){ trap 'log "signal during queued restore: ignored"' TERM INT HUP
  log "signal while queued"; exec 9>"$D/gpu-exclusive.lock"
  if flock -n 9 && ! glm_up && [ -z "$(gpu_queue_others)" ]; then
    rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
    log "GPUs free and :8029 down: restoring GLM from $LIVE"
    if timeout -k 20 1380 bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1 && glm_up; then log "GLM daily restored on :8029"
    else log "GLM RESTORE FAILED (boot-queued-restore.log)"; fi
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 4; }
trap queued_signal TERM INT HUP

DECISION="VOID the unit ended before the summary"
FINISHED=0 WAS_RUNNING= CLIENTS_CAPTURED=0 FLAKES=0 BOOTED=0
gpcoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
memoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
pwrlim(){ nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%s%.0f", (NR>1?" ":""), $1}'; }
l2df(){ df -h --output=used,size,pcent "$L2X" 2>/dev/null | tail -1 | tr -s ' '; }
running(){ [ "$(sudo docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
wipe_l2(){ mountpoint -q "$L2X" || return 0; sudo find "$L2X" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
# stop every QUIESCE container that runs now; prints the ones it stopped
quiesce(){ local c s=; for c in $QUIESCE; do running "$c" || continue; sudo docker stop -t 30 "$c" >/dev/null 2>&1 && s="$s $c"; done; echo "${s# }"; }
wait_vram_free(){ local i; for i in $(seq 36); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && return 0; sleep 5; done; return 1; }
teardownx(){ sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAMEX" || return 0
  sudo docker rm -f "$NAMEX" >/dev/null 2>&1; wait_vram_free || log "WARN: VRAM not free 180 s after removing $NAMEX"
  sudo "$D/shm-sweep.sh" quiet >> "$R/shm-sweep.log" 2>&1 || true; }
# vLLM's 10 s stats lines in a container log: max Running, max Waiting, max GPU KV cache usage
engstats(){ printf 'max Running %s, max Waiting %s, max GPU KV cache usage %s %%' \
  "$(grep -aoE 'Running: [0-9]+ reqs' "$1" | tr -dc '0-9\n' | sort -n | tail -1)" \
  "$(grep -aoE 'Waiting: [0-9]+ reqs' "$1" | tr -dc '0-9\n' | sort -n | tail -1)" \
  "$(grep -aoE 'GPU KV cache usage: [0-9.]+%' "$1" | grep -oE '[0-9.]+' | sort -n | tail -1)"; }

finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during finish: ignored"' TERM INT HUP
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  if sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAMEX"; then sudo docker logs "$NAMEX" > "$C/container-final.log" 2>&1; fi
  teardownx
  bash $D/daily-power.sh stock >/dev/null 2>&1
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  if [ "$BOOTED" = 1 ]; then
    if [ -n "$(gpu_queue_others)" ]; then log "GPU queue continues ($(gpu_queue_others)); GLM restore left to the next unit"
    elif timeout -k 20 1380 bash "$LIVE" > "$R/boot-restore.log" 2>&1 && glm_up; then log "GLM daily restored on :8029"
    else log "GLM RESTORE FAILED (boot-restore.log)"; DECISION="VOID GLM daily restore failed; $DECISION"; fi
  fi
  # An interrupted snapshot precedes all client stops/boots: leave those clients untouched.
  if [ "$CLIENTS_CAPTURED" = 1 ]; then
    local now=; now=$(quiesce)   # anything a boot restarted (launch-daily-v0280.sh: owui-proxy) goes back to its start state
    [ -n "$now" ] && log "stopped after the restore: $now (started again below if it ran at the unit's start)"
    if [ -n "$WAS_RUNNING" ]; then
      glm_up || log "WARN: GLM :8029 not serving; restarting the direct clients anyway"
      if sudo docker start $WAS_RUNNING >/dev/null 2>&1; then log "restarted: $WAS_RUNNING"
      else log "RESTART FAILED: $WAS_RUNNING (start by hand: sudo docker start $WAS_RUNNING)"; fi; fi
  fi
  log "eval-l2 at the end: $(l2df)"
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  log "=== $UNIT $1 ==="; log "DECISION: $DECISION"; }
void(){ DECISION="VOID $*"; echo "DECISION: $DECISION" >> "$R/summary.txt"; finish VOID; exit 3; }
# Arm cleanup before locking; BOOTED means restore responsibility, including inherited GLM downtime.
# Keep the libraries' EXIT housekeeping after finish, with its original exit status and quoting.
finish_on_exit(){ local rc=$?; [ "$BOOTED" = 0 ] || finish EXIT; return "$rc"; }
arm_exit_cleanup(){ trap "finish_on_exit; ${1:-:}" EXIT; }
previous_exit_trap=$(trap -p EXIT)
if [ -n "$previous_exit_trap" ]; then
  eval "${previous_exit_trap/trap -- /arm_exit_cleanup }"
else arm_exit_cleanup; fi
unset previous_exit_trap
gpu_lock || exit $?
BOOTED=1
trap 'log "signal"; DECISION="VOID signal (the run was interrupted)"; finish ABORTED; exit 4' TERM INT HUP

# Capture before drain can exit; publish completeness only after every client was inspected.
for c in $QUIESCE; do running "$c" && WAS_RUNNING="$WAS_RUNNING $c"; done; WAS_RUNNING=${WAS_RUNNING# }
CLIENTS_CAPTURED=1
gateway_drain   # Olla routes nothing to :8020 / :8022 until this unit exits
st=$(quiesce)
log "lock held; GLM at entry: $(glm_up && echo healthy || echo down); :$PORTX $(curl -sf -m 5 $U/health >/dev/null && echo answers || echo empty); direct clients stopped: ${st:-none}; phases $PHASES; decode kinds $DEC_KINDS concs $DEC_CONCS runs $DEC_RUNS; passes $PASSES; datasets $DATASETS; concs $CONCS; SG_N $SG_N seed $SEED; SB_N $SB_N out $SB_OUT; boot EXP=1 SEQS=$SEQS_X KV_BYTES=$KV_X PCIE_IPC=1 BSS=1 CC_EXTRA=$CCX; want power $WANT_PWR W; offsets now core $(gpcoff) / memory $(memoff), power $(pwrlim) W; GLM restore helper md5 $(md5f "$LIVE"); eval-l2 $(l2df)"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
[ "$(md5f "$LAUNCH27")" = "$LAUNCH27_MD5" ] || void "under the lock: launch-daily.sh md5 $(md5f "$LAUNCH27") != $LAUNCH27_MD5"
cp "$LAUNCH27" "$R/launch-daily-at-lock.sh"
sudo docker logs glm53 > "$R/glm-final.log" 2>&1 || true
sudo docker logs glm53-preflight > "$R/glm-preflight-final.log" 2>&1 || true
sudo docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
wait_vram_free || log "WARN: VRAM not free 180 s after stopping the GLM daily"
log "GLM daily stopped (logs archived)"
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAMEX"; then
  sudo docker logs "$NAMEX" > "$C/container-foreign-at-entry.log" 2>&1
  log "NOTE: a $NAMEX container existed at entry ($(sudo docker inspect -f '{{.Config.Image}}' "$NAMEX" 2>/dev/null)); log saved, removed"; teardownx; fi
mountpoint -q "$L2X" || sudo bash $D/eval-l2-dio.sh >> "$R/audit.log" 2>&1
mountpoint -q "$L2X" || void "eval-l2 not mounted at $L2X"

printf 'tag\tpass\tdataset\tconc\tt_boot\tlauncher_md5\timage\timage_id\tenv_n\tenv_sha\tkeys_n\twindow\tslots\tcache\tpolicy\tpower_limit_w\tgpc_boot\tmem_boot\tgpc_after\tmem_after\tvram_free\tmem_line\tpwr_after\tpwr_line\tboot_tries\tvllm\taot_hashes\taot_loaded\n' > "$R/boots.tsv"
printf 'tag\tpass\tdataset\tconc\tserial_lo\tserial_hi\twarmups\tcatfile\trc\tcontainer\tn_req\tt0\tt1\tserver\n' > "$R/runs.tsv"
printf 'tag\tforeign\n' > "$R/foreign.tsv"
printf 'tag\tpreemptions\n' > "$R/preempt.tsv"
REF_CFG= REF_IMGID= REF_POOL= REF_AOT=
# B_* = the current boot's record; written to boots.tsv once the boot's work is over (with the after readings).
# keys_n / window / policy are TabbyAPI fields: '-' here. env_sha = sha256 of the container's Args (in order) + Env (sorted).
bootrow(){ printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$B_TAG" "$B_PASS" "$B_DS" "$B_CONC" "$B_T" "$B_MD5" "$B_IMG" "$B_IMGID" "$B_ENVN" "$B_CFG" - - "$B_SLOTS" "$B_POOL" - \
  "$B_PWR" "$B_GPC" "$B_MEM" "${1:--}" "${2:--}" "$B_FREE" "$B_MEMLINE" "${3:--}" "$B_PWRLINE" "$B_TRIES" "$B_VER" \
  "${B_AOT:--}" "$B_AOTN" >> "$R/boots.tsv"; }

bootx(){ local tag=$1 try bl ok=0 st args
  B_TAG=$tag B_PASS=$2 B_DS=$3 B_CONC=$4
  for try in 1 2 3; do   # R233 / r675 / r793: the warmup flake ("CUDA error: invalid argument") is a lottery at this pin
    teardownx
    wipe_l2
    B_MD5=$(md5f "$LAUNCH27"); B_T=$(date -Is); bl=$C/boot-$tag-$try.log
    [ "$B_MD5" = "$LAUNCH27_MD5" ] || void "[$tag] launch-daily.sh md5 $B_MD5 != $LAUNCH27_MD5 (changed mid-run)"
    if env -i HOME="$HOME" USER="$(id -un)" PATH="$PATH" EXP=1 SEQS=$SEQS_X KV_BYTES=$KV_X PCIE_IPC=1 BSS=1 CC_EXTRA="$CCX" \
         POOL_MIN=$POOL_LO POOL_MAX=$POOL_HI MIN_FREE_MIB=$FREE_MIN bash "$LAUNCH27" > "$bl" 2>&1 \
       && curl -sf -m 5 $U/health >/dev/null; then ok=1; break; fi
    sudo docker logs "$NAMEX" > "$C/container-$tag-failed-$try.log" 2>&1
    FLAKES=$((FLAKES + 1))
    log "[$tag] boot attempt $try/3 FAILED: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,160}' "$bl" | tail -1)"
  done
  [ "$ok" = 1 ] || void "[$tag] NO BOOT after 3 attempts: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,160}' "$bl" | tail -1)"
  B_TRIES=$try
  st=$(quiesce); [ -n "$st" ] && log "[$tag] stopped again after the boot: $st (launch-daily-v0280.sh restarts owui-proxy)"
  bash $D/daily-power.sh stock 2>&1 | grep -aiE '^WARN' | tee -a "$R/audit.log"   # EXP boots stay at stock; belt and braces
  sudo docker inspect "$NAMEX" > "$C/inspect-$tag.json" 2>/dev/null
  sudo docker logs "$NAMEX" > "$C/container-$tag-boot.log" 2>&1
  B_IMG=$(sudo docker inspect -f '{{.Config.Image}}' "$NAMEX" 2>/dev/null)
  B_IMGID=$(sudo docker inspect -f '{{.Image}}' "$NAMEX" 2>/dev/null | cut -c1-19)
  args=$(sudo docker inspect -f '{{json .Args}}' "$NAMEX" 2>/dev/null)
  # Args in order + Env SORTED: docker inspect lists Env in a nondeterministic order across boots of one launcher (R793)
  B_CFG=$( { echo "$args"; sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAMEX" 2>/dev/null | sort; } | sha256sum | cut -c1-12)
  B_ENVN=$(sudo docker inspect -f '{{len .Config.Env}}' "$NAMEX" 2>/dev/null)
  B_POOL=$(grep -aoE 'Pool [0-9]+' "$bl" | tail -1 | tr -dc 0-9)
  B_SLOTS=$(grep -aoE 'SEQS [0-9]+' "$bl" | tail -1 | tr -dc 0-9)
  B_VER=$(grep -aoE '\(vllm [^,]+' "$bl" | tail -1 | cut -d' ' -f2)
  B_FREE=$(vram_free | tr -s ' ' '/' | sed 's#/$##')
  B_MEMLINE=$(grep -aoE 'clock offsets: .*' "$bl" | tail -1 | tr '\t' ' ')
  B_PWRLINE="launcher: $(grep -aoE 'power [0-9./]+ W' "$bl" | tail -1)"
  B_PWR=$(pwrlim); B_GPC=$(gpcoff); B_MEM=$(memoff)
  B_AOT=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$C/container-$tag-boot.log" | cut -d/ -f2 | sort -u | paste -sd, -)
  B_AOTN=$(grep -ac 'Directly load AOT' "$C/container-$tag-boot.log")
  log "[$tag] UP after $try attempt(s): image $B_IMG ($B_IMGID); launcher md5 $B_MD5; vllm $B_VER; engine config sha $B_CFG (env $B_ENVN); pool $B_POOL (SEQS 16 at this pin: $R675_POOL); SEQS $B_SLOTS; aot hashes ${B_AOT:-none} (loaded $B_AOTN); block $(grep -aoE 'Setting attention block size to [0-9]+' "$C/container-$tag-boot.log" | head -1 | tr -dc 0-9); power $B_PWR W ($B_PWRLINE); core offsets $B_GPC; memory offsets $B_MEM (${B_MEMLINE:-no clock-offset line}); VRAM free $B_FREE"
  [ "$(grep -ac '0.29 nvfp4 EXP UP' "$bl")" -ge 1 ] || { bootrow; void "[$tag] the launcher did not report the EXP path (no '0.29 nvfp4 EXP UP' line)"; }
  [ "$B_IMG" = "$WANT_IMG" ] || { bootrow; void "[$tag] image '$B_IMG' != $WANT_IMG"; }
  [ -n "$B_POOL" ] || { bootrow; void "[$tag] no pool in the launcher output"; }
  [ "$B_SLOTS" = "$SEQS_X" ] || { bootrow; void "[$tag] SEQS '$B_SLOTS' != $SEQS_X"; }
  case "$args" in *'--max-num-seqs 64 '*) ;; *) bootrow; void "[$tag] --max-num-seqs 64 not on the container";; esac
  case "$args" in *"--kv-cache-memory-bytes $KV_X "*) ;; *) bootrow; void "[$tag] --kv-cache-memory-bytes $KV_X not on the container";; esac
  case "$args" in *max_cudagraph_capture_size*320*) ;; *) bootrow; void "[$tag] max_cudagraph_capture_size 320 not on the container";; esac
  [ "$B_PWR" = "$WANT_PWR" ] || { bootrow; void "[$tag] power limits at boot '$B_PWR' W, want stock '$WANT_PWR' W"; }
  if [ "$B_GPC" != "$WANT_GPC" ] || [ "$B_MEM" != "$WANT_MEM" ]; then bootrow
    void "[$tag] clock offsets at boot: core '$B_GPC' memory '$B_MEM', want core '$WANT_GPC' memory '$WANT_MEM'"; fi
  if [ -z "$REF_CFG" ]; then REF_CFG=$B_CFG REF_IMGID=$B_IMGID REF_POOL=$B_POOL
    [ "$B_POOL" = "$R675_POOL" ] || log "[$tag] NOTE: pool $B_POOL differs from the SEQS 16 pool at this pin ($R675_POOL); the band $POOL_LO-$POOL_HI passed it"
  elif [ "$B_CFG/$B_IMGID/$B_POOL" != "$REF_CFG/$REF_IMGID/$REF_POOL" ]; then bootrow
    void "[$tag] config changed mid-run: engine config / image id / pool $B_CFG/$B_IMGID/$B_POOL vs first boot $REF_CFG/$REF_IMGID/$REF_POOL"; fi
  # the AOT artifact: reference = the first boot that logs one; a later boot logging another set is a different compile
  if [ -n "$B_AOT" ]; then
    if [ -z "$REF_AOT" ]; then REF_AOT=$B_AOT
    elif [ "$B_AOT" != "$REF_AOT" ]; then bootrow; void "[$tag] AOT compile artifact changed: $B_AOT vs $REF_AOT"; fi
  else log "[$tag] WARN: no torch_aot_compile hash in the boot log (compile identity unverified for this boot)"; fi; }

snap(){ curl -s -m 20 "$U/metrics" > "$1" 2>/dev/null; }
# prom A B -> JSON of counter deltas B - A (summed over label sets; null when B lacks the metric) + B's queue gauges (r793)
prom(){ python3 - "$1" "$2" <<'PY'
import json, sys
def load(p):
    m = {}
    for ln in open(p, errors="replace"):
        if ln.startswith("#") or not ln.strip():
            continue
        k, _, v = ln.rstrip("\n").rpartition(" ")
        try:
            m[k.split("{")[0]] = m.get(k.split("{")[0], 0.0) + float(v)
        except ValueError:
            pass
    return m
a, b = load(sys.argv[1]), load(sys.argv[2])
K = {"requests": "vllm:request_success_total", "e2e_count": "vllm:e2e_request_latency_seconds_count",
     "prompt": "vllm:prompt_tokens_total", "gen": "vllm:generation_tokens_total",
     "pc_q": "vllm:prefix_cache_queries_total", "pc_h": "vllm:prefix_cache_hits_total",
     "ext_q": "vllm:external_prefix_cache_queries_total", "ext_h": "vllm:external_prefix_cache_hits_total",
     "drafts": "vllm:spec_decode_num_drafts_total", "draft_tok": "vllm:spec_decode_num_draft_tokens_total",
     "accepted": "vllm:spec_decode_num_accepted_tokens_total", "preempt": "vllm:num_preemptions_total",
     "offload_bytes": "vllm:kv_offload_total_bytes_total"}
d = {k: (round(b[v] - a.get(v, 0.0), 3) if v in b else None) for k, v in K.items()}
d["running_end"] = b.get("vllm:num_requests_running")
d["waiting_end"] = b.get("vllm:num_requests_waiting")
print(json.dumps(d))
PY
}
jget(){ python3 -c 'import json,sys; d=json.loads(sys.argv[1]); v=d.get(sys.argv[2]); r=d.get("requests"); v=(r if r is not None else d.get("e2e_count")) if sys.argv[2]=="req" else v; print("-" if v is None else (int(v) if float(v).is_integer() else v))' "$1" "$2"; }

# ======================= PHASE 1: decode_ss on one boot =======================
P1_BAD=
phase1(){ local kind c tag=P1-decode m0 m1 d out res pre ga ma pa
  bootx "$tag" - decode -
  NONCE=$(( $(date +%s) % 100000 ))
  log "[phase 1] decode_ss $DEC_KINDS at c = $DEC_CONCS, $DEC_RUNS runs, $DEC_TOK forced tokens, seed prefix r794-$NONCE-"
  for kind in $DEC_KINDS; do for c in $DEC_CONCS; do
    m0=$R/phase1/metrics-$kind-c$c-0.prom; m1=$R/phase1/metrics-$kind-c$c-1.prom; out=$R/phase1/probe-$kind-c$c
    snap "$m0"
    python3 "$R/probes/decode_ss.py" --url $U --model $MODEL --kind "$kind" --conc "$c" --tokens $DEC_TOK --runs $DEC_RUNS \
      --seed-prefix "r794-$NONCE-" --out "$R/decode-$kind.jsonl" > "$out.out" 2> "$out.err"
    sleep 2; snap "$m1"
    d=$(prom "$m0" "$m1"); echo "$d" > "$R/phase1/server-$kind-c$c.json"
    pre=$(jget "$d" preempt)
    res=$(grep -a '^RESULT ' "$out.out" | tail -1 | cut -d' ' -f2-)
    if [ -z "$res" ]; then P1_BAD="$P1_BAD $kind-c$c:no-result"
      log "[phase 1 $kind c$c] PROBE FAILED: $(grep -a . "$out.out" "$out.err" 2>/dev/null | tail -1 | cut -c1-200)"
    else
      local nr; nr=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["runs"])' "$res")
      [ "$nr" = "$DEC_RUNS" ] || P1_BAD="$P1_BAD $kind-c$c:runs=$nr"
      log "[phase 1 $kind c$c] $(python3 -c 'import json,sys; s=json.loads(sys.argv[1]); print("agg %s (min/max %s), per stream %s, accept/draft %s, TTFT %s s, windows %s s, good runs %s" % tuple(s[k] for k in ("ss_agg_tps_median", "ss_agg_tps_min_max", "ss_per_stream_tps_median", "accept_per_draft_median", "ttft_s_median", "ss_window_s", "runs")))' "$res"); server: preemptions $pre, accepted $(jget "$d" accepted) of $(jget "$d" draft_tok) drafted, generation tokens $(jget "$d" gen), requests $(jget "$d" req), running/waiting at end $(jget "$d" running_end)/$(jget "$d" waiting_end)"
    fi
    [ "$pre" = 0 ] || P1_BAD="$P1_BAD $kind-c$c:preemptions=$pre"
  done; done
  sudo docker logs "$NAMEX" > "$C/container-$tag.log" 2>&1
  local nerr; nerr=$(grep -acE ' ERROR |^ERROR|Traceback|OutOfMemoryError|illegal memory|CUDA error' "$C/container-$tag.log")
  [ "$nerr" = 0 ] || P1_BAD="$P1_BAD engine-errors=$nerr"
  ga=$(gpcoff); ma=$(memoff); pa=$(pwrlim); bootrow "$ga" "$ma" "$pa"
  [ "$ga" = "$WANT_GPC" ] && [ "$ma" = "$WANT_MEM" ] && [ "$pa" = "$WANT_PWR" ] || void "[$tag] clocks / power after phase 1: core '$ga' memory '$ma' power '$pa' W"
  log "[phase 1] engine: $(engstats "$C/container-$tag.log"); error lines $nerr; preemptions total $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{s+=$NF} END {print s+0}'); VRAM free now $(vram_free)"
  P1_STATE=$([ -z "$P1_BAD" ] && echo OK || echo "FLAGGED:${P1_BAD}")
  log "PHASE 1: $P1_STATE"; }
P1_STATE="not run"
case " $PHASES " in *" 1 "*) phase1 ;; esac

# ======================= PHASE 2: vllm bench serve, fresh boot per cell =======================
# conc concurrent non-stream 64-token chat requests on short out-of-set prompts (r793's warm-up, :8029 here)
warm(){ python3 - "$MODEL" "$2" "$1" "$WARM_TOK" "$U" <<'EOF'
import json, sys, threading, urllib.request
model, conc, tag, n, url = sys.argv[1], max(int(sys.argv[2]), 1), sys.argv[3], int(sys.argv[4]), sys.argv[5]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
got = []
def one(i):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": n, "min_tokens": n,
            "messages": [{"role": "user", "content": f"Warm-up {tag} slot {i}: describe a lighthouse at dusk in three sentences."}]}
    try:
        rq = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        got.append((json.load(op.open(rq, timeout=300)).get("usage") or {}).get("completion_tokens"))
    except Exception as e:
        print(f"warm-up error: {e.__class__.__name__}: {e}"[:200])
ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"{len(got)}/{conc} ok, completion tokens {got}")
EOF
}

first=1
cell(){ local ps=$1 ds=$2 conc=$3 n=$4 rc t0 t1 tag=$1-$2-c$3 w0 m0 m1 wd sd q g fr pre ga ma pa waves; shift 4
  bootx "$tag" "$ps" "$ds" "$conc"
  w0=$C/metrics-$tag-0-prewarm.prom; m0=$C/metrics-$tag-1-start.prom; m1=$C/metrics-$tag-2-end.prom
  snap "$w0"
  log "[$tag] warm-up: $(warm "$tag" "$conc" 2>&1 | tail -1)"
  sleep 2; snap "$m0"
  # metrics self-test: the warm-up must move the counters by exactly conc requests and conc x WARM_TOK tokens, spec on
  wd=$(prom "$w0" "$m0"); q=$(jget "$wd" req); g=$(jget "$wd" gen)
  log "[$tag] warm-up counters: requests $q, generation tokens $g, drafts $(jget "$wd" drafts), preemptions $(jget "$wd" preempt), running/waiting now $(jget "$wd" running_end)/$(jget "$wd" waiting_end)"
  if [ "$q" != "$conc" ] || [ "$g" != "$((conc * WARM_TOK))" ] || [ "$(jget "$wd" pc_h)" = - ] || [ "$(jget "$wd" ext_h)" = - ] \
     || [ "$(jget "$wd" accepted)" = - ] || [ "$(jget "$wd" draft_tok)" = - ] || [ "$(jget "$wd" preempt)" = - ] \
     || [ "$(jget "$wd" running_end)" != 0 ] || [ "$(jget "$wd" waiting_end)" != 0 ]; then
    [ $first = 1 ] && void "[$tag] metrics self-test failed on the first cell (want requests $conc, generation tokens $((conc * WARM_TOK)), GPU and external (tier) prefix-cache, spec-decode and preemption counters present, nothing running): $wd"
    log "[$tag] WARN: metrics self-test failed: $wd"; fi
  t0=$(date -Is)
  timeout -k 30 "$CLIENT_TIMEOUT" sudo docker run --rm --name "$CLIENT_NAME" --network host "${CENV[@]}" \
    -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro -v "$R/results":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url "$U" --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer /model ${TRC_ARGS[@]+"${TRC_ARGS[@]}"} --temperature 0 \
    --request-rate inf --max-concurrency "$conc" --num-warmups 0 --num-prompts "$n" --no-oversample \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    --metadata "unit=$UNIT" "tag=$tag" "pass=$ps" "dataset=$ds" "conc=$conc" "num_warmups=0" \
      "unit_warmup=${conc}x non-stream out-of-set" "boot=fresh per cell" "thinking=template-default" \
      "length_forcing=min_tokens" "client=vllm-v0.30.0+vllm_bench_tabby" \
      "server=vllm launch-daily.sh EXP 1 SEQS $SEQS_X KV_BYTES $KV_X port $PORTX" "power=stock" \
    "$@" > "$C/client-$tag.log" 2>&1
  rc=$?; t1=$(date -Is); sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  sleep 3; snap "$m1"
  sd=$(prom "$m0" "$m1"); echo "$sd" > "$R/server/$tag.json"
  sudo docker logs "$NAMEX" > "$C/container-$tag.log" 2>&1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ps" "$ds" "$conc" 0 0 0 - "$rc" - "$n" "$t0" "$t1" \
    "server/$tag.json" >> "$R/runs.tsv"
  ga=$(gpcoff); ma=$(memoff); pa=$(pwrlim); bootrow "$ga" "$ma" "$pa"
  q=$(jget "$sd" req); fr=$([ "$q" = - ] && echo "?" || echo $((q - n)))   # every client request finishes (rc 0) or fails; extra = foreign
  printf '%s\t%s\n' "$tag" "$fr" >> "$R/foreign.tsv"
  [ "$fr" = 0 ] || log "[$tag] WARN: the server finished $q requests for $n client requests (foreign traffic, or failures)"
  pre=$(jget "$sd" preempt); printf '%s\t%s\n' "$tag" "$pre" >> "$R/preempt.tsv"
  [ "$pre" = 0 ] || log "[$tag] WARN: $pre preemptions during the cell (the pool did not hold $conc requests)"
  waves=$(awk -v n="$n" -v c="$conc" 'BEGIN {printf "%.2f", n / c}')
  log "[$tag] rc $rc; $(grep -c '^shim: patched' "$C/client-$tag.log") shim line; $(grep -aoE '^shim: [0-9]+ samples' "$C/client-$tag.log" | cut -d' ' -f2) sampled; waves $waves ($n requests / c$conc); server: requests $q/$n, generation tokens $(jget "$sd" gen), prompt tokens $(jget "$sd" prompt), cached $(jget "$sd" pc_h) GPU + $(jget "$sd" ext_h) tier, accepted $(jget "$sd" accepted) of $(jget "$sd" draft_tok) drafted, preemptions $pre, running/waiting at end $(jget "$sd" running_end)/$(jget "$sd" waiting_end); engine $(engstats "$C/container-$tag.log"); $(grep -aE '^(Successful requests|Failed requests|Benchmark duration|Output token throughput|Median TPOT)' "$C/client-$tag.log" | sed -E 's/ {2,}/ /g' | paste -sd';' -); engine ERROR lines $(grep -acE ' ERROR |^ERROR' "$C/container-$tag.log") tracebacks $(grep -ac Traceback "$C/container-$tag.log") OOM $(grep -acE 'OutOfMemoryError|out of memory' "$C/container-$tag.log"); offsets after core $ga memory $ma, power $pa W"
  # the shim's main() path first runs here: a client that produced nothing must not burn 27 more boots
  if [ $first = 1 ]; then first=0
    [ -s "$R/results/$tag.json" ] && [ -s "$R/results/$tag.samples.tsv" ] && grep -q '^shim: patched' "$C/client-$tag.log" \
      || void "first client run produced no result/manifest: $(grep -avE '^\s*$' "$C/client-$tag.log" | tail -2 | cut -c1-200)"; fi
  [ "$ga" = "$WANT_GPC" ] && [ "$ma" = "$WANT_MEM" ] && [ "$pa" = "$WANT_PWR" ] || void "[$tag] clocks / power after the cell: core '$ga' memory '$ma' power '$pa' W"; }

RUN_DECISION="phase 2 not run"
if [ "$(echo " $PHASES " | grep -c ' 2 ')" = 1 ]; then
  for ps in $PASSES; do
    for ds in $DATASETS; do
      for c in $CONCS; do
        case $ds in
          sharegpt)  cell "$ps" sharegpt "$c" "$SG_N" --dataset-name sharegpt --dataset-path "/data/$(basename "$SG")" --seed "$SEED" ;;
          specbench) cell "$ps" specbench "$c" "$SB_N" --dataset-name spec_bench --dataset-path "/data/$(basename "$SB")" \
                       --spec-bench-output-len "$SB_OUT" --seed "$SEED" ;;
          *) void "unknown dataset $ds" ;;
        esac
      done
    done
  done
  log "all cells done; boot attempts that failed: $FLAKES"
  python3 "$R/probes/std_bench_summary.py" --runs "$R/runs.tsv" --results "$R/results" --boots "$R/boots.tsv" \
    --sb-data "$SB" --sb-out "$SB_OUT" --want-gpc "$WANT_GPC" --want-mem "$WANT_MEM" --want-pwr "$WANT_PWR" --cache-max "$CACHE_MAX" \
    --spread-max "$SPREAD_MAX" --spread-exempt "$SPREAD_EXEMPT" --decide "$PASSES" --json "$R/summary.json" \
    --csv "$R/summary.csv" > "$R/summary.txt" 2>&1
  sed 's/^/  /' "$R/summary.txt" | tee -a "$R/audit.log"
  d=$(grep -a '^DECISION: ' "$R/summary.txt" | tail -1 | sed 's/^DECISION: //')
  RUN_DECISION=${d:-"VOID the summary printed no decision: $(tail -1 "$R/summary.txt" | cut -c1-200)"}
  log "RUN DECISION (R731 rule over the whole run, as R793): $RUN_DECISION"
  # [R794] the same rule per cell (dataset x concurrency); cells-verdict.tsv; last line = the composed verdict
  python3 - "$R" "$PASSES" "$DATASETS" "$CONCS" "$SPREAD_MAX" "$SPREAD_EXEMPT" "$RUN_DECISION" > "$R/cells-verdict.txt" 2>&1 <<'PY'
import csv, json, re, sys
R, passes, dss, concs, smax, exempt, run_dec = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), \
    [int(c) for c in sys.argv[4].split()], float(sys.argv[5]), {int(x) for x in sys.argv[6].split(",") if x}, sys.argv[7]
S = json.load(open(f"{R}/summary.json"))
rows = {r["tag"]: r for r in S["runs"]}
spreads = {(ds, lab): s for ds, lab, s in S["spreads"]}
tsv = lambda f: {r["tag"]: r for r in csv.DictReader(open(f"{R}/{f}"), delimiter="\t")}  # noqa: E731
foreign, preempt = tsv("foreign.tsv"), tsv("preempt.tsv")
alltags = [f"{p}-{d}-c{c}" for p in passes for d in dss for c in concs]
mention = lambda t, s: re.search(rf"(?<![\w-]){re.escape(t)}(?!\w)", s) is not None  # noqa: E731
glob = [f for f in S["flags"] if not any(mention(t, f) for t in alltags)]   # flags naming no cell fail every cell
out, ok_n = [], 0
for ds in dss:
    for c in concs:
        tags = [f"{p}-{ds}-c{c}" for p in passes]
        why = []
        if run_dec.startswith("VOID"):
            why.append("run VOID")
        for t in tags:
            if "out_tps" not in rows.get(t, {}):
                why.append(f"{t} missing")
            why += [f"flag: {f}" for f in S["flags"] if mention(t, f)]
            fv, pv = (foreign.get(t) or {}).get("foreign", "?"), (preempt.get(t) or {}).get("preemptions", "?")
            if fv != "0":
                why.append(f"{t} foreign {fv}")
            if pv != "0":
                why.append(f"{t} preemptions {pv}")
        why += [f"flag: {f}" for f in glob]
        s = spreads.get((ds, f"c{c}"))
        if len(passes) < 2:
            why.append("a single pass: no A/B replication")
        elif s is None:
            why.append("no A/B spread")
        elif s > smax and c not in exempt:
            why.append(f"A/B spread {s:.2f} % > {smax:g} %")
        vals = [rows.get(t, {}).get("out_tps") for t in tags] + [None, None]
        a_, b_ = vals[0], vals[1]
        verdict = "PUBLISHABLE" if not why else "NOT-PUBLISHABLE"
        ok_n += not why
        out.append((ds, c, a_, b_, s, "yes" if c not in exempt else "no", verdict, "; ".join(dict.fromkeys(why))))
with open(f"{R}/cells-verdict.tsv", "w") as fh:
    fh.write("dataset\tconc\tout_tps_A\tout_tps_B\tspread_pct\tspread_gated\tverdict\treasons\n")
    for r in out:
        fh.write("\t".join("-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + "\n")
for ds, c, a_, b_, s, g, v, w in out:
    print(f"{ds:<10} c{c:<3} A {a_ if a_ is None else round(a_, 1)} B {b_ if b_ is None else round(b_, 1)} "
          f"spread {'-' if s is None else f'{s:.2f} %'} (gated {g}): {v}{'  ' + w if w else ''}")
bad = [f"{ds} c{c} ({w[:160]}{'...' if len(w) > 160 else ''})" for ds, c, _, _, _, _, v, w in out if v != "PUBLISHABLE"]
print(f"CELLS: {ok_n}/{len(out)} PUBLISHABLE" + (f"; not: {'; '.join(bad)}" if bad else ""))
PY
  sed 's/^/  /' "$R/cells-verdict.txt" | tee -a "$R/audit.log"
fi
CELLS=$(grep -a '^CELLS: ' "$R/cells-verdict.txt" 2>/dev/null | tail -1 | sed 's/^CELLS: //')
case "$RUN_DECISION" in
  VOID*) DECISION="$RUN_DECISION" ;;
  "phase 2 not run") DECISION="NOT-PUBLISHABLE phase 2 not run (PHASES=$PHASES); phase 1 $P1_STATE" ;;
  *) if [ -z "$CELLS" ]; then DECISION="VOID the per-cell verdict printed nothing: $(tail -1 "$R/cells-verdict.txt" 2>/dev/null | cut -c1-200)"
     else n_ok=${CELLS%%/*}; n_all=$(echo "$CELLS" | sed -E 's#^[0-9]+/([0-9]+).*#\1#')
       if [ "$n_ok" = "$n_all" ] && [ "$P1_STATE" = OK ]; then DECISION="PUBLISHABLE (phase 1 OK; $CELLS; run: $RUN_DECISION)"
       else DECISION="NOT-PUBLISHABLE $CELLS; phase 1 $P1_STATE; run: $RUN_DECISION"; fi; fi ;;
esac
finish DONE
