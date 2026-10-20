#!/usr/bin/env bash
# r27-ba4-std-bench (2026-10-20): R793's standard `vllm bench serve` benchmark (ShareGPT V3 + Spec-Bench) re-measured on
#   the BA4 27B daily (r27-l1a-std-bench.sh with names and launcher md5 changed). Measurement protocol, instruments, shapes, passes and seeds are unchanged from R793 (2026-09-28), for
#   dashed "prefill interleaved" lines on the 27B public README's decode figure (docs/img/decode-scaling-16.svg), as the
#   Flash-Next README has them (user 2026-09-28). The solid lines there are R675 (decode_ss.py, one boot, 2026-09-23).
# LINEAGE: r787d-std-bench.sh (= R731b's protocol with r736's instrument fixes) for the protocol; r675-27b-curves.sh for how a
#   unit boots the 27B daily; r27-l1a-curves.sh for taking the GPUs from GLM and restoring it with queue hand-off + shm sweep.
# WHAT IS MEASURED: /srv/qwen5090/launch-daily.sh exactly as the daily path runs it (EXP unset -> :8020, container vllm-27b,
#   nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV pinned 14.86 GB, MTP ns3, SEQS 16, pcie_ipc, batch-sharded sampling, CPU tier 16 GiB,
#   native disk tier on /srv/qwen5090/native-l2), launcher md5 7cb1a981b0d7db5a54dd8e03c52e1b09 asserted before the lock,
#   under it and at every boot. :8020 because r675 measured the daily on its own port: the 27B daily is not live while
#   GLM holds the GPUs, and this is the served configuration (EXP=1 would boot the :8029 experiment shape instead).
#   Image vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan (BA4 since 2026-10-20) selects v0.31/V2, triton mamba, FlashInfer GDN prefill, split-KV=1 and
#   draft UB plan=1. The launcher deliberately keeps the literal "0.29 nvfp4 DAILY UP" label; its version field is v0.31.
# PROTOCOL (R731b / R787d, changes marked [27B]):
#   vLLM v0.30.0 client (vllm/vllm-openai:v0.30.0, never pulled) + probes/vllm_bench_tabby.py UNCHANGED. [27B] The shim is
#   kept against vLLM on purpose: its Spec-Bench fix (get_samples does not forward skip_chat_template, so the client
#   templates the prompt and the server templates it again) is a client bug, not a TabbyAPI one; the v0.30.0 image has no
#   pandas for SpecBench.load_data; the sample manifest is what the summary's same-sample check reads; and min_tokens =
#   max_completion_tokens (TABBY_FORCE_LEN=1) is native in vLLM (decode_ss.py forces length the same way on this daily,
#   with MTP acceptance 0.46-0.68 in R675). The request is therefore byte-for-byte the Flash-Next R787d request shape.
#   ShareGPT V3 SG_N 400 --seed 7310 (reference-reply lengths), Spec-Bench all 480 x 256 tokens, backend openai-chat, stream +
#   include_usage, --temperature 0, thinking = the served template default (reasoning_effort medium), --request-rate inf,
#   --max-concurrency c, --num-warmups 0, percentiles 50/90/99, --save-detailed. [27B] c = 1 2 4 6 8 12 16 (the x axis of
#   the 27B figure; 16 = the daily's SEQS). Passes A and B, each ShareGPT c1..c16 then Spec-Bench c1..c16; a FRESH boot of
#   the daily for every cell (28 boots): GPU prefix cache and CPU tier start empty in every cell.
#   Before each cell: `conc` concurrent non-stream 64-token chat requests on short out-of-set prompts (the unit's warm-up).
# SERVER-SIDE ACCOUNTING [27B]: this daily logs no per-request lines (no --enable-log-requests), so the TabbyAPI container
#   parsing is replaced by /metrics: raw snapshots (cells/metrics-<tag>-{0-prewarm,1-start,2-end}.prom) around the warm-up
#   and around the client; the start->end delta goes to server/<tag>.json and runs.tsv's `server` column, which
#   probes/std_bench_summary.py (R793 addition, inert on older runs) reads for: finished requests == client requests
#   (foreign traffic), generation tokens == client sum(output_lens), cached share (GPU prefix + external tier hits over
#   prompt tokens), tau / depth from the spec-decode counters. Self-test on every warm-up: the counters must move by
#   exactly conc requests and conc x 64 generation tokens; on the first cell a failure (e.g. a metric renamed) is VOID at
#   once, not after three hours.
# DISK TIER [27B]: the fresh boot empties the GPU cache and the CPU tier but not native-l2. The tier stores prompt blocks
#   only (offload_prompt_only) of 1,472 tokens; vLLM's ShareGPT sampler keeps prompts <= 1,024 tokens, so no ShareGPT
#   prompt can be stored or hit. A Spec-Bench prompt >= 1,472 tokens could be stored in one cell and hit in a later one:
#   the preflight counts them (report-only), and the per-cell cached share <= CACHE_MAX (1 %) is the gate, as R731. The
#   unit never deletes tier files. df of native-l2 is logged at the start and the end.
# POWER [27B]: POWER=stock (default) sets daily-power.sh stock after each boot, as r675 did; every published 27B number,
#   including the R675 solid lines these dashed lines are drawn over, is at stock (600 / 575 W), and daily-power.sh's
#   policy is "experiments at stock so published numbers stay comparable". POWER=cap keeps the launcher's 400 W cap (the
#   daily's production setting) and gates on 400 / 400 W instead. Every boot and every cell end reads power.limit and the
#   core / memory offsets (want core 0, memory +4500, set by the launcher itself at every boot); a drift is VOID.
# TRAFFIC: Olla drained for the unit's lifetime (lib/gateway-drain.sh). Direct clients stopped
#   (QUIESCE, default hermes hermes-webui owui-proxy): owui-proxy's upstream list falls through to 172.17.0.1:8020 when
#   :8022 is down, and launch-daily-v0280.sh ends with `docker restart owui-proxy`, so the unit stops it again after EVERY
#   boot. At the end only the containers that were running at the unit's start are started again.
# RESTORE: archive glm53's logs and remove glm53 / glm53-preflight before the first 27B boot. After every 27B teardown,
#   shm-sweep.sh quiet removes orphan CPU-offload shm. Restore GLM (:8029) via r858/restart-glm-daily.sh unless another
#   unit is queued; keep client configs as they were. A queued signal restores only with the lock free and :8029 down.
# DECISION (pre-registered, the summary prints it; last audit.log line `DECISION: ...`): R731's rule unchanged.
#   VOID <why>  a boot failed 3 times, clocks / power / launcher / image / container config / pool changed, the metrics
#               self-test failed on the first cell, preflight failed, or a signal. The unit stops at the first one.
#   PUBLISHABLE both passes complete; every cell rc 0, 0 failed, completed == num_prompts; server requests == client
#               requests and server generation tokens == client output tokens; Spec-Bench outputs all SB_OUT; the same
#               sample manifest in every cell of a dataset; cached <= 1 % per cell; A/B spread on output tok/s <= 3 % at
#               every level except c1 (SPREAD_EXEMPT=1, as R731: c1 is published with its spread beside it).
#   NOT-PUBLISHABLE <why> otherwise.
# GPU BUDGET ~3 h 20 min (r793-IMPL-NOTES.md): 28 boots x ~3.5 min (R675: 151 s per boot plus one 71 s warmup flake in
#   two attempts) ~ 100 min; measurement ~50 min per pass from R675's decode rates x ~0.85; + drain wait (<= 15 min) +
#   GLM restore (allow up to 23 min). Historical GPU estimate; L1a boot/compile time is unmeasured. RuntimeMaxSec 43200.
# RUN (on flan after installing this file; .new + mv, never over a running file):
#   sudo systemd-run --unit=r27-ba4-std-bench --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r27-ba4-std-bench.sh
#   Stop: sudo systemctl stop r27-ba4-std-bench (teardown, queue-aware GLM restore, restart stopped clients).
set -uo pipefail
export HOME=${HOME:-$HOME}
export PATH="$HOME/.local/bin:$PATH"   # r675: launch-daily-v0280.sh pre-warms with llama-benchy (~/.local/bin)
UNIT=${UNIT:-r27-ba4-std-bench}
D=/srv/qwen5090
R=${R:-$D/results/$(date -u +%F)-$UNIT-$(date -u +%H%M)}
[ -e "$R/audit.log" ] && { echo "ABORT: $R already holds a run"; exit 3; }
mkdir -p "$R/results" "$R/probes" "$R/cells" "$R/server"
cp "$0" "$R/" 2>/dev/null
C=$R/cells
LAUNCH27=$D/launch-daily.sh
LAUNCH27_MD5=${LAUNCH27_MD5:-7cb1a981b0d7db5a54dd8e03c52e1b09}
WANT_IMG=${WANT_IMG:-vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan}
R675_POOL=1391795            # R675's boot of the same daily path (2026-09-23): logged against, not gated
LIVE=$D/r858/restart-glm-daily.sh
GLM_U=http://127.0.0.1:8029
NAME27=vllm-27b
U=http://127.0.0.1:8020
MODEL=qwen3.8-27b
MDIR=$D/models/qwen3.8-27b-nvidia-nvfp4   # launch-daily.sh MODEL=; the client tokenizes with the served checkpoint's tokenizer
DS=$D/datasets/std-bench
SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json; SG_SHA=35f0e213
SB=$DS/spec_bench_question.jsonl;                 SB_SHA=4b6d33e7
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.30.0}
PROBES=$D/probes
SHIM=$PROBES/vllm_bench_tabby.py
PARSE=$PROBES/parse_container.py   # imported by the summary (unused for these cells)
SUMMARY=$PROBES/std_bench_summary.py
PASSES=${PASSES:-"A B"}
DATASETS=${DATASETS:-"sharegpt specbench"}
CONCS=${CONCS:-"1 2 4 6 8 12 16"}
SG_N=${SG_N:-400}
SB_N=${SB_N:-480}
SEED=${SEED:-7310}
SB_OUT=${SB_OUT:-256}
WARM_TOK=64
WANT_GPC=${WANT_GPC:-"0 0"}
WANT_MEM=${WANT_MEM:-"4500 4500"}
POWER=${POWER:-stock}
case "$POWER" in
  stock) WANT_PWR=${WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')} ;;
  cap)   WANT_PWR=${WANT_PWR:-"400 400"} ;;
  *) echo "ABORT: POWER must be stock or cap (got $POWER)"; exit 3 ;;
esac
SPREAD_MAX=${SPREAD_MAX:-3}
SPREAD_EXEMPT=${SPREAD_EXEMPT-1}
CACHE_MAX=${CACHE_MAX:-0.01}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-2400}   # R731's 1200 s was sized for Flash-Next; 27B Spec-Bench c1 ~ 800-900 s
CLIENT_TRC=${CLIENT_TRC:-0}              # 1 = --trust-remote-code for the client tokenizer (preflight and client alike)
QUIESCE=${QUIESCE-"hermes hermes-webui owui-proxy"}
CLIENT_NAME=$UNIT-client
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
md5f(){ md5sum < "$1" | cut -c1-32; }

# ---------------- checks before the lock (nothing touched) ----------------
for f in "$LAUNCH27" "$D/launch-daily-v0280.sh" "$LIVE" "$D/daily-power.sh" "$D/shm-sweep.sh" "$D/lib/gpu-queue.sh" "$D/lib/gateway-drain.sh" \
         "$SHIM" "$PARSE" "$SUMMARY" "$SG" "$SB" "$MDIR/tokenizer.json" "$MDIR/tokenizer_config.json"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ "$(md5f "$LAUNCH27")" = "$LAUNCH27_MD5" ] || { log "ABORT: $LAUNCH27 md5 $(md5f "$LAUNCH27") != $LAUNCH27_MD5 (the 27B daily changed since this unit was written)"; exit 3; }
grep -q "^DAILY_IMG=$WANT_IMG " "$LAUNCH27" || { log "ABORT: launch-daily.sh DAILY_IMG is not $WANT_IMG"; exit 3; }
sudo docker image inspect "$WANT_IMG" >/dev/null 2>&1 || { log "ABORT: daily image $WANT_IMG not on flan"; exit 3; }
# snapshot the inputs: a probe edited while the unit is queued must not change what this run measures
cp "$SHIM" "$PARSE" "$SUMMARY" "$R/probes/"
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

# preflight (CPU only, --network none): the shim patches this image (manifest recorder included), the 27B tokenizer loads
# offline, the full Spec-Bench file loads through the shim's reader; report-only: Spec-Bench prompts that fill >= 1 / 2
# disk-tier blocks of 1,472 tokens once templated (the only prompts that could hit native-l2 across cells)
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
l2df(){ df -h --output=used,size,pcent "$D/native-l2" 2>/dev/null | tail -1 | tr -s ' '; }
running(){ [ "$(sudo docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
# stop every QUIESCE container that runs now; prints the ones it stopped (call as a plain statement or in $(...): no state)
quiesce(){ local c s=; for c in $QUIESCE; do running "$c" || continue; sudo docker stop -t 30 "$c" >/dev/null 2>&1 && s="$s $c"; done; echo "${s# }"; }
wait_vram_free(){ local i; for i in $(seq 36); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && return 0; sleep 5; done; return 1; }
teardown27(){ sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAME27" || return 0
  sudo docker rm -f "$NAME27" >/dev/null 2>&1; wait_vram_free || log "WARN: VRAM not free 180 s after removing $NAME27"
  sudo "$D/shm-sweep.sh" quiet >> "$R/shm-sweep.log" 2>&1 || true; }

finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during finish: ignored"' TERM INT HUP
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  if sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAME27"; then sudo docker logs "$NAME27" > "$C/container-final.log" 2>&1; fi
  teardown27
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
  log "native-l2 at the end: $(l2df)"
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
log "lock held; GLM at entry: $(glm_up && echo healthy || echo down); :8020 $(curl -sf -m 5 $U/health >/dev/null && echo answers || echo empty); direct clients stopped: ${st:-none}; passes $PASSES; datasets $DATASETS; concs $CONCS; SG_N $SG_N seed $SEED; SB_N $SB_N out $SB_OUT; POWER $POWER (want $WANT_PWR W); offsets now core $(gpcoff) / memory $(memoff), power $(pwrlim) W; GLM restore helper md5 $(md5f "$LIVE"); native-l2 $(l2df)"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
[ "$(md5f "$LAUNCH27")" = "$LAUNCH27_MD5" ] || void "under the lock: launch-daily.sh md5 $(md5f "$LAUNCH27") != $LAUNCH27_MD5"
cp "$LAUNCH27" "$R/launch-daily-at-lock.sh"
sudo docker logs glm53 > "$R/glm-final.log" 2>&1 || true
sudo docker logs glm53-preflight > "$R/glm-preflight-final.log" 2>&1 || true
sudo docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
wait_vram_free || log "WARN: VRAM not free 180 s after stopping the GLM daily"
log "GLM daily stopped (logs archived)"

printf 'tag\tpass\tdataset\tconc\tt_boot\tlauncher_md5\timage\timage_id\tenv_n\tenv_sha\tkeys_n\twindow\tslots\tcache\tpolicy\tpower_limit_w\tgpc_boot\tmem_boot\tgpc_after\tmem_after\tvram_free\tmem_line\tpwr_after\tpwr_line\tboot_tries\tvllm\n' > "$R/boots.tsv"
printf 'tag\tpass\tdataset\tconc\tserial_lo\tserial_hi\twarmups\tcatfile\trc\tcontainer\tn_req\tt0\tt1\tserver\n' > "$R/runs.tsv"
REF_CFG= REF_IMGID= REF_POOL=
# B_* = the current boot's record; written to boots.tsv once the cell is over (with the after-cell readings).
# keys_n / window / policy are TabbyAPI fields: '-' here. env_sha = sha256 of the container's Args + Env (the whole
# engine command line and environment), env_n = its Env count, cache = the pool in tokens, slots = SEQS.
bootrow(){ printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$B_TAG" "$B_PASS" "$B_DS" "$B_CONC" "$B_T" "$B_MD5" "$B_IMG" "$B_IMGID" "$B_ENVN" "$B_CFG" - - "$B_SLOTS" "$B_POOL" - \
  "$B_PWR" "$B_GPC" "$B_MEM" "${1:--}" "${2:--}" "$B_FREE" "$B_MEMLINE" "${3:--}" "$B_PWRLINE" "$B_TRIES" "$B_VER" >> "$R/boots.tsv"; }

boot27(){ local tag=$1 try bl ok=0 st
  B_TAG=$tag B_PASS=$2 B_DS=$3 B_CONC=$4
  for try in 1 2 3; do   # R233 / r675: the warmup flake ("CUDA error: invalid argument") is a lottery at this pin; draw again
    teardown27
    B_MD5=$(md5f "$LAUNCH27"); B_T=$(date -Is); bl=$C/boot-$tag-$try.log
    [ "$B_MD5" = "$LAUNCH27_MD5" ] || void "[$tag] launch-daily.sh md5 $B_MD5 != $LAUNCH27_MD5 (changed mid-run)"
    if env -i HOME="$HOME" USER="$(id -un)" PATH="$PATH" bash "$LAUNCH27" > "$bl" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then ok=1; break; fi
    sudo docker logs "$NAME27" > "$C/container-$tag-failed-$try.log" 2>&1
    FLAKES=$((FLAKES + 1))
    log "[$tag] boot attempt $try/3 FAILED: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,120}' "$bl" | tail -1)"
  done
  [ "$ok" = 1 ] || void "[$tag] NO BOOT after 3 attempts: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,120}' "$bl" | tail -1)"
  B_TRIES=$try
  st=$(quiesce); [ -n "$st" ] && log "[$tag] stopped again after the boot: $st (launch-daily-v0280.sh restarts owui-proxy)"
  [ "$POWER" = stock ] && bash $D/daily-power.sh stock 2>&1 | grep -aiE '^WARN' | tee -a "$R/audit.log"
  sudo docker inspect "$NAME27" > "$C/inspect-$tag.json" 2>/dev/null
  B_IMG=$(sudo docker inspect -f '{{.Config.Image}}' "$NAME27" 2>/dev/null)
  B_IMGID=$(sudo docker inspect -f '{{.Image}}' "$NAME27" 2>/dev/null | cut -c1-19)
  # Args in order + Env SORTED: docker inspect lists Env in a nondeterministic order across boots of one launcher
  # (first run 2026-09-28 15:52 VOIDed at A-sharegpt-c2 on an order-only difference, same 12 variables).
  B_CFG=$( { sudo docker inspect -f '{{json .Args}}' "$NAME27" 2>/dev/null; sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME27" 2>/dev/null | sort; } | sha256sum | cut -c1-12)
  B_ENVN=$(sudo docker inspect -f '{{len .Config.Env}}' "$NAME27" 2>/dev/null)
  B_POOL=$(grep -aoE 'Pool [0-9]+' "$bl" | tail -1 | tr -dc 0-9)
  B_SLOTS=$(grep -aoE 'SEQS [0-9]+' "$bl" | tail -1 | tr -dc 0-9)
  B_VER=$(grep -aoE '\(vllm [^,]+' "$bl" | tail -1 | cut -d' ' -f2)
  B_FREE=$(vram_free | tr -s ' ' '/' | sed 's#/$##')
  B_MEMLINE=$(grep -aoE 'clock offsets: .*' "$bl" | tail -1 | tr '\t' ' ')
  B_PWRLINE="launcher: $(grep -aoE 'power [0-9./]+ W' "$bl" | tail -1); unit: POWER=$POWER"
  B_PWR=$(pwrlim); B_GPC=$(gpcoff); B_MEM=$(memoff)
  log "[$tag] UP after $try attempt(s): image $B_IMG ($B_IMGID); launcher md5 $B_MD5; vllm $B_VER; engine config sha $B_CFG (env $B_ENVN); pool $B_POOL (R675 $R675_POOL); SEQS $B_SLOTS; power $B_PWR W ($B_PWRLINE); core offsets $B_GPC; memory offsets $B_MEM (${B_MEMLINE:-no clock-offset line}); VRAM free $B_FREE"
  [ "$(grep -ac '0.29 nvfp4 DAILY UP' "$bl")" -ge 1 ] || { bootrow; void "[$tag] the launcher did not report the DAILY path (no '0.29 nvfp4 DAILY UP' line)"; }
  [ "$B_IMG" = "$WANT_IMG" ] || { bootrow; void "[$tag] image '$B_IMG' != $WANT_IMG"; }
  [ -n "$B_POOL" ] || { bootrow; void "[$tag] no pool in the launcher output"; }
  [ "$B_PWR" = "$WANT_PWR" ] || { bootrow; void "[$tag] power limits at boot '$B_PWR' W, want '$WANT_PWR' W (POWER=$POWER)"; }
  if [ "$B_GPC" != "$WANT_GPC" ] || [ "$B_MEM" != "$WANT_MEM" ]; then bootrow
    void "[$tag] clock offsets at boot: core '$B_GPC' memory '$B_MEM', want core '$WANT_GPC' memory '$WANT_MEM'"; fi
  if [ -z "$REF_CFG" ]; then REF_CFG=$B_CFG REF_IMGID=$B_IMGID REF_POOL=$B_POOL
    [ "$B_POOL" = "$R675_POOL" ] || log "[$tag] NOTE: pool $B_POOL differs from R675's $R675_POOL (the launcher's band passed it)"
  elif [ "$B_CFG/$B_IMGID/$B_POOL" != "$REF_CFG/$REF_IMGID/$REF_POOL" ]; then bootrow
    void "[$tag] config changed mid-run: engine config / image id / pool $B_CFG/$B_IMGID/$B_POOL vs first boot $REF_CFG/$REF_IMGID/$REF_POOL"; fi; }

# conc concurrent non-stream chat requests of exactly WARM_TOK tokens on short prompts that are in neither dataset: they warm
# the kernels at this batch size and never touch a measured prompt's cache pages (each is far below one 1,472-token block)
warm(){ python3 - "$MODEL" "$2" "$1" "$WARM_TOK" <<'EOF'
import json, sys, threading, urllib.request
model, conc, tag, n = sys.argv[1], max(int(sys.argv[2]), 1), sys.argv[3], int(sys.argv[4])
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
got = []
def one(i):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": n, "min_tokens": n,
            "messages": [{"role": "user", "content": f"Warm-up {tag} slot {i}: describe a lighthouse at dusk in three sentences."}]}
    try:
        rq = urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
        got.append((json.load(op.open(rq, timeout=300)).get("usage") or {}).get("completion_tokens"))
    except Exception as e:
        print(f"warm-up error: {e.__class__.__name__}: {e}"[:200])
ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"{len(got)}/{conc} ok, completion tokens {got}")
EOF
}

snap(){ curl -s -m 20 "$U/metrics" > "$1" 2>/dev/null; }
# prom A B -> JSON of counter deltas B - A (summed over label sets; null when B lacks the metric) + B's queue gauges.
# Names = probes/mm_probe.py's, + request_success_total (requests) and e2e_request_latency_seconds_count (its fallback).
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

first=1
cell(){ local ps=$1 ds=$2 conc=$3 n=$4 rc t0 t1 tag=$1-$2-c$3 w0 m0 m1 wd sd q g fr ga ma pa; shift 4
  boot27 "$tag" "$ps" "$ds" "$conc"
  w0=$C/metrics-$tag-0-prewarm.prom; m0=$C/metrics-$tag-1-start.prom; m1=$C/metrics-$tag-2-end.prom
  snap "$w0"
  log "[$tag] warm-up: $(warm "$tag" "$conc" 2>&1 | tail -1)"
  sleep 2; snap "$m0"
  # metrics self-test: the warm-up must move the counters by exactly conc requests and conc x WARM_TOK tokens, spec on
  wd=$(prom "$w0" "$m0"); q=$(jget "$wd" req); g=$(jget "$wd" gen)
  log "[$tag] warm-up counters: requests $q, generation tokens $g, drafts $(jget "$wd" drafts), running/waiting now $(jget "$wd" running_end)/$(jget "$wd" waiting_end)"
  if [ "$q" != "$conc" ] || [ "$g" != "$((conc * WARM_TOK))" ] || [ "$(jget "$wd" pc_h)" = - ] || [ "$(jget "$wd" ext_h)" = - ] \
     || [ "$(jget "$wd" accepted)" = - ] || [ "$(jget "$wd" draft_tok)" = - ] \
     || [ "$(jget "$wd" running_end)" != 0 ] || [ "$(jget "$wd" waiting_end)" != 0 ]; then
    [ $first = 1 ] && void "[$tag] metrics self-test failed on the first cell (want requests $conc, generation tokens $((conc * WARM_TOK)), GPU and external (tier) prefix-cache and spec-decode counters present, nothing running): $wd"
    log "[$tag] WARN: metrics self-test failed: $wd"; fi
  t0=$(date -Is)
  timeout -k 30 "$CLIENT_TIMEOUT" sudo docker run --rm --name "$CLIENT_NAME" --network host "${CENV[@]}" \
    -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro -v "$R/results":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8020 --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer /model ${TRC_ARGS[@]+"${TRC_ARGS[@]}"} --temperature 0 \
    --request-rate inf --max-concurrency "$conc" --num-warmups 0 --num-prompts "$n" --no-oversample \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    --metadata "unit=$UNIT" "tag=$tag" "pass=$ps" "dataset=$ds" "conc=$conc" "num_warmups=0" \
      "unit_warmup=${conc}x non-stream out-of-set" "boot=fresh per cell" "thinking=template-default" \
      "length_forcing=min_tokens" "client=vllm-v0.30.0+vllm_bench_tabby" "server=vllm launch-daily.sh :8020" \
      "power=$POWER" \
    "$@" > "$C/client-$tag.log" 2>&1
  rc=$?; t1=$(date -Is); sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  sleep 3; snap "$m1"
  sd=$(prom "$m0" "$m1"); echo "$sd" > "$R/server/$tag.json"
  sudo docker logs "$NAME27" > "$C/container-$tag.log" 2>&1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ps" "$ds" "$conc" 0 0 0 - "$rc" - "$n" "$t0" "$t1" \
    "server/$tag.json" >> "$R/runs.tsv"
  ga=$(gpcoff); ma=$(memoff); pa=$(pwrlim); bootrow "$ga" "$ma" "$pa"
  q=$(jget "$sd" req); fr=$([ "$q" = - ] && echo "?" || echo $((q - n)))   # every client request finishes (rc 0) or fails; extra = foreign
  printf '%s\t%s\n' "$tag" "$fr" >> "$R/foreign.tsv"
  [ "$fr" = 0 ] || log "[$tag] WARN: the server finished $q requests for $n client requests (foreign traffic, or failures)"
  log "[$tag] rc $rc; $(grep -c '^shim: patched' "$C/client-$tag.log") shim line; $(grep -aoE '^shim: [0-9]+ samples' "$C/client-$tag.log" | cut -d' ' -f2) sampled; server: requests $q/$n, generation tokens $(jget "$sd" gen), prompt tokens $(jget "$sd" prompt), cached $(jget "$sd" pc_h) GPU + $(jget "$sd" ext_h) tier, accepted $(jget "$sd" accepted) of $(jget "$sd" draft_tok) drafted, preemptions $(jget "$sd" preempt), running/waiting at end $(jget "$sd" running_end)/$(jget "$sd" waiting_end); $(grep -aE '^(Successful requests|Failed requests|Benchmark duration|Output token throughput|Median TPOT)' "$C/client-$tag.log" | sed -E 's/ {2,}/ /g' | paste -sd';' -); engine ERROR lines $(grep -acE ' ERROR |^ERROR' "$C/container-$tag.log") tracebacks $(grep -ac Traceback "$C/container-$tag.log") OOM $(grep -acE 'OutOfMemoryError|out of memory' "$C/container-$tag.log"); offsets after core $ga memory $ma, power $pa W"
  # the shim's main() path first runs here: a client that produced nothing must not burn 27 more boots
  if [ $first = 1 ]; then first=0
    [ -s "$R/results/$tag.json" ] && [ -s "$R/results/$tag.samples.tsv" ] && grep -q '^shim: patched' "$C/client-$tag.log" \
      || void "first client run produced no result/manifest: $(grep -avE '^\s*$' "$C/client-$tag.log" | tail -2 | cut -c1-200)"; fi
  [ "$ga" = "$WANT_GPC" ] && [ "$ma" = "$WANT_MEM" ] && [ "$pa" = "$WANT_PWR" ] || void "[$tag] clocks / power after the cell: core '$ga' memory '$ma' power '$pa' W"; }

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
DECISION=${d:-"VOID the summary printed no decision: $(tail -1 "$R/summary.txt" | cut -c1-200)"}
FOREIGN=$(awk -F'\t' '$2 != "0" {printf "%s%s=%s", (n++ ? ", " : ""), $1, $2}' "$R/foreign.tsv" 2>/dev/null)
if [ -n "$FOREIGN" ]; then log "foreign traffic: $FOREIGN"
  case "$DECISION" in PUBLISHABLE*) DECISION="NOT-PUBLISHABLE foreign (non-benchmark) requests in cells: $FOREIGN";; esac; fi
finish DONE
