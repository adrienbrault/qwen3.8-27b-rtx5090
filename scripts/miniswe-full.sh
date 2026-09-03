#!/usr/bin/env bash
# SWE-Bench Verified (500) with mini-SWE-agent (pinned below), builtin benchmarks/swebench.yaml = the
# leaderboard's "bash-only" setting (one bash tool via tool calls, step_limit 250) + miniswe/qwen38-local.yaml
# (serving plumbing only), on the DAILY stack via boot-daily-miniswe.sh (:8030, SEQS=16) at W=16 workers
# (R159: c16 is the last cell before the admission floor). Leaderboard-SHAPED, not a submission: the
# official swebench harness runs locally (miniswe-score.sh), in the same official images the agent ran in.
# Flow per chunk of CHUNK instances (dataset order, --slice):  pre-pull images (next chunk pre-pulls in the
# background while this one runs) -> agents -> score in the background (previous chunk's scoring is waited
# on and its images pruned first) -> tier check (>=80% => cycle the eval engine and wipe; engines down =
# race-free). Then a retry pass for INFRA exits (mini-swe records errored instances in preds.json with an
# empty patch, so a plain resume would skip them), the final cumulative score, and the daily restore.
# Images come through mirror.gcr.io (Google's Hub pull-through cache, own quota) tagged with their docker.io
# names; Docker Hub (anonymous ~100 manifest requests / 1 h / IP, probed 2026-09-02) is only the fallback.
# Scoring uses --cache_level instance (miniswe-score.sh): the harness must NOT delete images it did not see at its start,
# or the next chunk (pre-pulled in parallel) loses its images and re-pulls from Hub (2026-09-03 R166: 58 agent-phase
# CalledProcessError + 16 scoring errors before the fix). # A running campaign can be fed ahead by miniswe-prepull-mirror.sh (separate unit). Disk: root holds <=3 chunks of ~4G images (pre-pull waits for >=200G free).
# Resumable: rerun == resume (preds.json). Smoke gate (slice 0:1, w1) runs first on a fresh output dir.
#   usage: miniswe-full.sh [K]   (K of 500, default 500; env CHUNK=40 W=16 START=0 RUN_SUFFIX= SKIP_SMOKE=0
#          BOOT=<engine boot script, default boot-daily-miniswe.sh> GATE=<script run "pre"/"post" with $R, e.g. probes/needle_gate.sh>)
#   unit:  sudo systemd-run --unit=miniswe --collect -p User=adrienbrault -p RuntimeMaxSec=100000 bash /srv/qwen5090/miniswe-full.sh 500
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TERM=dumb
K=${1:-500}; CHUNK=${CHUNK:-40}; W=${W:-12}; START=${START:-0}
# GPU-exclusive lock (2026-09-03): two units that both tear down every vllm-* container must never overlap — the
# R163 battery and this runner once killed each other's engines. Held until this process exits (after finish()).
source /srv/qwen5090/lib/gpu-queue.sh   # OPERATIONS §12: registered here so the unit ahead skips its daily restore
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { echo "$(date -Is) waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }   # START: first chunk index on a resume (skips re-pulling pruned chunks)
R=/srv/qwen5090/results/2026-09-02-miniswe-rh${RUN_SUFFIX:-}; OUT=$R/out; mkdir -p "$OUT"
VENV=/srv/qwen5090/miniswe/.venv; MSWA=2.4.6
CFG=/srv/qwen5090/miniswe/qwen38-local.yaml
EVAL_L2=${EVAL_L2:-/srv/qwen5090/eval-l2}
U=http://127.0.0.1:8030
RUNID=miniswe; SMODEL=qwen3.8-27b-miniswe
DS=princeton-nlp/SWE-bench_Verified
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
settle(){ # $1 = idle settle seconds after the GPUs report free (default 60; 300 after a failed TP2 boot —
  # 2026-09-02: three back-to-back TP2 boots with a 45 s settle right after a kernel-panic reboot all died in TP1
  # warmup ("CUDA error: invalid argument", TP0 then hangs in shm_broadcast); a 300 s settle booted first try.
  # R162 2026-09-03: on a healthy box 60/120/180 s all booted first try, so the long settle is the retry path only)
  timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
finish(){ # $1 = status line; always: kill agents, capture engine log, restore daily
  pkill -f "[m]ini-extra swebench" 2>/dev/null; sleep 3
  sudo docker logs vllm-eval > "$R/engine.log" 2>&1 || true
  grep -aciE "error|traceback" "$R/engine.log" | sed "s/^/[engine] error-lines: /" | tee -a "$R/audit.log"
  settle
  bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  log "=== MINISWE $1 ==="; }
trap 'log "### SIGTERM (unit RuntimeMaxSec?) — restoring daily ###"; finish "ABORTED"; exit 4' TERM
# WIPE_L2=1 (R166): cold-tier start for a fair pairing (the fp8 run R160 started at a 3% tier). Under the lock, so no
# eval engine can be using eval-l2; the daily's tier is a different mount.
if [ "${WIPE_L2:-0}" = 1 ]; then
  sudo docker ps --format '{{.Names}}' | grep -qxE 'vllm-eval|vllm-exp' && { log "WIPE_L2: an eval engine is still up under the lock?! aborting"; exit 1; }
  sudo find "$EVAL_L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
  log "eval-l2 namespace sets wiped for a cold-tier start: now $(df --output=pcent "$EVAL_L2" | tail -1 | tr -dc 0-9)%"
fi

# ---- bootstrap (idempotent) ----
if [ ! -x "$VENV/bin/mini-extra" ]; then
  mkdir -p "$(dirname "$VENV")"; rm -rf "$VENV"
  python3 -m venv "$VENV" && "$VENV/bin/pip" -q install --upgrade pip && "$VENV/bin/pip" -q install "mini-swe-agent==$MSWA" datasets \
    || { log "venv bootstrap FAILED"; exit 1; }
fi
. "$VENV/bin/activate"
BUILTIN=$(python3 -c "from minisweagent.config import builtin_config_dir; print(builtin_config_dir/'benchmarks'/'swebench.yaml')" 2>/dev/null | tail -1)   # import prints banner lines first
[ -f "$BUILTIN" ] || { log "builtin swebench.yaml not found at $BUILTIN"; exit 1; }
log "=== MINISWE start K=$K chunk=$CHUNK w=$W | mini-swe-agent $(pip show mini-swe-agent 2>/dev/null | awk '/^Version/{print $2}') litellm $(pip show litellm 2>/dev/null | awk '/^Version/{print $2}') swebench $(/srv/qwen5090/swebench-eval/.venv/bin/pip show swebench 2>/dev/null | awk '/^Version/{print $2}') | cfg $BUILTIN + $CFG ==="
cp "$BUILTIN" "$R/swebench.yaml.used"; cp "$CFG" "$R/qwen38-local.yaml.used"
[ -s "$R/instances.txt" ] || python3 - "$DS" > "$R/instances.txt" <<'PYEOF'
import sys; from datasets import load_dataset
for x in load_dataset(sys.argv[1], split="test"): print(x["instance_id"])
PYEOF
N=$(wc -l < "$R/instances.txt"); [ "$N" -ge 500 ] || { log "dataset load FAILED ($N ids)"; exit 1; }
((K>N)) && K=$N
ids(){ sed -n "$(($1+1)),${2}p" "$R/instances.txt"; }   # $1=start (0-based) $2=end (exclusive)
img(){ echo "docker.io/swebench/sweb.eval.x86_64.${1//__/_1776_}:latest" | tr 'A-Z' 'a-z'; }
pull_mirror(){ # $1=docker.io image name — pull through mirror.gcr.io and tag with the docker.io name (2026-09-03: the retry pass went to Hub directly)
  local im=$1 m="${MIRROR:-mirror.gcr.io}/${1#docker.io/}"
  sudo docker pull -q "$m" >/dev/null 2>"$R/pull-retry.err" && sudo docker tag "$m" "$im" && { sudo docker rmi "$m" >/dev/null 2>&1; return 0; }; return 1; }
prepull(){ # $1=start $2=end — pull missing images; 429 => sleep 10 min; waits for >=200G free on /
  local s=$1 e=$2 iid im tries err="$R/pull-$1.err"
  for iid in $(ids "$s" "$e"); do
    im=$(img "$iid"); sudo docker image inspect "$im" >/dev/null 2>&1 && continue
    while [ "$(df -k --output=avail / | tail -1 | tr -dc 0-9)" -lt 209715200 ]; do echo "$(date -Is) [prepull $s] <200G free on / — waiting for a prune" >> "$R/audit.log"; sleep 300; done
    # mirror first (Google's Docker Hub pull-through cache: no per-IP Hub quota, 4 GB in ~9 s; 2026-09-02),
    # tagged with the docker.io name so mini-swe / the harness find it; Docker Hub is the fallback.
    if sudo docker pull -q "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" >/dev/null 2>"$err"; then
      sudo docker tag "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" "$im" && sudo docker rmi "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" >/dev/null 2>&1; continue
    fi
    echo "$(date -Is) [prepull $s] mirror miss on $iid: $(head -c 120 "$err" | tr '\n' ' ') — falling back to Docker Hub" >> "$R/audit.log"
    tries=0
    until sudo docker pull -q "$im" >/dev/null 2>"$err"; do
      tries=$((tries+1))
      if grep -qiE "toomanyrequests|rate ?limit|429" "$err"; then echo "$(date -Is) [prepull $s] 429 on $iid — sleeping 10 min" >> "$R/audit.log"; sleep 600
      else echo "$(date -Is) [prepull $s] pull error on $iid: $(head -c 160 "$err" | tr '\n' ' ')" >> "$R/audit.log"; sleep 60; fi
      [ $tries -ge 15 ] && { echo "$(date -Is) [prepull $s] GIVING UP on $iid" >> "$R/audit.log"; break; }
    done
  done
  echo "$(date -Is) [prepull $s] done ($(ids "$s" "$e" | wc -l) ids) disk free $(df -h / | awk 'NR==2{print $4}')" >> "$R/audit.log"
}
prune(){ # $1=start $2=end — leftover agent containers + this chunk's images (ONLY between chunks, after scoring)
  sudo docker ps -aq --filter name=minisweagent- | xargs -r sudo docker rm -f >/dev/null 2>&1
  for iid in $(ids "$1" "$2"); do sudo docker rmi -f "$(img "$iid")" >/dev/null 2>&1; done
  log "[prune $1] disk free $(df -h / | awk 'NR==2{print $4}')"
}
tier_pct(){ df --output=pcent "$EVAL_L2" | tail -1 | tr -dc 0-9; }
boot(){ local att; for att in 1 2; do
    if bash "${BOOT:-/srv/qwen5090/boot-daily-miniswe.sh}" > "$R/boot-$(date +%H%M%S).log" 2>&1; then
      grep -ahE "KV pool|max_num_seqs|EVAL UP" "$R"/boot-*.log | tail -3 | sed "s/^/[boot] /" | tee -a "$R/audit.log"; return 0; fi
    sudo docker logs vllm-eval > "$R/boot-fail-$att-$(date +%H%M%S).log" 2>&1 || true   # capture BEFORE the rm (lost the 20:15 log once)
    log "[boot] attempt $att failed: $(grep -ahE 'FAILED|Error' "$R"/boot-*.log | tail -1 | cut -c1-150)"; settle 300
  done; return 1; }
watchdog(){ local fails=0 low=0 ma; while true; do sleep 60
    if curl -sf -m 10 $U/health >/dev/null; then fails=0; else fails=$((fails+1)); fi
    ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$ma" -lt 4 ]; then low=$((low+1)); else low=0; fi
    if [ $low -ge 2 ]; then log "### MemAvailable ${ma}G twice — killing the scorer (rerunnable) to protect the host (2026-09-02 OOM-panic) ###"; pkill -f "[s]webench.harness.run_evaluation"; sudo docker ps -q --filter name=sweb.eval | xargs -r sudo docker rm -f >/dev/null 2>&1; low=0; fi
    if [ $fails -ge 3 ]; then log "### ENGINE DEAD (health failed 3x) — killing agents ###"; touch "$R/ENGINE_DEAD"; pkill -f "[m]ini-extra swebench"; return; fi
  done; }
run_slice(){ # $1=slice $2=workers [$3...=extra args]; returns mini-extra's rc
  local sl=$1 w=$2; shift 2
  timeout "${CHUNK_TIMEOUT:-14400}" mini-extra swebench --subset verified --split test --slice "$sl" -w "$w" -o "$OUT" \
    -c "$BUILTIN" -c "$CFG" "$@" >> "$R/run.log" 2>&1; return $?
}
progress(){ python3 - "$OUT" <<'PYEOF'
import json,glob,os,sys,collections
OUT=sys.argv[1]; p=json.load(open(OUT+'/preds.json')) if os.path.exists(OUT+'/preds.json') else {}
st=collections.Counter()
for f in glob.glob(OUT+'/*/*.traj.json'):
    try: st[json.load(open(f))['info'].get('exit_status')]+=1
    except Exception: st['<unreadable>']+=1
print(f"### PROGRESS preds={len(p)} nonempty={sum(1 for v in p.values() if v.get('model_patch'))} exits={dict(st)} ###")
PYEOF
}
metrics(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:(spec_decode_num_(accepted|draft)_tokens_total|num_preemptions_total|prefix_cache_(hits|queries)_total|kv_offload)" | sed "s/^/[$1] /" | cut -c1-160 | tee -a "$R/audit.log"; }

# ---- engine ----
log "stopping the daily (GPU-exclusive campaign); tier $EVAL_L2 at $(tier_pct)% ; disk free $(df -h / | awk 'NR==2{print $4}')"
settle
boot || { log "BOOT FAILED twice"; finish "BOOT-FAILED"; exit 1; }
if [ -n "${GATE:-}" ]; then bash "$GATE" pre "$R" || { log "### PRE-GATE FAILED ($GATE) — not running the campaign on a broken engine ###"; finish "GATE-FAILED"; exit 3; }; fi
watchdog & WD=$!

# ---- smoke gate: one instance, one worker (skipped when the output dir already has preds) ----
E0=$((CHUNK<K?CHUNK:K))
if [ ! -s "$OUT/preds.json" ] && [ "${SKIP_SMOKE:-0}" != 1 ]; then
  prepull 0 1
  log "[smoke] slice 0:1 w1 start"; run_slice 0:1 1; log "[smoke] rc=$?"
  if ! python3 - "$OUT" <<'PYEOF'
import json,glob,sys
fs=glob.glob(sys.argv[1]+'/*/*.traj.json')
if not fs: print("### SMOKE FAIL: no traj.json ###"); sys.exit(1)
t=json.load(open(fs[0])); info=t.get('info',{}); st=info.get('exit_status'); msgs=t.get('messages',[])
asst=[m for m in msgs if m.get('role')=='assistant']; tc=sum(1 for m in asst if m.get('tool_calls'))
u=(asst[0].get('extra',{}).get('response',{}).get('usage') or {}) if asst else {}
print(f"### SMOKE exit_status={st} steps={len(asst)} with_tool_calls={tc} step0_prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} patch_len={len(info.get('submission') or '')} exc={str(info.get('exception_str',''))[:200]} ###")
ok = tc>0 and st in ('Submitted','LimitsExceeded') and (u.get('prompt_tokens') or 0)>1500
sys.exit(0 if ok else 1)
PYEOF
  then log "### SMOKE GATE FAILED — see $OUT/*/*.traj.json and run.log ###"; tail -30 "$R/run.log" | cut -c1-200 >> "$R/audit.log"; kill $WD 2>/dev/null; finish "SMOKE-FAILED"; exit 2; fi
  log "[smoke] gate passed"
fi

# ---- chunks ----
E0=$((START+CHUNK)); ((E0>K)) && E0=$K
prepull "$START" "$E0"; PP=""; SC=""; PREV_S=-1; PREV_E=-1
for ((S=START; S<K; S+=CHUNK)); do
  E=$((S+CHUNK)); ((E>K)) && E=$K
  [ -n "$PP" ] && { wait "$PP"; PP=""; }                       # this chunk's images must be local
  NS=$E; NE=$((E+CHUNK)); ((NE>K)) && NE=$K
  ((NS<K)) && { prepull "$NS" "$NE" & PP=$!; }                  # next chunk pulls while this one runs
  log "### CHUNK $S:$E w=$W start | tier $(tier_pct)% | load $(cut -d' ' -f1 /proc/loadavg) | free $(free -g | awk '/Mem/{print $7}')G ###"
  run_slice "$S:$E" "$W"; RC=$?
  [ -f "$R/ENGINE_DEAD" ] && { log "### ABORT: engine died during chunk $S (resume later: rerun this unit) ###"; [ -n "$PP" ] && kill "$PP" 2>/dev/null; [ -n "$SC" ] && wait "$SC"; finish "ENGINE-DEAD"; exit 3; }
  ((RC==124)) && log "### CHUNK $S:$E TIMED OUT (${CHUNK_TIMEOUT:-14400}s) — unfinished instances resume on rerun ###"
  progress | tee -a "$R/audit.log"; metrics "after-chunk-$S"
  [ -n "$SC" ] && { wait "$SC"; SC=""; prune "$PREV_S" "$PREV_E"; }   # previous chunk scored -> its images go
  ( SCORE_WORKERS=${SCORE_WORKERS:-3} bash /srv/qwen5090/miniswe-score.sh "$R" "$RUNID" 2>&1 | grep -aE "predictions|OFFICIAL|error_ids|non-zero" | sed "s/^/[score $S] /" >> "$R/audit.log" ) & SC=$!
  PREV_S=$S; PREV_E=$E
  if [ "$(tier_pct)" -ge 80 ] && ((E<K)); then
    log "tier at $(tier_pct)% — cycling the eval engine to wipe it (engine down = race-free)"
    kill $WD 2>/dev/null; settle
    sudo find "$EVAL_L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
    boot || { log "REBOOT FAILED twice"; [ -n "$PP" ] && kill "$PP" 2>/dev/null; wait "$SC"; finish "REBOOT-FAILED"; exit 1; }
    watchdog & WD=$!
  fi
done
[ -n "$SC" ] && { wait "$SC"; SC=""; }

# ---- retry pass: infra exits only (pull/exec failures, connection/timeouts) ----
RETRY=$(python3 - "$OUT" <<'PYEOF'
import json,glob,os,sys
OUT=sys.argv[1]; p=json.load(open(OUT+'/preds.json'))
INFRA={'CalledProcessError','TimeoutExpired','APIConnectionError','ServiceUnavailableError','InternalServerError','APIError','Timeout','ConnectionError','OSError','RuntimeError','ReadTimeout','APITimeoutError'}
ids=[]
for iid,v in p.items():
    if v.get('model_patch'): continue
    f=f"{OUT}/{iid}/{iid}.traj.json"
    if not os.path.exists(f): ids.append(iid); continue
    st=json.load(open(f))['info'].get('exit_status') or ''
    if st in INFRA or st.endswith('ConnectionError') or st.endswith('TimeoutError'): ids.append(iid)
print(" ".join(ids))
PYEOF
)
if [ -n "$RETRY" ]; then
  NR=$(echo "$RETRY" | wc -w); log "### RETRY pass: $NR infra-exit instances: $(echo "$RETRY" | cut -c1-300) ###"
  for iid in $RETRY; do im=$(img "$iid"); t=0; sudo docker image inspect "$im" >/dev/null 2>&1 || { until pull_mirror "$im" || sudo docker pull -q "$im" >/dev/null 2>"$R/pull-retry.err"; do t=$((t+1)); echo "$(date -Is) [retry-pull] $iid: $(head -c 120 "$R/pull-retry.err")" >> "$R/audit.log"; [ $t -ge 6 ] && break; sleep 600; done; }; done
  FILT="^($(echo "$RETRY" | tr ' ' '|'))$"
  [ -f "$R/ENGINE_DEAD" ] || run_slice "0:$K" "$W" --filter "$FILT" --redo-existing; log "[retry] rc=$?"
  for iid in $RETRY; do rm -rf "$R/logs/run_evaluation/$RUNID/$SMODEL/$iid"; done   # force re-evaluation
  progress | tee -a "$R/audit.log"
fi

# ---- final cumulative score + handoff ----
bash /srv/qwen5090/miniswe-score.sh "$R" "$RUNID" 2>&1 | grep -aE "predictions|OFFICIAL|error_ids|non-zero" | sed "s/^/[score FINAL] /" | tee -a "$R/audit.log"
progress | tee -a "$R/audit.log"; metrics final
if [ -n "${GATE:-}" ]; then bash "$GATE" post "$R" || log "### POST-GATE FAILED ($GATE) — retrieval degraded during the run; score is NOT attributable to KV quality ###"; fi
HIT=$(sudo docker logs vllm-eval 2>&1 | grep -aoE "External prefix cache hit rate: [0-9.]+%" | tail -1)
log "### GPU PHASE DONE K=$K | ${HIT:-no-hit-line} | tier $(tier_pct)% ($(sudo du -sh "$EVAL_L2" 2>/dev/null | cut -f1)) ###"
kill $WD 2>/dev/null
sudo docker ps -aq --filter name=minisweagent- | xargs -r sudo docker rm -f >/dev/null 2>&1
finish "DONE"
[ -n "$PREV_S" ] && ((PREV_S>=0)) && prune "$PREV_S" "$PREV_E"
