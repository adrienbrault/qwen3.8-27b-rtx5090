#!/usr/bin/env bash
# Re-score the SWE-bench instances a miniswe-full.sh campaign reported as ERRORS (2026-09-03, R166 campaign).
# Why: chunk scoring runs in the background while the next chunk's images are pre-pulled; the swebench harness
# (`--cache_level env`) removes instance images it did not see at its start, so images of the following chunk can
# vanish before that chunk's own scoring, which then pulls from Docker Hub — anonymous quota exhausted
# ("toomanyrequests") → "Error occurred while pulling image" for 16 instances of chunk 280 (all had Submitted
# trajectories). The FINAL pass re-evaluates only instances without a report and would hit the same wall.
# This unit waits for the listed GPU units to finish (scoring loads the CPUs; decode measurements first), then:
# error ids from the harness report (+ any prediction without a report) → images pulled through mirror.gcr.io and
# tagged with their docker.io names → per-instance eval dirs deleted → miniswe-score.sh (cumulative, rerunnable).
# Up to 3 passes while errors remain. CPU + docker only; never touches an engine.
#   usage: miniswe-rescore-errors.sh RESULTS_DIR [RUN_ID]   env: WAIT_UNITS="u1 u2" SCORE_WORKERS=3 MIRROR=mirror.gcr.io
#   unit:  sudo systemd-run --unit=miniswe-rescore --collect -p User=adrienbrault -p RuntimeMaxSec=28800 bash /srv/qwen5090/miniswe-rescore-errors.sh /srv/qwen5090/results/2026-09-02-miniswe-rh-nvfp4-r166
set -uo pipefail
R=${1:?results dir}; RUNID=${2:-miniswe}; MODEL=qwen3.8-27b-miniswe; MIRROR=${MIRROR:-mirror.gcr.io}
WAIT_UNITS=${WAIT_UNITS:-"miniswe-nvfp4-r166 r166c-tier r167-embed"}
log(){ echo "$(date -Is) [rescore] $*" | tee -a "$R/audit.log"; }
img(){ echo "swebench/sweb.eval.x86_64.${1//__/_1776_}:latest" | tr 'A-Z' 'a-z'; }
for u in $WAIT_UNITS; do while systemctl is-active --quiet "$u"; do sleep 60; done; done
log "start (GPU units done: $WAIT_UNITS)"
error_ids(){ python3 - "$R" "$MODEL" "$RUNID" <<'PYEOF'
import json,os,sys
R,model,runid=sys.argv[1:4]
ids=set()
rep=f"{R}/{model}.{runid}.json"
if os.path.exists(rep): ids.update(json.load(open(rep)).get('error_ids',[]))
preds=json.load(open(f"{R}/out/preds.json"))
for iid,p in preds.items():
    if p.get('model_patch') and not os.path.exists(f"{R}/logs/run_evaluation/{runid}/{model}/{iid}/report.json"): ids.add(iid)
print(" ".join(sorted(ids)))
PYEOF
}
for pass in 1 2 3; do
  IDS=$(error_ids); [ -n "$IDS" ] || { log "no error instances left"; break; }
  log "pass $pass: $(echo $IDS | wc -w) instances: $(echo $IDS | cut -c1-400)"
  miss=0
  for iid in $IDS; do
    im=$(img "$iid"); sudo docker image inspect "docker.io/$im" >/dev/null 2>&1 && continue
    if sudo docker pull -q "$MIRROR/$im" >/dev/null 2>"$R/rescore-pull.err"; then sudo docker tag "$MIRROR/$im" "docker.io/$im" && sudo docker rmi "$MIRROR/$im" >/dev/null 2>&1
    else miss=$((miss+1)); log "mirror pull FAILED for $iid: $(head -c 160 "$R/rescore-pull.err" | tr '\n' ' ')"; fi
  done
  log "images ready ($miss mirror misses); disk free $(df -h / | awk 'NR==2{print $4}')"
  for iid in $IDS; do rm -rf "$R/logs/run_evaluation/$RUNID/$MODEL/$iid"; done
  SCORE_WORKERS=${SCORE_WORKERS:-3} bash /srv/qwen5090/miniswe-score.sh "$R" "$RUNID" 2>&1 | grep -aE "predictions|OFFICIAL|error_ids|non-zero" | sed "s/^/[score RESCORE-$pass] /" | tee -a "$R/audit.log"
  for iid in $IDS; do sudo docker rmi -f "docker.io/$(img "$iid")" >/dev/null 2>&1; done
done
log "done: $(grep -a "OFFICIAL cumulative" "$R/audit.log" | tail -1 | cut -c1-160)"
