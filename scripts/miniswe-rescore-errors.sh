#!/usr/bin/env bash
# Re-score the error_instances of a finished mini-SWE-agent campaign (2026-09-03: 23 sympy instances errored in
# scoring on transient Docker Hub 500s at image pull; the harness re-runs any instance without a report.json).
# Pulls each errored instance's image through mirror.gcr.io (tagged with its docker.io name), then reruns
# miniswe-score.sh with few workers (CPU+docker only: safe beside a running engine/campaign). Up to 3 passes.
#   usage: miniswe-rescore-errors.sh RESULTS_DIR [RUN_ID]      env: SCORE_WORKERS=2 MIRROR=mirror.gcr.io
set -uo pipefail
R=${1:?results dir}; RUNID=${2:-miniswe}; MODEL=qwen3.8-27b-miniswe
log(){ echo "$(date -Is) [rescore] $*" | tee -a "$R/audit.log"; }
img(){ echo "docker.io/swebench/sweb.eval.x86_64.${1//__/_1776_}:latest" | tr 'A-Z' 'a-z'; }
PULLED=""
for pass in 1 2 3; do
  ids=$(python3 -c "import json,sys;r=json.load(open('$R/$MODEL.$RUNID.json'));print(' '.join(r.get('error_ids',[])))")
  [ -z "$ids" ] && { log "no error instances left"; break; }
  log "pass $pass: $(echo "$ids" | wc -w) error instances: $(echo "$ids" | cut -c1-200)"
  for iid in $ids; do
    im=$(img "$iid"); sudo docker image inspect "$im" >/dev/null 2>&1 && continue
    if sudo docker pull -q "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" >/dev/null 2>"$R/rescore-pull.err"; then
      sudo docker tag "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" "$im" && sudo docker rmi "${MIRROR:-mirror.gcr.io}/${im#docker.io/}" >/dev/null 2>&1; PULLED="$PULLED $im"
    else log "mirror miss on $iid: $(head -c 120 "$R/rescore-pull.err" | tr '\n' ' ')"; fi
    rm -rf "$R/logs/run_evaluation/$RUNID/$MODEL/$iid"
  done
  SCORE_WORKERS=${SCORE_WORKERS:-2} bash /srv/qwen5090/miniswe-score.sh "$R" "$RUNID" 2>&1 | grep -aE "predictions|OFFICIAL|error_ids|non-zero" | sed "s/^/[score RESCORE$pass] /" | tee -a "$R/audit.log"
done
for im in $PULLED; do sudo docker rmi -f "$im" >/dev/null 2>&1; done   # images pulled here go away again (~4 GB each)
log "done: $(grep -a "OFFICIAL cumulative" "$R/audit.log" | tail -1 | cut -c1-160)"
