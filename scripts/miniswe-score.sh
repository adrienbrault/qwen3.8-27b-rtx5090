#!/usr/bin/env bash
# Official SWE-Bench-Verified scoring of a mini-SWE-agent batch run. CPU+docker only — safe beside a
# serving engine. usage: miniswe-score.sh RESULTS_DIR [RUN_ID]   (reads RESULTS_DIR/out/preds.json)
# - model_name_or_path is normalized to a slash-free name (the harness builds log paths from it).
# - Runs in the OFFICIAL swebench/sweb.eval images — the same ones the agent ran in, so scoring right
#   after a chunk needs no second Docker Hub pull.
# - Cumulative + rerunnable: the harness skips instances already evaluated under RUN_ID (per-instance
#   report.json under logs/run_evaluation/RUN_ID/MODEL/) and regenerates the summary from the
#   predictions passed — so the final call must pass the FULL preds (it does: preds.json is cumulative).
#   To re-evaluate an instance, delete its logs/run_evaluation/RUN_ID/MODEL/<iid> dir first.
# - --cache_level instance (2026-09-03, R166 campaign): with `env` the harness deletes every sweb.eval image that
#   was not present when it started (clean_images/should_remove) — and chunk scoring runs in the background while
#   the NEXT chunk pre-pulls, so the next chunk's images vanished right before it ran; mini-swe then pulled them
#   implicitly from Docker Hub, burned the anonymous quota (toomanyrequests), and 16+ instances errored in scoring
#   and 58 in the agent phase. The campaign's own prune (per chunk, after scoring) is what bounds disk use.
set -uo pipefail
R=${1:?results dir}; RUNID=${2:-miniswe}
SV=/srv/qwen5090/swebench-eval; MODEL=qwen3.8-27b-miniswe
[ -x "$SV/.venv/bin/python" ] || { echo "scorer venv missing — run deepswe-score.sh once to bootstrap"; exit 1; }
python3 - "$R" "$MODEL" <<'PYEOF'
import json,sys
R,model=sys.argv[1],sys.argv[2]
d=json.load(open(R+'/out/preds.json'))
n=e=0
with open(R+'/preds.jsonl','w') as f:
    for iid,p in d.items():
        n+=1; patch=p.get('model_patch') or ''
        if not patch: e+=1
        f.write(json.dumps({'instance_id':iid,'model_name_or_path':model,'model_patch':patch})+'\n')
print(f"predictions: {n} ({e} empty)")
PYEOF
cd "$R" || exit 1
"$SV/.venv/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path "$R/preds.jsonl" --max_workers "${SCORE_WORKERS:-6}" --run_id "$RUNID" --namespace swebench \
  --cache_level instance --timeout 1800 >> "$R/score.log" 2>&1 || { echo "swebench harness non-zero — tail score.log:"; tail -8 "$R/score.log"; }
python3 - "$R" "$MODEL" "$RUNID" <<'PYEOF'
import json,glob,sys
R,model,runid=sys.argv[1:4]
reps=glob.glob(f"{R}/{model}.{runid}.json")
if not reps: print("### REPORT MISSING ###"); sys.exit(1)
r=json.load(open(reps[0]))
g=lambda k: r.get(k)
print(f"### OFFICIAL cumulative: resolved {g('resolved_instances')}/{g('submitted_instances')} submitted (of {g('total_instances')}); completed {g('completed_instances')}, errors {g('error_instances')}, empty {g('empty_patch_instances')} ###")
if g('error_instances'): print("error_ids:", " ".join(r.get('error_ids',[])[:40]))
PYEOF
