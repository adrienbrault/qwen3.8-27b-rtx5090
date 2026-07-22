#!/usr/bin/env bash
# Sanitize DeepSWE predictions and rescore the instances the artifact broke.
#
# WHY (2026-07-21): R2E-Gym's task images ship locally-modified build/config
# files (sphinx: tox.ini, astropy: pyproject.toml — their test-runner wiring).
# The end-of-session `git diff` exports those pre-existing deltas INSIDE every
# patch. On the official images those hunks don't match, `git apply` rejects the
# WHOLE patch, and swebench's GNU-patch fallback reverse-applies it ("Assuming
# -R") and wrecks the tree -> every test fails. 63/500 instances (ALL 43 scored
# sphinx + 15 astropy + 4 mpl + 1 psf) were killed mechanically this way.
#
# Sanitizer (uniform over all 500, not cherry-picked): drop per-file diffs for
# ROOT-level build/config files {tox.ini, pyproject.toml, setup.cfg, setup.py}
# IFF the patch also touches at least one other file. These files are never the
# fix in these repos and the official eval.sh ignores them entirely.
# Instances whose sanitized patch differs AND were not already resolved get
# re-evaluated under run_id "deepswefix"; verdicts merge (fix-run wins).
set -uo pipefail
R=${1:-/srv/qwen5090/results/2026-07-20-deepswe-full}
SV=/srv/qwen5090/swebench-eval
. "$SV/.venv/bin/activate"

python3 - "$R" <<'PYEOF'
import json,re,sys
R=sys.argv[1]
DENY={"tox.ini","pyproject.toml","setup.cfg","setup.py"}
def sanitize(patch):
    if not patch: return patch
    blocks=re.split(r'(?m)^(?=diff --git )',patch)
    head,blocks=(blocks[0],blocks[1:]) if blocks and not blocks[0].startswith("diff --git") else ("",blocks)
    keep,drop=[],[]
    for b in blocks:
        m=re.match(r'diff --git a/(\S+) b/(\S+)',b)
        (drop if m and m.group(2) in DENY else keep).append(b)
    return (head+"".join(keep)) if (drop and keep) else patch
rep=json.load(open(f"{R}/qwen3.6-27b-tiers.deepswe.json"))
resolved=set(rep["resolved_ids"])
out,changed=[],[]
for line in open(f"{R}/preds.jsonl"):
    p=json.loads(line)
    s=sanitize(p["model_patch"])
    if s!=p["model_patch"] and p["instance_id"] not in resolved:
        changed.append(p["instance_id"])
        out.append({**p,"model_patch":s})
with open(f"{R}/preds_fix.jsonl","w") as f:
    for p in out: f.write(json.dumps(p)+"\n")
print(f"sanitized+rescoring: {len(changed)} instances")
PYEOF

cd "$R"
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path "$R/preds_fix.jsonl" \
  --max_workers 6 --run_id deepswefix --namespace swebench \
  --cache_level env --timeout 1800 \
  >> "$R/score.log" 2>&1 || { echo "SANITIZE-RESCORE HARNESS FAILED"; tail -5 "$R/score.log"; exit 1; }

python3 - "$R" <<'PYEOF'
import json,glob,sys
R=sys.argv[1]
base=json.load(open(f"{R}/qwen3.6-27b-tiers.deepswe.json"))
fix=json.load(open(f"{R}/qwen3.6-27b-tiers.deepswefix.json"))
resolved=set(base["resolved_ids"])|set(fix["resolved_ids"])
r2e={}
for f in glob.glob(R+'/traj/*.jsonl'):
    for line in open(f):
        line=line.strip()
        if not line: continue
        d=json.loads(line); r2e[d['ds']['instance_id']]=(d.get('reward') or 0)>0
n=500
ids=set(r2e)
r2e_only=sorted(i for i in ids if r2e[i] and i not in resolved)
off_only=sorted(i for i in ids if not r2e[i] and i in resolved)
print(f"### FINAL(sanitized) resolved={len(resolved)}/{n} ({100*len(resolved)/n:.1f}%) | "
      f"fix-run recovered={len(set(fix['resolved_ids']))}/{len(fix['submitted_ids'])} | "
      f"R2E-solved={sum(r2e.values())} | R2E-only={len(r2e_only)} | official-only={len(off_only)} ###")
print("errors in fix-run:",len(fix.get("error_ids",[])))
PYEOF
