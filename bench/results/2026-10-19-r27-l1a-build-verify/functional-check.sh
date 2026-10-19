#!/usr/bin/env bash
# Functional check of the public 27B clean-room image before it is published (2026-10-19; user: "yes please improve").
# Boots the public repo's own scripts/serve.sh (its boot-log/patch-marker checks run) three times, A B A:
#   A = the served tag vllm-qwen38:v0310-r27c-A_S-L1a-ubplan, B = the clean-room image, A again,
# and captures greedy chat completions for a fixed prompt set on each. A == A' proves the probe is deterministic across
# boots; B == A then means the clean-room build behaves like the served one. Restores GLM (restart-glm-daily.sh) at the end.
#   ssh flan 'bash -s' < flan/public-27b-functional-check.sh -- <cleanroom image tag> [public tree dir]
set -euo pipefail
[ "${1:-}" = -- ] && shift
CR=${1:?clean-room image tag}; TREE=${2:-/srv/qwen5090/cleanroom-27b/repo}
ROOT=/srv/qwen5090
SERVED=vllm-qwen38:v0310-r27c-A_S-L1a-ubplan
R=$ROOT/results/public-27b-functional-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$R"
exec > >(tee "$R/run.log") 2>&1
export GPU_QUEUE_NAME=public-27b-functional
. "$ROOT/lib/gpu-queue.sh"
exec 9>"$ROOT/gpu-exclusive.lock"; flock 9
echo "== $(date -Is) lock held; served=$SERVED cleanroom=$CR tree=$TREE"
sudo -n docker image inspect --format '{{.Id}}' "$SERVED" "$CR" | tee "$R/image-ids.txt"
restore(){ echo "== restore GLM"; sudo -n docker rm -f pub27b-check >/dev/null 2>&1 || true
  bash "$ROOT/r858/restart-glm-daily.sh" > "$R/glm-restore.log" 2>&1 && echo "GLM restore rc=0" || echo "GLM restore rc=$?"; }
trap restore EXIT
sudo -n docker stop -t 60 glm53 >/dev/null; echo "glm53 stopped"
PROMPTS=(
  "Write a Python function that parses ISO-8601 durations such as P3DT4H5M into seconds, with tests."
  "Explain, in four short paragraphs, why the sky is blue and why sunsets are red."
  "A train leaves at 09:40 and arrives at 13:05 after one 12-minute stop. How long was it moving? Show the steps."
  "Translate into French: The committee postponed the decision until the budget figures are confirmed."
  "Write a bash one-liner that lists the five largest files under the current directory, human-readable sizes."
)
arm(){ local tag=$1 label=$2 out=$R/$2
  mkdir -p "$out"; echo "== $(date -Is) boot $label ($tag)"
  if ! (cd "$TREE" && IMAGE=$tag MODEL_DIR=$ROOT/models/qwen3.8-27b-nvidia-nvfp4 PORT=8031 NAME=pub27b-check GPUS=0,1 \
        DOCKER="sudo -n docker" CACHE_DIR=$ROOT/cache/pub27b-check-$label HEALTH_TIMEOUT=1800 bash scripts/serve.sh) \
        > "$out/serve.log" 2>&1; then
    echo "BOOT FAIL $label"; tail -20 "$out/serve.log"; sudo -n docker logs pub27b-check > "$out/engine.log" 2>&1 || true; return 1; fi
  echo "boot ok $label: $(grep -cE 'OK|PASS' "$out/serve.log") check lines"
  local i=0
  for p in "${PROMPTS[@]}"; do i=$((i+1))
    python3 - "$p" > "$out/p$i.json" <<'PY'
import json, sys, urllib.request
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": sys.argv[1]}], "temperature": 0, "seed": 0,
        "max_tokens": 768, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8031/v1/chat/completions", json.dumps(body).encode(),
                             {"content-type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
m = r["choices"][0]["message"]
print(json.dumps({"content": m.get("content"), "reasoning": m.get("reasoning_content"),
                  "finish": r["choices"][0]["finish_reason"], "usage": r["usage"]}, sort_keys=True))
PY
    python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps([d["content"],d["reasoning"]]))' "$out/p$i.json" \
      | sha256sum | cut -c1-16 > "$out/p$i.sha"
  done
  sudo -n docker logs pub27b-check > "$out/engine.log" 2>&1 || true
  (cd "$TREE" && NAME=pub27b-check DOCKER="sudo -n docker" bash scripts/serve.sh --stop) > "$out/stop.log" 2>&1 || true
  sudo -n docker rm -f pub27b-check >/dev/null 2>&1 || true
  for g in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && break; sleep 5; done
  sleep 30
}
arm "$SERVED" A1; arm "$CR" B; arm "$SERVED" A2
echo "== greedy hashes (A1 B A2)"
same_aa=1; same_ab=1
for i in $(seq ${#PROMPTS[@]}); do a1=$(cat "$R/A1/p$i.sha"); b=$(cat "$R/B/p$i.sha"); a2=$(cat "$R/A2/p$i.sha")
  echo "p$i $a1 $b $a2"; [ "$a1" = "$a2" ] || same_aa=0; [ "$a1" = "$b" ] || same_ab=0; done
if [ $same_aa = 1 ] && [ $same_ab = 1 ]; then V="PASS clean-room image greedy-identical to the served image on ${#PROMPTS[@]} prompts (A==A' deterministic)"
elif [ $same_aa = 0 ]; then V="INCONCLUSIVE served image not greedy-deterministic across boots (A!=A')"
else V="FAIL clean-room image differs from the served image while A==A'"; fi
echo "SUMMARY public-27b-functional $V" | tee "$R/VERDICT.txt"
