#!/usr/bin/env bash
# BA4 lean promotion gate (OPERATIONS §18, user 2026-10-19: "we should be able to be much faster and efficient"). Replaces
# the unit-j35 recurrent certificate for BA4. One GPU unit, ~30 min:
#   L1a boot -> decode probe x2 (control) + tool smoke -> BA4 boot -> decode probe + tool smoke -> decide.py (rule in its
#   header, written before the data) -> GLM restored unless another unit is queued.
# Both arms boot through the daily launcher (EXP=1, :8029) with the exact knobs of the screen that measured BA4
# (results/2026-10-19-r27-audition-screen-070155-3532152, boots r1-01-A_S_L1a / r1-02-L1a_BA4, launch-env.nul).
#   install: rsync -a scripts/r27-ba4-lean/ scripts/decode_fidelity.py <serving host>:/srv/qwen5090/ba4-lean/
#   (OPERATOR_BIN: the directory holding llama-benchy and tool-eval-bench, default $HOME/.local/bin)
#   run:     ssh <serving host> 'sudo systemd-run --unit=r27-ba4-lean-$(date -u +%H%M%S) -p RuntimeMaxSec=7200 bash /srv/qwen5090/ba4-lean/gate.sh'
#   (r<digits> unit name and summary.txt: the flan watch only follows those; the first ba4-lean-* runs never reached it)
set -euo pipefail
export HOME=${HOME:-/root} USER=${USER:-root}   # systemd units start without HOME; launch-daily-v0280.sh needs it (first run, 2026-10-19)
export PATH=${OPERATOR_BIN:-$HOME/.local/bin}:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin   # the launcher pre-warms with llama-benchy (second run)
S=/srv/qwen5090/results/2026-10-19-r27-audition-screen-070155-3532152
H=/srv/qwen5090/ba4-lean
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-ba4-lean-gate
CORPUS=/srv/qwen5090/r156-corpus.jsonl
URL=http://127.0.0.1:8029
mkdir -p "$R"; exec > >(tee -a "$R/gate.log") 2>&1
GPU_QUEUE_NAME=ba4-lean . /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock 9
finish() {
  rc=$?
  docker logs vllm-exp > "$R/last-engine.log" 2>&1 || true
  timeout 120 docker rm -f vllm-exp >/dev/null 2>&1 || true
  flock -u 9
  if [ -n "$(gpu_queue_others)" ]; then echo "GLM restore skipped: queued $(gpu_queue_others)"
  else bash /srv/qwen5090/r858/restart-glm-daily.sh || echo "SUMMARY ba4-lean RESTORE-FAILED glm daily down"; fi
  echo "SUMMARY ba4-lean $(cat "$R/verdict.txt" 2>/dev/null || echo "FAIL rc=$rc (no verdict)") results=$R" | tee -a "$R/summary.txt"
}
trap finish EXIT
docker stop -t 80 glm53 > "$R/glm-stop.log" 2>&1 || true
mkdir -p "$R/ba4-inputs"; cp -a "$S/preflight/ba4-inputs/." "$R/ba4-inputs/"

boot() {  # <arm label> <screen boot dir> [extra serving args, appended to EXTRA_ARGS_APPEND]
  local arm=$1 src=$S/$2 extra=${3:-}
  mkdir -p "$R/$arm-cores"
  sudo find /srv/qwen5090/eval-l2 -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +   # no tier hits across arms
  python3 - "$src/launch-env.nul" "$S" "$src" "$R" "$arm" > "$R/$arm-launch.env" <<'PY'
import shlex, sys
env_file, screen, boot, out, arm = sys.argv[1:]
drop = {"EXP_PORT", "EXP_L2", "EXP_INNER_LAUNCHER", "PATH", "HOME", "USER"}
for kv in open(env_file, "rb").read().decode().split("\0"):
    if not kv or "=" not in kv: continue
    k, v = kv.split("=", 1)
    if k in drop or k.startswith("R27_"): continue
    v = v.replace(boot + "/cores", f"{out}/{arm}-cores").replace(screen + "/preflight/ba4-inputs", out + "/ba4-inputs")
    if screen in v: raise SystemExit(f"unmapped screen path in {k}")
    if k in ("EXTRA_ENV_APPEND", "EXTRA_MOUNT_APPEND"):
        # The harness parsed these with shell quoting; the daily launcher word-splits them raw, so a quoted
        # '-e 'VAR=...'' reached the container as an env var NAMED 'VAR (third run: BA4 96/96 INACTIVE). Unquote.
        parts = shlex.split(v)
        if any(" " in x for x in parts): raise SystemExit(f"{k}: a value with spaces cannot cross the launcher")
        v = " ".join(parts)
    print(f"export {k}={shlex.quote(v)}")
PY
  [ -z "$extra" ] || printf 'export EXTRA_ARGS_APPEND="$EXTRA_ARGS_APPEND "%q\n' "$extra" >> "$R/$arm-launch.env"
  [ -z "${BOOT_XENV:-}" ] || printf 'export EXTRA_ENV_APPEND="$EXTRA_ENV_APPEND "%q\n' "$BOOT_XENV" >> "$R/$arm-launch.env"
  [ -z "${BOOT_KV_BYTES:-}" ] || echo "export KV_BYTES=$BOOT_KV_BYTES" >> "$R/$arm-launch.env"
  [ -z "${BOOT_XMOUNT:-}" ] || printf 'export EXTRA_MOUNT_APPEND="$EXTRA_MOUNT_APPEND "%q\n' "$BOOT_XMOUNT" >> "$R/$arm-launch.env"
  ( set -a; . "$R/$arm-launch.env"; EXP=1 bash /srv/qwen5090/launch-daily.sh ) > "$R/$arm-boot.log" 2>&1 \
    || { tail -30 "$R/$arm-boot.log"; return 1; }
  tail -1 "$R/$arm-boot.log"
}
probe() {  # <tag>
  for ctx in 0 30000; do
    python3 "$H/decode_fidelity.py" run --url "$URL" --corpus "$CORPUS" --out "$R/$1-ctx$ctx.jsonl" \
      --chunks 20 --ctx "$ctx" --tokens 512 > "$R/$1-ctx$ctx.log"
  done
}

if [ "${MODE:-decode}" = kernel ]; then
  # Kernel-error arm (kernel_decide.py rule): eager BA4 boot with a TEST-ONLY hook (ba4_errstat_hook.py + .pth, mounted
  # into dist-packages) that wraps vllm_ba4.runtime.try_apply and measures BA4 vs fp64 vs the original GEMM on every live
  # M=4 call. Eager, because CUDA-graph replay runs no Python. The vllm_ba4 files stay byte-identical (a patched
  # runtime.py failed BA4's code-hash admission: implementation_changed). The launcher's 0131 log assert does not fire in
  # eager and returns nonzero after a HEALTHY boot; the engine stays up, so check it directly.
  SP=/usr/local/lib/python3.12/dist-packages
  mkdir -p "$R/errstat"; chmod 777 "$R/errstat"
  # Eager needs more activation headroom than the graph daily: the 14.86 GB pin OOM'd rank 0 in FlashInfer autotune and
  # hung the boot (first kernel run). Pool size does not affect what this arm measures.
  BOOT_KV_BYTES=9000000000 BOOT_XENV="-e BA4_ERRSTAT=/errstat" \
    BOOT_XMOUNT="-v $H/ba4_errstat_hook.py:$SP/ba4_errstat_hook.py:ro -v $H/ba4_errstat_hook.pth:$SP/ba4_errstat_hook.pth:ro -v $R/errstat:/errstat" \
    boot BA4 r1-02-L1a_BA4 "--enforce-eager" \
    || curl -sf -m 10 "$URL/v1/models" >/dev/null || { echo "eager BA4 engine not up"; exit 1; }
  grep -q "SM12X BA4 ACTIVE" <(docker logs vllm-exp 2>&1) || { echo "BA4 not active"; exit 1; }
  python3 "$H/decode_fidelity.py" run --url "$URL" --corpus "$CORPUS" --out "$R/kernel-ctx0.jsonl" --chunks 20 --ctx 0 --tokens 256 > "$R/kernel-ctx0.log"
  python3 "$H/decode_fidelity.py" run --url "$URL" --corpus "$CORPUS" --out "$R/kernel-ctx30000.jsonl" --chunks 5 --ctx 30000 --tokens 256 > "$R/kernel-ctx30000.log"
  python3 "$H/toolsmoke.py" "$URL" "$R/kernel-tools.json" || true
  docker logs vllm-exp > "$R/BA4-engine.log" 2>&1; timeout 120 docker rm -f vllm-exp >/dev/null
  set +e; python3 "$H/kernel_decide.py" "$R/errstat" > "$R/VERDICT.json"; drc=$?; set -e
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); print("kernel", v["verdict"], "ratio_mean_abs", round(v.get("ratio_mean_abs",0),4), "ratio_rms", round(v.get("ratio_rms",0),4), json.dumps(v.get("checks")))' "$R/VERDICT.json" > "$R/verdict.txt"
  cat "$R/verdict.txt"; exit $drc
fi
if [ "${MODE:-decode}" = tooleval ]; then
  # Tool-eval A/B (tool_decide.py rule), booted as served, the r27 QUALIFY invocation.
  for pair in "L1a r1-01-A_S_L1a" "BA4 r1-02-L1a_BA4"; do
    set -- $pair
    boot "$1" "$2"
    [ "$1" = L1a ] || grep -q "SM12X BA4 ACTIVE" <(docker logs vllm-exp 2>&1) || { echo "BA4 not active"; exit 1; }
    ( cd "$HOME" && timeout 5400 tool-eval-bench --base-url "$URL/v1" --model qwen3.8-27b --temperature 0.6 --top-p 0.95 \
        --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$1.json" > "$R/tooleval-$1.log" 2>&1 )
    python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$1.json" "$1" || true
    docker logs vllm-exp > "$R/$1-engine.log" 2>&1; timeout 120 docker rm -f vllm-exp >/dev/null
  done
  set +e; python3 "$H/tool_decide.py" "$R" > "$R/VERDICT.json"; drc=$?; set -e
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); print("tooleval", v["verdict"], v["L1a"]["mean"], v["L1a"]["ci95"], v["BA4"]["mean"], v["BA4"]["ci95"])' "$R/VERDICT.json" > "$R/verdict.txt"
  cat "$R/verdict.txt"; exit $drc
fi
if [ "${MODE:-decode}" = gsm ]; then
  # Task-accuracy A/B (gsm_decide.py rule): the R299 think-off recipe, the single quotes are load-bearing (R299 header).
  THINK_OFF="--default-chat-template-kwargs '{\"preserve_thinking\":true,\"enable_thinking\":false}'"
  for pair in "L1a r1-01-A_S_L1a" "BA4 r1-02-L1a_BA4"; do
    set -- $pair
    boot "$1" "$2" "$THINK_OFF"
    [ "$1" = L1a ] || grep -q "SM12X BA4 ACTIVE" <(docker logs vllm-exp 2>&1) || { echo "BA4 not active"; exit 1; }
    timeout 7200 /srv/qwen5090/venv-lmeval/bin/lm_eval --model local-chat-completions \
      --model_args "base_url=$URL/v1/chat/completions,model=qwen3.8-27b,tokenizer=/srv/qwen5090/models/qwen3.8-27b-nvidia-nvfp4,num_concurrent=4,max_retries=1,tokenized_requests=False" \
      --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0 --log_samples \
      --output_path "$R/gsm-$1" > "$R/gsm-$1.log" 2>&1
    docker logs vllm-exp > "$R/$1-engine.log" 2>&1; timeout 120 docker rm -f vllm-exp >/dev/null
  done
  set +e; python3 "$H/gsm_decide.py" "$R" > "$R/VERDICT.json"; drc=$?; set -e
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); print("gsm8k", v["verdict"], v["acc_L1a"], v["acc_BA4"], v["only_L1a_correct"], v["only_BA4_correct"], v["p_one_sided"])' "$R/VERDICT.json" > "$R/verdict.txt"
  cat "$R/verdict.txt"; exit $drc
fi
if [ -n "${REUSE_L1A:-}" ]; then   # the L1a arm of an earlier gate run (same corpus/knobs; L1a is the unchanged daily)
  cp -r "$REUSE_L1A"/L1a-* "$R"/; echo "reusing L1a arm from $REUSE_L1A"
else
  boot L1a r1-01-A_S_L1a
  probe L1a-1; probe L1a-2
  python3 "$H/toolsmoke.py" "$URL" "$R/L1a-tools.json"
  docker logs vllm-exp > "$R/L1a-engine.log" 2>&1; timeout 120 docker rm -f vllm-exp >/dev/null
fi

boot BA4 r1-02-L1a_BA4
grep -q "SM12X BA4 ACTIVE" <(docker logs vllm-exp 2>&1) || { echo "BA4 not active after boot:"; docker logs vllm-exp 2>&1 | grep -a "SM12X BA4" | head -3; }
probe BA4
python3 "$H/toolsmoke.py" "$URL" "$R/BA4-tools.json"
docker logs vllm-exp > "$R/BA4-engine.log" 2>&1

set +e; python3 "$H/decide.py" "$R" > "$R/VERDICT.json"; drc=$?; set -e
python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); print(v["verdict"], json.dumps(v["checks"]))' "$R/VERDICT.json" > "$R/verdict.txt"
cat "$R/verdict.txt"
exit $drc
