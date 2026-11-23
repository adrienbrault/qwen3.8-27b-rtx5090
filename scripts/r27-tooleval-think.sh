#!/usr/bin/env bash
# r27-tooleval-think (2026-11-23): tool-eval with earlier reasoning sent back. tool-eval-bench 2.1.0 (every tool-eval row
# before this date) drops each assistant turn's reasoning from the history it resends; the Qwen3.8 template then renders
# an empty <think> block on every earlier turn. One boot of the served launcher, the published sampling (69 scenarios x
# 4 trials, T 0.6, top-p 0.95, top-k 20, parallel 8), three arms on the same server:
#   A  tool-eval-bench 2.1.0 as released
#   B  2.1.0 + reasoning echo (scripts/tooleval_echo.py: reasoning_content on tool-calling turns, the 3.0.0 rule)
#   C  tool-eval-bench 3.0.0 as released (sends tool-calling turns' reasoning back)
# scripts/think_echo_probe.py first checks that the server renders echoed reasoning at all.
#   TOOLEVAL21=<venv with tool-eval-bench==2.1.0>/bin RESULTS=<results dir> bash scripts/r27-tooleval-think.sh
# against a server already running on :8020 (scripts/serve-v031-served-daily.sh). C runs through uvx.
set -uo pipefail
U=http://127.0.0.1:8020; T21=${TOOLEVAL21:?venv bin dir with tool-eval-bench 2.1.0}
R=$RESULTS/$(date -u +%F-%H%M%S)-r27-tooleval-think; mkdir -p "$R"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$R/run.log"; }
python3 scripts/think_echo_probe.py "$R/echo-probe.json" --url "$U" --model qwen3.8-27b | tee -a "$R/run.log"
COMMON=(--base-url "$U/v1" --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8)
score() { python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["trial_statistics"]; print("mean", t["final_score_mean"], "sd", t["final_score_stddev"], "median", t["final_score_median"])' "$1" 2>&1; }
"$T21/tool-eval-bench" "${COMMON[@]}" --json-file "$R/A-21.json" > "$R/A-21.log" 2>&1; log "A 2.1.0 rc=$? $(score "$R/A-21.json")"
"$T21/python" scripts/tooleval_echo.py "${COMMON[@]}" --json-file "$R/B-21echo.json" > "$R/B-21echo.log" 2>&1
log "B 2.1.0+echo rc=$? $(score "$R/B-21echo.json") $(grep -a 'tooleval_echo:' "$R/B-21echo.log")"
uvx --from tool-eval-bench==3.0.0 tool-eval-bench "${COMMON[@]}" --json-file "$R/C-30.json" > "$R/C-30.log" 2>&1; log "C 3.0.0 rc=$? $(score "$R/C-30.json")"
