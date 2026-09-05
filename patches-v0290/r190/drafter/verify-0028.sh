#!/bin/sh
# Usage: sh deliver/verify-0028.sh (offline, from any directory).
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK="$ROOT/.work/0028"
mkdir -p "$WORK"
# Brief28 is scripts-only: there is no source diff to dry-run or apply.
# Compile/test copies; leave src/ and evidence/ untouched.
cp "$ROOT"/deliver/*.py "$ROOT/deliver/drafter-reference.json" "$WORK/"
PYTHON=${PYTHON:-python3}
"$PYTHON" -m py_compile "$WORK"/*.py
rm -f "$WORK/tiny.jsonl"
"$PYTHON" "$WORK/build_calib_corpus.py" --dry-run --out "$WORK/tiny.jsonl"
"$PYTHON" "$WORK/recalibrate_drafter.py" --dry-run
"$PYTHON" "$WORK/measure_acceptance.py" --dry-run
"$PYTHON" -m unittest discover -s "$WORK" -p test_phase_a.py
if "$PYTHON" "$WORK/build_calib_corpus.py" --count 0 --out "$WORK/invalid.jsonl" >"$WORK/invalid.log" 2>&1; then
    echo 'FAIL invalid arguments accepted' >&2
    exit 1
fi
if "$PYTHON" "$WORK/recalibrate_drafter.py" --stage quantize >"$WORK/blocked.log" 2>&1; then
    echo 'FAIL unavailable quantization reported success' >&2
    exit 1
fi
echo 'PASS py_compile: all 4 Python scripts'
echo 'PASS dry-runs, 3 dependency-free tests, invalid CLI and blocked quantization'
echo 'SKIP patch apply: BRIEF28 explicitly requests scripts, no patch'
echo 'UNVERIFIED GPU capture; NOT DELIVERED: executable GPTQ/export pipeline'
