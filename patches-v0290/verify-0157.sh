#!/bin/sh
# Usage: sh deliver/verify-0157.sh (offline, from any directory).
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK="$ROOT/.work/0157"
PATCH="$ROOT/deliver/0157-gdn-common-metadata-once-v0290.diff"
mkdir -p "$WORK"
rm -rf "$WORK/vllm"
cp -R "$ROOT/src/vllm" "$WORK/vllm"
patch -d "$WORK" -p1 --fuzz=0 --batch --forward --dry-run < "$PATCH"
set +e
patch -d "$WORK" -p1 --fuzz=0 --batch --forward < "$PATCH"
status=$?
set -e
if [ "$status" -ne 0 ]; then
    echo "FAIL: real apply exited $status" >&2
    exit "$status"
fi
echo 'PASS: dry-run and real apply at fuzz 0 (exit 0)'
python3 -m py_compile \
    "$WORK/vllm/v1/attention/backends/gdn_attn.py" \
    "$WORK/vllm/v1/worker/gpu/attn_utils.py" \
    "$WORK/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
echo 'PASS: py_compile all 3 touched files'
python3 "$ROOT/deliver/test-0157.py" "$WORK/vllm" "$ROOT/src/vllm"
