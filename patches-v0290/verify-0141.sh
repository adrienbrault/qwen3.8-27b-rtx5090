#!/usr/bin/env bash
# Usage: bash deliver/verify-0141.sh (offline, from any directory)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
scratch="$root/.work/0141"
mkdir -p "$scratch"
rm -rf "$scratch/vllm"
cp -R src/vllm "$scratch/vllm"
patch -d "$scratch" -p1 --fuzz=0 --batch --forward --dry-run < deliver/0141-gdn-decode-tuning-v0290.diff
if patch -d "$scratch" -p1 --fuzz=0 --batch --forward < deliver/0141-gdn-decode-tuning-v0290.diff; then
    echo 'Real apply: PASS (exit 0, fuzz 0)'
else
    status=$?
    echo "Real apply: FAIL (exit $status)" >&2
    exit "$status"
fi
python3 -m py_compile "$scratch/vllm/envs.py" \
    "$scratch/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py" \
    "$scratch/vllm/third_party/flash_linear_attention/ops/gdn_decode_config.py" \
    deliver/gdn_decode_microbench.py deliver/test_0141.py
echo 'py_compile: PASS'
python3 deliver/test_0141.py "$scratch"
echo '0141 offline verification: PASS'
