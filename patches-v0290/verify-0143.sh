#!/usr/bin/env bash
# Usage: bash deliver/verify-0143.sh (offline, from the supplied source-dump workspace).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p .work/0143
if [ -e .work/0143/vllm ]; then rm -rf .work/0143/vllm; fi
cp -R src/vllm .work/0143/vllm
patch -d .work/0143 -p1 --fuzz=0 --batch --forward --dry-run < deliver/0143-vocab-collective-tags-v0290.diff
if patch -d .work/0143 -p1 --fuzz=0 --batch --forward < deliver/0143-vocab-collective-tags-v0290.diff; then
    echo 'real apply: PASS (fuzz 0)'
else
    status=$?
    echo "real apply: FAIL ($status)" >&2
    exit "$status"
fi
while IFS= read -r path; do
    python3 -m py_compile ".work/0143/$path"
done < deliver/touched-0143.txt
echo 'py_compile: PASS (7 touched files)'
python3 deliver/test-0143.py .work/0143
echo 'verify-0143: PASS'
