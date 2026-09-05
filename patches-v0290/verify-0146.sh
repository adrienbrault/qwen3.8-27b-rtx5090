#!/usr/bin/env bash
# Usage: bash deliver/verify-0146.sh (offline, from any directory).
set -euo pipefail
cd "$(dirname "$0")/.."
scratch=.work/0146
mkdir -p "$scratch"
rm -rf "$scratch/vllm"
cp -R src/vllm "$scratch/vllm"
patch -d "$scratch" -p1 --fuzz=0 --batch --forward --dry-run < deliver/0146-syncfree-seqlens-v0290.diff
if patch -d "$scratch" -p1 --fuzz=0 --batch --forward < deliver/0146-syncfree-seqlens-v0290.diff; then
    echo 'PASS: dry-run and real apply, fuzz=0'
else
    status=$?
    echo "FAIL: real apply exited $status" >&2
    exit "$status"
fi
while IFS= read -r file; do
    python3 -m py_compile "$scratch/$file"
done < deliver/touched-0146.txt
echo 'PASS: py_compile all 6 touched Python files'
python3 deliver/test-0146.py "$scratch"
stack=.work/0146-census
mkdir -p "$stack"
rm -rf "$stack/vllm"
cp -R "$scratch/vllm" "$stack/vllm"
patch -d "$stack" -p1 --fuzz=0 --batch --forward --dry-run < evidence/r190d/0142c-dflash-sync-census-v0290.diff
if patch -d "$stack" -p1 --fuzz=0 --batch --forward < evidence/r190d/0142c-dflash-sync-census-v0290.diff; then
    python3 -m py_compile "$stack/vllm/envs.py" "$stack/vllm/v1/worker/gpu/model_runner.py" "$stack/vllm/v1/worker/gpu/sync_census.py"
else
    status=$?
    echo "FAIL: census stack apply exited $status" >&2
    exit "$status"
fi
echo 'PASS: 0142c census stacked dry-run, real apply (fuzz=0), py_compile'
echo 'PASS: 5 dependency-free tests'
echo 'PASS: 0146 offline verification complete (no torch/vllm imports)' 
