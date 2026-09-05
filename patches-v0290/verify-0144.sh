#!/usr/bin/env bash
# Usage: bash deliver/verify-0144.sh (offline, from any directory).
set -euo pipefail
cd "$(dirname "$0")/.."
work="$PWD/.work/0144"
rm -rf "$work"
mkdir -p "$work"
cp -R src/vllm src/pcie_ipc_ar21 "$work/"
patch --dry-run -d "$work" -p1 --fuzz=0 --batch --forward < deliver/0144-pcie-ipc-fused-norm-v0290.diff
if patch -d "$work" -p1 --fuzz=0 --batch --forward < deliver/0144-pcie-ipc-fused-norm-v0290.diff; then
    echo 'PASS: dry-run and real apply at fuzz 0'
else
    exit 1
fi
while IFS= read -r file; do
    PYTHONPYCACHEPREFIX="$work/pycache" python3 -m py_compile "$work/$file"
done < deliver/touched-python.txt
PYTHONPYCACHEPREFIX="$work/pycache" python3 -m py_compile deliver/pcie_fused_norm_check.py
cmp deliver/pcie_ipc_fused_norm.cu "$work/pcie_ipc_ar21/csrc/pcie_ipc_fused_norm.cu"
cmp src/pcie_ipc_ar21/include/flashinfer/comm/pcie_ipc_all_reduce.cuh "$work/pcie_ipc_ar21/include/flashinfer/comm/pcie_ipc_all_reduce.cuh"
echo 'PASS: touched Python compiles; kernel copy matches; vendor header unchanged'
python3 deliver/test-0144.py
echo 'PASS: 0144 offline verification complete (CUDA build/execution not run)'
