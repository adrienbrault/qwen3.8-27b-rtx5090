#!/usr/bin/env bash
# Usage: bash deliver/verify-0142.sh (offline, from any directory).
set -euo pipefail
cd "$(dirname "$0")/.."
for id in 0142 0142b; do
    scratch=".work/$id"
    rm -rf "$scratch"
    mkdir -p "$scratch"
    cp -R src/vllm "$scratch/vllm"
    if [[ "$id" == 0142 ]]; then slug=dflash-sync-census; else slug=dflash-device-side-accept; fi
    diff="$PWD/deliver/$id-$slug-v0290.diff"
    patch -d "$scratch" -p1 --fuzz=0 --batch --forward --dry-run < "$diff"
    patch -d "$scratch" -p1 --fuzz=0 --batch --forward < "$diff" || { status=$?; echo "real apply failed: $status"; exit "$status"; }
    python3 - "$scratch" "$diff" <<'PY'
import pathlib, py_compile, sys
root = pathlib.Path(sys.argv[1])
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    if line.startswith('+++ b/') and line.endswith('.py'):
        py_compile.compile(str(root/line[6:]), doraise=True)
PY
    echo "$id: dry-run, real apply (fuzz=0), py_compile PASS"
done
# Verify composition too; production images can stack these independent layers.
patch -d .work/0142 -p1 --fuzz=0 --batch --forward --dry-run < deliver/0142b-dflash-device-side-accept-v0290.diff
patch -d .work/0142 -p1 --fuzz=0 --batch --forward < deliver/0142b-dflash-device-side-accept-v0290.diff || { status=$?; exit "$status"; }
python3 -m py_compile .work/0142/vllm/envs.py .work/0142/vllm/v1/worker/gpu/model_runner.py
python3 deliver/test-0142.py .work/0142
printf '%s\n' '0142 + 0142b: stacked apply and dependency-free tests PASS' 'VERIFY-0142 PASS (no torch/vllm imports; no GPU)'
