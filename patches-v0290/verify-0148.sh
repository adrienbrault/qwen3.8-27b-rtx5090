#!/usr/bin/env bash
# Usage: bash deliver/verify-0148.sh (offline, from any directory; requires python3 and patch).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SCRATCH="$ROOT/.work/0148"
CONTROL="$ROOT/.work/0148-without-0138"
DELIVER="$ROOT/deliver"
rm -rf -- "$SCRATCH" "$CONTROL"
mkdir -p "$SCRATCH" "$CONTROL"
cp -R "$ROOT/src/." "$SCRATCH/"
cp -R "$ROOT/src/vllm" "$CONTROL/vllm"

# Check patch scope and manifest without importing the dumped packages.
python3 - "$DELIVER" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
lines = (root / '0148-pcie-ipc-mtp-drafter-v0290.diff').read_text().splitlines()
assert len(lines) < 250, 'patch exceeds BRIEF30 budget'
paths = [line.removeprefix('+++ b/') for line in lines if line.startswith('+++ b/')]
assert paths == (root / 'touched-0148.txt').read_text().splitlines(), 'manifest mismatch'
assert all(p.startswith('vllm/') and p.endswith('.py') and '..' not in p for p in paths)
PY

checked_patch() {
    local label=$1
    shift
    if patch "$@" > "$SCRATCH/$label.log" 2>&1; then
        return 0
    else
        local patch_status=$?
        cat "$SCRATCH/$label.log" >&2
        return "$patch_status"
    fi
}

checked_patch 0147-dry -d "$SCRATCH" -p1 --fuzz=0 --batch --forward --dry-run \
    < "$ROOT/evidence/0147-bss-not-a-compile-factor-v0290.diff"
checked_patch 0147-apply -d "$SCRATCH" -p1 --fuzz=0 --batch --forward \
    < "$ROOT/evidence/0147-bss-not-a-compile-factor-v0290.diff"
echo 'PASS 0147 dry-run + real apply (fuzz=0, exit=0)'
checked_patch 0148-dry -d "$SCRATCH" -p1 --fuzz=0 --batch --forward --dry-run \
    < "$DELIVER/0148-pcie-ipc-mtp-drafter-v0290.diff"
checked_patch 0148-apply -d "$SCRATCH" -p1 --fuzz=0 --batch --forward \
    < "$DELIVER/0148-pcie-ipc-mtp-drafter-v0290.diff"
echo 'PASS 0148 dry-run + real apply (fuzz=0, exit=0)'

# 0147 is independent of 0138: leave it present in the negative control too.
checked_patch control-0147 -d "$CONTROL" -p1 --fuzz=0 --batch --forward \
    < "$ROOT/evidence/0147-bss-not-a-compile-factor-v0290.diff"
checked_patch control-remove-0138-dry -d "$CONTROL" -p1 --fuzz=0 --batch --reverse --dry-run \
    < "$ROOT/evidence/0138-pcie-ipc-all-reduce-v0290.diff"
checked_patch control-remove-0138 -d "$CONTROL" -p1 --fuzz=0 --batch --reverse \
    < "$ROOT/evidence/0138-pcie-ipc-all-reduce-v0290.diff"
test ! -e "$CONTROL/vllm/distributed/device_communicators/pcie_ipc_all_reduce.py"
# A failed dry-run leaves the reconstructed control untouched. Check the
# actual exit status; status 2 (tool/IO failure) is not a valid negative result.
if patch -d "$CONTROL" -p1 --fuzz=0 --batch --forward --dry-run \
    < "$DELIVER/0148-pcie-ipc-mtp-drafter-v0290.diff" > "$SCRATCH/control-reject-0148.log" 2>&1; then
    echo 'FAIL 0148 unexpectedly applies without 0138' >&2
    exit 1
else
    verify_status=$?
    if [ "$verify_status" -ne 1 ]; then
        cat "$SCRATCH/control-reject-0148.log" >&2
        exit "$verify_status"
    fi
fi
echo 'PASS no-0138 control: reversed full 0138 at fuzz=0; 0148 rejected (exit=1)'

export PYTHONPYCACHEPREFIX="$SCRATCH/pycache"
python3 -m py_compile "$SCRATCH/vllm/config/parallel.py"
while IFS= read -r p; do
    python3 -m py_compile "$SCRATCH/$p"
done < "$DELIVER/touched-0148.txt"
echo 'PASS py_compile: all 5 files touched by 0147 + 0148'
if python3 "$DELIVER/test-0148.py" "$SCRATCH" > "$SCRATCH/tests.log" 2>&1; then
    echo 'PASS dependency-free unit tests (14 tests; details: .work/0148/tests.log)'
else
    verify_status=$?
    cat "$SCRATCH/tests.log" >&2
    exit "$verify_status"
fi
echo 'VERIFY-0148 PASS (offline; no torch/vllm imports, native build or GPU execution)'
