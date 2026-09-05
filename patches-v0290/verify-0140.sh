#!/bin/sh
# Usage (offline, repo root): sh deliver/verify-0140.sh
set -eu
cd "$(dirname "$0")/.."
rm -rf .work/0140 .work/0140a
mkdir -p .work/0140 .work/0140a
cp -R src/vllm .work/0140/vllm
cp -R src/vllm .work/0140a/vllm
apply() {
    tree=$1; diff=$2
    patch -d "$tree" -p1 --fuzz=0 --batch --forward --dry-run < "$diff"
    if patch -d "$tree" -p1 --fuzz=0 --batch --forward < "$diff"; then
        :
    else
        status=$?
        echo "Real apply failed ($status): $diff" >&2
        exit "$status"
    fi
    python3 - "$tree" "$diff" <<'PY'
import pathlib, py_compile, sys
root=pathlib.Path(sys.argv[1])
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    if line.startswith('+++ b/') and line.endswith('.py'):
        py_compile.compile(str(root/line[6:]), doraise=True)
PY
}
apply .work/0140 deliver/0139-nvfp4-marlin-allowlist-v0290.diff
apply .work/0140 deliver/0140-nvfp4-shape-dispatch-v0290.diff
apply .work/0140a deliver/0140a-nvfp4-shape-dispatch-v0290.diff
python3 -m py_compile deliver/nvfp4_gemm_census.py deliver/build_shape_census.py deliver/make_dispatch_table.py deliver/test_0140.py
python3 deliver/test_0140.py .work/0140
python3 deliver/test_0140.py .work/0140a
python3 deliver/build_shape_census.py > .work/0140/census.md
cmp deliver/gemm_shape_census.md .work/0140/census.md
printf '%s\n' '0140 + 0139: fuzz-0 dry-run/apply + py_compile + pure-Python tests PASS' '0140a standalone: fuzz-0 dry-run/apply + py_compile + pure-Python tests PASS' 'Generated shape census matches: PASS'
