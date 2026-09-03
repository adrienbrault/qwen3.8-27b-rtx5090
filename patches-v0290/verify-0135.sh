#!/bin/sh
# verify 0135 against a PRISTINE extraction of the …-revival-prs vLLM package (codex's original verify copied a tree it
# had already patched in place, so its control was circular — 2026-09-03). Usage:
#   sh verify-0135.sh <tgz whose top-level dir is the vllm package, e.g. v0290prs-vllm.tgz>
# Steps: dry-run --fuzz=0 → real apply → CONTROL (second apply must fail) → py_compile. Exit non-zero on any failure.
set -eu
OUT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PATCH="$OUT_DIR/0135-embed-uva-offload-v0290.diff"
TGZ=${1:?usage: verify-0135.sh <package tgz>}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/vllm-0135-verify.XXXXXX")
trap 'rm -rf -- "$WORK"' EXIT HUP INT TERM
mkdir -p "$WORK/x" "$WORK/tree"; tar -C "$WORK/x" -xzf "$TGZ"
top=$(ls "$WORK/x" | head -1); mv "$WORK/x/$top" "$WORK/tree/vllm"
test -f "$WORK/tree/vllm/__init__.py"
if grep -q maybe_offload_embeddings "$WORK/tree/vllm/model_executor/model_loader/utils.py"; then echo "FAILED: tree already carries 0135 (not pristine)" >&2; exit 1; fi
echo "+ dry-run"; patch --dry-run --batch -p1 --fuzz=0 -d "$WORK/tree" < "$PATCH"
echo "+ apply";   patch --batch --forward -p1 --fuzz=0 -d "$WORK/tree" < "$PATCH" >/dev/null
echo "+ control (second apply must fail)"
if patch --dry-run --batch --forward -p1 --fuzz=0 -d "$WORK/tree" < "$PATCH" >/dev/null 2>&1; then echo "FAILED: second apply succeeded" >&2; exit 1; fi
echo "+ py_compile"
python3 -m py_compile "$WORK/tree/vllm/entrypoints/llm.py" "$WORK/tree/vllm/model_executor/model_loader/utils.py" \
  "$WORK/tree/vllm/model_executor/offloader/base.py" "$WORK/tree/vllm/model_executor/offloader/uva.py"
grep -q maybe_offload_embeddings "$WORK/tree/vllm/model_executor/model_loader/utils.py"
echo "VERIFY-0135 OK"
