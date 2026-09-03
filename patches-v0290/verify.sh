#!/bin/sh
set -eu

OUT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUNDLE_DIR=$(CDPATH= cd -- "$OUT_DIR/../.." && pwd)
PATCH_DIR="$OUT_DIR/patches-v0290"
WORK_DIR=$(mktemp -d /private/tmp/vllm-v0290-verify.XXXXXX)
trap 'rm -rf -- "$WORK_DIR"' EXIT HUP INT TERM
cp -R "$BUNDLE_DIR/v029src" "$WORK_DIR/tree"

apply_package_patch() {
    patch_name=$1
    echo "+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < $patch_name"
    patch --dry-run -p1 --fuzz=0 --batch \
        -d "$WORK_DIR/tree/vllm" < "$PATCH_DIR/$patch_name"
    patch -p1 --fuzz=0 --batch \
        -d "$WORK_DIR/tree/vllm" < "$PATCH_DIR/$patch_name" >/dev/null
}

apply_tree_patch() {
    patch_path=$1
    patch_name=$(basename -- "$patch_path")
    echo "+ patch --dry-run -p1 --fuzz=0 -d tree < $patch_name"
    patch --dry-run -p1 --fuzz=0 --batch \
        -d "$WORK_DIR/tree" < "$patch_path"
    patch -p1 --fuzz=0 --batch \
        -d "$WORK_DIR/tree" < "$patch_path" >/dev/null
}

for patch_name in \
    0101-sm120-nvfp4kv-fa2-routing-v0280.diff \
    0103-sm120-nvfp4-xqa-decode-v0280.diff \
    0104-mtp-drafter-full-cudagraph-v0280.diff \
    0105-dflash-noncausal-nvfp4-fa2-v0280.diff \
    0106-dflash2-selector-sampling-v0280.diff \
    0107-dflash-quantized-draft-loader.diff \
    0108-gdn-kernels-v0280.diff \
    0109-dflash-noncausal-complete-v0280.diff \
    0111-replayssm-spec-v0280.diff \
    0112-xqa-verify-v0280.diff \
    0113-dflash-speculator-graphs-v0280.diff
do
    apply_package_patch "$patch_name"
done

for patch_name in \
    0116-dflash-nvfp4-revival.diff \
    0117-dflash-nvfp4-warmup.diff \
    0118b-dflash-eager-escape-rebased.diff \
    0119-dflash-nvfp4-fullgraph-width.diff \
    0129-dflash-nvfp4-drafter-graphs.diff \
    0131-nvfp4-pooled-int-workspace.diff
do
    apply_tree_patch "$PATCH_DIR/$patch_name"
done

apply_tree_patch "$OUT_DIR/0132-masked-nvfp4-xqa-sm120-v0290.diff"
apply_tree_patch "$OUT_DIR/0133-gdn-packed-decode-bv16-v0290.diff"
apply_tree_patch "$OUT_DIR/0134-dflash-no-eagle-block-drop-v0290.diff"

for file in $(
    for patch_path in "$PATCH_DIR"/*.diff \
        "$OUT_DIR/0132-masked-nvfp4-xqa-sm120-v0290.diff" \
        "$OUT_DIR/0133-gdn-packed-decode-bv16-v0290.diff" \
        "$OUT_DIR/0134-dflash-no-eagle-block-drop-v0290.diff"
    do
        sed -n 's@^+++ b/@@p' "$patch_path"
    done | sed 's@^vllm/@@' | sort -u
)
do
    case "$file" in
        *.py)
            echo "+ python3 -m py_compile tree/vllm/$file"
            python3 -m py_compile "$WORK_DIR/tree/vllm/$file"
            ;;
    esac
done

echo "VERIFIED: phase 1 + 0132 + 0133 + 0134"
