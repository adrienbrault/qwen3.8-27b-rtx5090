#!/usr/bin/env bash
# Build the image served by serve-v031-l1a-daily.sh from this checkout.
# DRY_RUN=1 prints commands and stages/checks local COPY inputs; no Docker or network commands run.
# CHECK=1 also loads the compiled ops with the NVIDIA container runtime, without GPU computation.
# OVERLAY_JOBS=4 is Dockerfile.v0310's recorded default; NO_CACHE=1 bypasses Docker's build cache.
# DOCKER="sudo docker" selects a Docker command; LOG_DIR defaults to build-logs/v0310.
# VLLM_SOURCE_SHA256 defaults to the archive hash verified by the operator on 2026-10-19;
# csrc/ and cmake/patches/ matched a git clone of v0.31.0 (commit below).
# The commit was resolved from GitHub's refs/tags/v0.31.0 API; the Arctic hash/URL are from
# https://pypi.org/pypi/arctic-inference/0.1.1/json (metadata read 2026-10-19).
set -euo pipefail
set -E
cd "$(dirname "$0")/.."

DRY_RUN=${DRY_RUN:-0}
CHECK=${CHECK:-0}
NO_CACHE=${NO_CACHE:-0}
OVERLAY_JOBS=${OVERLAY_JOBS:-4}
LOG_DIR=${LOG_DIR:-build-logs/v0310}
VLLM_SOURCE_SHA256=${VLLM_SOURCE_SHA256:-e20baf18e40f88446f745999379db04c8f88c173dc7afde5dc6b869c06299ad0}
read -r -a DOCKER_CMD <<< "${DOCKER:-docker}"
export DOCKER_BUILDKIT=0

VLLM_TAG=vllm/vllm-openai:v0.31.0
VLLM_DIGEST=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
CUDA_TAG=nvidia/cuda:13.0.3-devel-ubuntu24.04
CUDA_DIGEST=sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
VLLM_ID=
CUDA_ID=
VLLM_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6
SOURCE_URL=https://codeload.github.com/vllm-project/vllm/tar.gz/$VLLM_COMMIT
ARCTIC_SHA256=36857e2fbe3ead34821a04be98dfab95086050c1e38199c8fdfe308234fdcf63
ARCTIC_URL=https://files.pythonhosted.org/packages/2a/d2/3bcebda042d08cfa0f35a2bbc6dd168d60fea60b6678824665c7a3665418/arctic_inference-0.1.1.tar.gz
MANIFEST=patches-v0310/served-l1a-manifest.txt
MANIFEST_SHA256=8423a61a2f8e8f1acab6a9bfc82a4373afce3108f3c7bfa37292cb98b8dd0cdd
MANIFEST_TOOL=scripts/v031_manifest.py
ARCTIC_BUILD_REQUIREMENTS=scripts/arctic-build-requirements.txt
T=vllm-qwen38:v0310-r27c-A-nvfp4kv-mtp
S=vllm-qwen38:v0310-r27c-A_S-splitkv
FINAL=vllm-qwen38:v0310-r27c-A_S-L1a-ubplan
CTX=

log() { printf '%s\n' "$*"; }
die() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
print_cmd() { printf '%q ' "$@"; printf '\n'; }
run() { print_cmd "$@"; if [ "$DRY_RUN" != 1 ]; then "$@"; fi; }
cleanup() { if [ -n "$CTX" ]; then rm -rf -- "$CTX"; fi; }
trap cleanup EXIT
trap 'rc=$?; printf "FAILED (exit %s): %s\n" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

for flag in DRY_RUN CHECK NO_CACHE; do
  case ${!flag} in 0|1) ;; *) die "$flag must be 0 or 1";; esac
done
[[ "$OVERLAY_JOBS" =~ ^[1-9][0-9]*$ ]] || die "OVERLAY_JOBS must be a positive integer"
[ ${#DOCKER_CMD[@]} -gt 0 ] || die "DOCKER is empty"
for required in python3 mktemp cp tar; do command -v "$required" >/dev/null || die "$required is required"; done
if [ "$DRY_RUN" != 1 ]; then
  command -v curl >/dev/null || die "curl is required"
  command -v "${DOCKER_CMD[0]}" >/dev/null || die "Docker command is unavailable: ${DOCKER_CMD[0]}"
fi
EXPECT=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' scripts/serve-v031-l1a-daily.sh)
[ "$EXPECT" = "$FINAL" ] || die "final tag differs from DAILY_IMG in serve-v031-l1a-daily.sh"

file_hash() { python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"; }
require_hash() {
  local file=$1 expected=$2 label=$3 actual
  actual=$(file_hash "$file")
  log "SHA256 $label: $actual"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label SHA256 pin must contain 64 lowercase hexadecimal characters"
  [ "$actual" = "$expected" ] || die "$label SHA256 mismatch: expected $expected, got $actual"
}
require_hash "$MANIFEST" "$MANIFEST_SHA256" "served manifest"

if [ -n "${TMPDIR:-}" ]; then
  CTX=$(mktemp -d "${TMPDIR%/}/v031-build.XXXXXXXX")
else
  CTX=$(mktemp -d)
fi
CTX=$(cd "$CTX" && pwd)
mkdir -p "$CTX/out/A/patches-v0310" "$CTX/assets/wheels" "$CTX/assets/debs" \
  "$CTX/downloads" "$CTX/vllm-v0.31.0/csrc" "$CTX/vllm-v0.31.0/cmake/patches"
# Restore only the staged 0136 diff to phase A's single-argument lambda. The A_S Dockerfile
# then makes the served signature/comment edit; its input hash and L1a's hashes stay mandatory.
python3 - "$CTX" <<'PY'
import hashlib
from pathlib import Path
import re
import shlex
import shutil
import sys

src = Path("patches-v0310")
ctx = Path(sys.argv[1])
chain = ctx / "out/A/patches-v0310"
shutil.copy2(src / "series", chain)
for path in src.glob("*.diff"):
    shutil.copy2(path, chain)
for name in ("overlay", "pcie_ipc_ar21"):
    shutil.copytree(src / name, chain / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in ("apply_layer.py", "guard_flashinfer_workspace.py", "assert_sources.py", "pristine-hashes.json", "Dockerfile.v0310"):
    shutil.copy2(src / name, ctx / "out/A" / name)
for name in ("audit_fi_r27c.py", "test_cpu.py"):
    shutil.copy2(src / name, ctx / "out" / name)
for name in ("Dockerfile.A_S-splitkv", "Dockerfile.A_S-L1a-ubplan", "L1a-draft-ub-plan.diff"):
    shutil.copy2(src / name, ctx / name)
# The public recipes express Python RUN bodies as heredocs. Render the same bodies
# as python -c arguments for the legacy builder, retaining every source/hash assertion.
for name in ("Dockerfile.A_S-splitkv", "Dockerfile.A_S-L1a-ubplan"):
    path = ctx / name
    def inline_python(match):
        args, body = match.groups()
        return "python3 -c " + shlex.quote("exec(" + repr(body) + ")") + args
    text, count = re.subn(r"python3 -([^\n]*?) <<'PY'\n(.*?)^PY\n", inline_python,
                         path.read_text(), flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise SystemExit("expected one Python heredoc in " + name)
    path.write_text(text)
patch = chain / "0136-opt-in-nvfp4-prefill-split-kv-v0310.diff"
if hashlib.sha256(patch.read_bytes()).hexdigest() != "9d058970054485af23b1290fc8ab52c4a26353e4f5abf388f84500f9df9685d8":
    raise SystemExit("0136 public diff drift; review the phase-A reconstruction")
text = patch.read_text()
old = ("+    # The guard takes (kv_data_type) in 0.6.18 and (kv_data_type, device) in\n"
       "+    # 0.7.x; accept any signature so the override cannot raise TypeError.\n"
       "+    setattr(flashinfer_prefill, guard_name, lambda *_args, **_kwargs: False)\n")
if text.count(old) != 1:
    raise SystemExit("0136 phase-A edit anchor is not unique")
text = text.replace(old, "+    setattr(flashinfer_prefill, guard_name, lambda _kv_data_type: False)\n")
text = text.replace("@@ -164,6 +164,37 @@", "@@ -164,6 +164,35 @@")
text = text.replace("@@ -903,6 +932,7 @@", "@@ -903,6 +930,7 @@")
patch.write_text(text)
for name in (chain / "series").read_text().splitlines():
    if not (chain / name).is_file():
        raise SystemExit("missing series entry: " + name)
print("Local build context staged; 0136 restored to the served phase-A form")
PY

check_copy_sources() {
  python3 - "$CTX" "$1" <<'PY'
from pathlib import Path
import shlex
import sys
root, dockerfile = map(Path, sys.argv[1:])
for line in dockerfile.read_text().splitlines():
    if not line.startswith("COPY "):
        continue
    words = shlex.split(line)
    if words[1].startswith("--from="):
        continue
    for name in words[1:-1]:
        if not (root / name).exists():
            raise SystemExit("missing COPY source in " + dockerfile.name + ": " + name)
PY
}
for df in "$CTX/out/A/Dockerfile.v0310" "$CTX/Dockerfile.A_S-splitkv" "$CTX/Dockerfile.A_S-L1a-ubplan"; do
  check_copy_sources "$df"
done

# These pins are registry digests. RepoDigests must contain the exact name@digest reference.
# Resolve that reference's local .Id for FROM and asset runs: it can differ between image stores.
# Pull only by digest when needed; never retag or replace an existing user tag.
ensure_base() {
  local tag=$1 expected=$2 result_var=$3 ref repo_digests actual platform
  ref=${tag%:*}@$expected
  if [ "$DRY_RUN" = 1 ]; then
    log "If local RepoDigests lacks $ref, pull that digest reference; then verify RepoDigests and resolve its linux/amd64 local .Id:"
    print_cmd "${DOCKER_CMD[@]}" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$ref"
    print_cmd "${DOCKER_CMD[@]}" pull --platform linux/amd64 "$ref"
    print_cmd "${DOCKER_CMD[@]}" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$ref"
    print_cmd "${DOCKER_CMD[@]}" image inspect --format '{{.Id}}' "$ref"
    print_cmd "${DOCKER_CMD[@]}" image inspect --format '{{.Os}}/{{.Architecture}}' "$ref"
    printf -v "$result_var" '%s' "<local .Id for $ref>"
    return
  fi
  repo_digests=$("${DOCKER_CMD[@]}" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$ref" 2>/dev/null) || repo_digests=
  if ! grep -Fxq -- "$ref" <<< "$repo_digests"; then
    run "${DOCKER_CMD[@]}" pull --platform linux/amd64 "$ref" || die "could not pull $ref"
    repo_digests=$("${DOCKER_CMD[@]}" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$ref") || die "could not inspect pulled base $ref"
  fi
  grep -Fxq -- "$ref" <<< "$repo_digests" || die "$ref RepoDigests mismatch: expected exact reference is absent"
  actual=$("${DOCKER_CMD[@]}" image inspect --format '{{.Id}}' "$ref")
  [[ "$actual" =~ ^sha256:[0-9a-f]{64}$ ]] || die "$ref has an invalid local .Id: $actual"
  platform=$("${DOCKER_CMD[@]}" image inspect --format '{{.Os}}/{{.Architecture}}' "$ref")
  [ "$platform" = linux/amd64 ] || die "$ref is $platform; linux/amd64 is required"
  printf -v "$result_var" '%s' "$actual"
  log "BASE VERIFIED $ref local .Id=$actual"
}
ensure_base "$VLLM_TAG" "$VLLM_DIGEST" VLLM_ID
ensure_base "$CUDA_TAG" "$CUDA_DIGEST" CUDA_ID

SOURCE_ARCHIVE=$CTX/downloads/vllm.tar.gz
run curl --fail --location --show-error --proto '=https' --tlsv1.2 --retry 3 --output "$SOURCE_ARCHIVE" "$SOURCE_URL"
if [ "$DRY_RUN" = 1 ]; then
  log "Require VLLM_SOURCE_SHA256=$VLLM_SOURCE_SHA256 to equal the downloaded vLLM tarball's SHA256 before extraction."
  print_cmd sha256sum "$SOURCE_ARCHIVE"
else
  require_hash "$SOURCE_ARCHIVE" "$VLLM_SOURCE_SHA256" "vLLM source tarball"
fi
run tar -xzf "$SOURCE_ARCHIVE" -C "$CTX/vllm-v0.31.0" --strip-components=1 \
  "vllm-$VLLM_COMMIT/csrc" "vllm-$VLLM_COMMIT/cmake/patches"

ARCTIC_ARCHIVE=$CTX/downloads/arctic_inference-0.1.1.tar.gz
run curl --fail --location --show-error --proto '=https' --tlsv1.2 --retry 3 --output "$ARCTIC_ARCHIVE" "$ARCTIC_URL"
if [ "$DRY_RUN" = 1 ]; then
  log "Require Arctic sdist SHA256=$ARCTIC_SHA256 before pip wheel."
else
  require_hash "$ARCTIC_ARCHIVE" "$ARCTIC_SHA256" "Arctic sdist"
fi
# Install the hashed PyPI tools in this container's disposable writable layer, leaving the
# pinned base and final image unchanged. setup.py needs grpc_tools.protoc as well as CMake.
cp "$ARCTIC_BUILD_REQUIREMENTS" "$CTX/downloads/arctic-build-requirements.txt"
log "Prepare Arctic wheel: install pinned binary tools from PyPI, then build the verified sdist without isolation or dependency downloads."
if ! run "${DOCKER_CMD[@]}" run --rm --network=bridge --user 0 \
  -e PIP_CONFIG_FILE=/dev/null -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e ARCTIC_INFERENCE_PRECOMPILED_OPS=0 \
  -v "$CTX/downloads:/inputs:ro" -v "$CTX/assets/wheels:/wheels" \
  --entrypoint bash "$VLLM_ID" -euo pipefail -c '
    trap '\''chmod -R a+rwX /wheels'\'' EXIT
    python3 -m pip --isolated install --index-url https://pypi.org/simple \
      --require-hashes --only-binary=:all: --no-deps --no-cache-dir --force-reinstall \
      -r /inputs/arctic-build-requirements.txt
    python3 -m pip --isolated wheel --no-index --no-deps --no-build-isolation --no-cache-dir \
      --wheel-dir /wheels /inputs/arctic_inference-0.1.1.tar.gz
  '; then
  die "Arctic wheel preparation failed in the throwaway pinned-base container; check the hashed PyPI tool install and pip wheel output above"
fi
if [ "$DRY_RUN" != 1 ]; then
  compgen -G "$CTX/assets/wheels/arctic_inference-0.1.1-*.whl" >/dev/null || die "pip wheel produced no Arctic wheel"
fi
rm -rf -- "$CTX/downloads"

TOOLS_CHECK='for tool in patch g++ ninja; do command -v "$tool" >/dev/null || exit 1; done'
NEED_DEBS=0
if [ "$DRY_RUN" = 1 ]; then
  print_cmd "${DOCKER_CMD[@]}" run --rm --network=none --entrypoint bash "$VLLM_ID" -c "$TOOLS_CHECK"
  log "Only if the base lacks patch/g++/ninja, download the missing tools and dependencies from its apt repositories:"
  NEED_DEBS=1
elif ! "${DOCKER_CMD[@]}" run --rm --network=none --entrypoint bash "$VLLM_ID" -c "$TOOLS_CHECK"; then
  NEED_DEBS=1
fi
if [ "$NEED_DEBS" = 1 ]; then
  run "${DOCKER_CMD[@]}" run --rm --user 0 -v "$CTX/assets/debs:/debs" --entrypoint bash "$VLLM_ID" -ec '
    trap '\''chmod -R a+rwX /debs'\'' EXIT
    apt-get update
    apt-get -y --download-only --no-install-recommends -o Dir::Cache::archives=/debs install patch g++ ninja-build
    rm -rf /debs/partial /debs/lock
  '
fi

build() {
  local tag=$1 df=$2 logfile
  shift 2
  local cmd=("${DOCKER_CMD[@]}" build --network=none --pull=false)
  if [ "$NO_CACHE" = 1 ]; then cmd+=(--no-cache); fi
  cmd+=("$@" -f "$df" -t "$tag" "$CTX")
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DOCKER_BUILDKIT=0 '; print_cmd "${cmd[@]}"; return
  fi
  logfile=$LOG_DIR/${tag#*:}.log
  log "BUILD $tag (DOCKER_BUILDKIT=0)"
  if ! "${cmd[@]}" 2>&1 | tee "$logfile"; then die "build failed for $tag; see $logfile"; fi
}
if [ "$DRY_RUN" != 1 ]; then mkdir -p "$LOG_DIR"; fi
build "$T" "$CTX/out/A/Dockerfile.v0310" --build-arg VLLM_BASE="$VLLM_ID" \
  --build-arg CUDA_TOOLCHAIN="$CUDA_ID" --build-arg OVERLAY_JOBS="$OVERLAY_JOBS"
build "$S" "$CTX/Dockerfile.A_S-splitkv"
build "$FINAL" "$CTX/Dockerfile.A_S-L1a-ubplan"

ACTUAL_MANIFEST=$LOG_DIR/built-l1a-manifest.txt
CONTENT_REPORT=$LOG_DIR/content-report.txt
CONTENT_RC=0
if [ "$DRY_RUN" = 1 ]; then
  printf '%q ' "${DOCKER_CMD[@]}" run --rm -i --network=none --entrypoint python3 "$FINAL" - generate
  printf '< %q > %q\n' "$MANIFEST_TOOL" "$ACTUAL_MANIFEST"
  print_cmd python3 "$MANIFEST_TOOL" compare "$MANIFEST" "$ACTUAL_MANIFEST"
else
  "${DOCKER_CMD[@]}" image inspect "$FINAL" >/dev/null || die "final image is missing"
  "${DOCKER_CMD[@]}" run --rm -i --network=none --entrypoint python3 "$FINAL" - generate \
    < "$MANIFEST_TOOL" > "$ACTUAL_MANIFEST"
  python3 "$MANIFEST_TOOL" compare "$MANIFEST" "$ACTUAL_MANIFEST" | tee "$CONTENT_REPORT" || CONTENT_RC=$?
fi

IDENTITY_CODE=$(cat <<'PY'
from pathlib import Path
import torch, vllm, flashinfer
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch
assert vllm.__version__.startswith("0.31.0"), vllm.__version__
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda.startswith("13.0"), torch.version.cuda
assert flashinfer.__version__ == "0.7.0.post1", flashinfer.__version__
assert hasattr(torch.ops._C, "rotary_embedding")
assert hasattr(torch.ops._moe_C, "topk_softmax")
torch.ops.load_library("/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so")
assert hasattr(torch.ops.vllm_sm12x, "reshape_and_cache_nvfp4")
site = Path(vllm.__file__).parent.parent
torch.ops.load_library(str(site / "pcie_ipc_ar21/_C.so"))
assert hasattr(torch.ops.pcie_ipc_ar21_native, "all_reduce")
for marker in ("0136", "0147", "0148", "0154"):
    assert (Path("/opt/prs-markers") / marker).is_file(), marker
assert "0158" in Path("/opt/prs-markers/upstream").read_text()
fi = (site / "vllm/v1/attention/backends/flashinfer.py").read_text()
sp = (site / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py").read_text()
assert "lambda *_args, **_kwargs: False" in fi
assert "_sm12x_draft_ub_plan_seq_lens" in fi and "SM12X draft UB plan USED" in fi
assert "SM12X draft UB plan ACTIVE" in sp and "_sm12x_wait_draft_plan(self)" in sp
print("IDENTITY OK", vllm.__version__, torch.__version__, flashinfer.__version__)
PY
)
IDENTITY_RC=0
if [ "$CHECK" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    print_cmd "${DOCKER_CMD[@]}" run --rm --network=none --runtime nvidia --gpus all \
      -e CUDA_VISIBLE_DEVICES= --entrypoint python3 "$FINAL" -c "$IDENTITY_CODE"
  else
    "${DOCKER_CMD[@]}" run --rm --network=none --runtime nvidia --gpus all \
      -e CUDA_VISIBLE_DEVICES= --entrypoint python3 "$FINAL" -c "$IDENTITY_CODE" \
      2>&1 | tee "$LOG_DIR/identity.log" || IDENTITY_RC=$?
  fi
fi

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN DONE: local COPY inputs checked; source/wheel/debs are planned; final tag matches the launcher."
  log "Base local .Id placeholders are resolved during a real run; image content and identity have not been checked."
else
  log "SUMMARY: image=$FINAL content_exit=$CONTENT_RC identity_requested=$CHECK identity_exit=$IDENTITY_RC"
  log "Manifest: $ACTUAL_MANIFEST; comparison: $CONTENT_REPORT; build logs: $LOG_DIR"
  [ "$CONTENT_RC" = 0 ] || die "content verification failed"
  [ "$IDENTITY_RC" = 0 ] || die "identity check failed; see $LOG_DIR/identity.log"
  log "DONE: $FINAL matches the required served sources and inventory."
fi
