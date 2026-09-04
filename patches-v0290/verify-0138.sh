#!/usr/bin/env bash
# Only reading/copying, patch --dry-run, and py_compile. No package imports.
# Run beside BRIEF21.md; retain the delivered .work21/new review sources.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPYCACHEPREFIX="$PWD/.work21/pycache"
python3 - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd()
scratch = root / '.work21' / 'verification-base'
scratch.mkdir(parents=True, exist_ok=True)
for source in (root / 's-image').rglob('*'):
    if not source.is_file() or source.suffix not in ('.py', '.orig'):
        continue
    section, *parts = source.relative_to(root / 's-image').parts
    if section == 'fi0616':
        target = Path('flashinfer', *parts)
    elif section == 'vllm-comm':
        if parts == ['parallel_state.py']:
            target = Path('vllm/distributed/parallel_state.py')
        elif parts == ['cuda_graph.py']:
            target = Path('vllm/compilation/cuda_graph.py')
        elif parts[0] == 'spec_decode':
            target = Path('vllm/v1', *parts)
        else:
            target = Path('vllm/distributed/device_communicators', *parts)
    elif section == 'vllm-worker':
        if parts[0] == 'dflash':
            target = Path('vllm/v1/worker/gpu/spec_decode', *parts)
        else:
            target = {
                'model_runner.py': Path('vllm/v1/worker/gpu/model_runner.py'),
                'gpu_worker.py': Path('vllm/v1/worker/gpu_worker.py'),
                'cudagraph_dispatcher.py': Path('vllm/v1/cudagraph_dispatcher.py'),
                'envs.py': Path('vllm/envs.py'),
            }[parts[0]]
    else:
        continue
    destination = scratch / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
PY
patch --dry-run --batch --forward -p1 --fuzz=0 \
    -d .work21/verification-base < 0138-pcie-ipc-all-reduce-v0290.diff
python3 -m py_compile \
    pcie_ipc_ar21/__init__.py pcie_ipc_ar21/build.py \
    pcie_ipc_ar21/fixed_config.py pcie_ipc_ar21/workspace.py \
    .work21/new/vllm/distributed/device_communicators/cuda_communicator.py \
    .work21/new/vllm/distributed/device_communicators/pcie_ipc_all_reduce.py \
    .work21/new/vllm/distributed/parallel_state.py \
    .work21/new/vllm/v1/worker/gpu_worker.py \
    .work21/new/vllm/v1/worker/gpu/model_runner.py \
    .work21/new/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
echo '0138: zero-fuzz patch dry-run and py_compile passed; native build/GPU/replay untested'
