#!/usr/bin/env python3
"""Print an isolated GPU-screen command. Never executes it or inspects a host."""
import argparse
import json
from pathlib import Path
import shlex
import uuid

CONTROL = 'vllm-qwen38:v0320-native-r1'
BA4 = 'vllm-qwen38:v0320-native-ba4-r1'


def command(model, state, ba4=False, dynamic_mtp=False):
    if not model.is_absolute() or not state.is_absolute():
        raise ValueError('model and candidate state paths must be absolute')
    # New per-engine state prevents either image from writing the daily's tier/cache.
    if state.name != 'v032-candidate':
        raise ValueError('state directory must end in v032-candidate')
    # Each generated command owns fresh tiers and compile caches. vLLM sees the
    # same /model container path for different host checkpoints, so its internal
    # namespace alone cannot establish checkpoint or engine identity.
    run_state = state / uuid.uuid4().hex
    transfer = {'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
                'kv_connector_extra_config': {'spec_name': 'TieringOffloadingSpec',
                 'cpu_bytes_to_use': 17179869184, 'offload_prompt_only': True,
                 'secondary_tiers': [{'type': 'fs', 'root_dir': '/l2',
                   'n_read_threads': 16, 'n_write_threads': 4, 'max_capacity_gb': 300,
                   'evict_scope': 'root', 'min_free_gb': 10}]}}
    env = {'VLLM_USE_V2_MODEL_RUNNER': '1',
           'VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV': '1',
           'VLLM_SM12X_PCIE_IPC_AR': '1', 'VLLM_SM12X_PCIE_IPC_MTP': '1',
           'VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC': '0',
           'VLLM_ALLREDUCE_USE_FLASHINFER': '0', 'VLLM_ALLREDUCE_USE_SYMM_MEM': '0',
           'VLLM_BATCH_INVARIANT': '0', 'VLLM_SM12X_BA_M4': '0',
           'VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE': '536870912',
           'TORCHINDUCTOR_COMPILE_THREADS': '8', 'MAX_JOBS': '4',
           'FLASHINFER_NUM_COMPILE_JOBS': '4', 'PYTHONHASHSEED': '0',
           'CUDA_MODULE_LOADING': 'LAZY'}
    cmd = ['docker', 'run', '--name', 'vllm-v032-candidate', '--restart', 'no',
           '--runtime', 'nvidia', '--gpus', 'all', '--ipc', 'host',
           '--memory', '52g', '--memory-swap', '52g',
           '-p', '127.0.0.1:18029:8000', '--entrypoint', 'python3',
           '-v', str(model) + ':/model:ro', '-v', str(run_state / 'l2') + ':/l2',
           '-v', str(run_state / 'cache') + ':/root/.cache']
    for key, value in env.items():
        cmd += ['-e', key + '=' + value]
    spec = {'method': 'qwen3_5_mtp', 'num_speculative_tokens': 3,
            'disable_eagle_block_drop': True}
    if dynamic_mtp:
        spec['num_speculative_tokens_per_batch_size'] = [[1, 2, 3], [3, 4, 2], [5, 16, 1]]
    cmd += [BA4 if ba4 else CONTROL, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', '/model', '--served-model-name', 'qwen3.8-27b-candidate',
            '--tensor-parallel-size', '2', '--attention-backend', 'FLASHINFER', '--kv-cache-dtype', 'nvfp4',
            '--kv-cache-memory-bytes', '15900000000', '--gpu-memory-utilization', '0.88',
            '--max-model-len', '262144', '--max-num-seqs', '16',
            '--max-num-batched-tokens', '8192', '--enable-prefix-caching',
            '--mamba-backend', 'triton', '--gdn-prefill-backend', 'flashinfer',
            '--mamba-cache-mode', 'align', '--mamba-ssm-cache-dtype', 'bfloat16',
            '--offload-backend', 'uva', '--cpu-offload-gb', '1',
            '--cpu-offload-params', 'embed_tokens', '--enable-batch-sharded-sampling',
            '--speculative-config', json.dumps(spec),
            '--compilation-config', json.dumps({'max_cudagraph_capture_size': 320,
                'pass_config': {'fuse_allreduce_rms': False}}),
            '--kv-transfer-config', json.dumps(transfer),
            '--limit-mm-per-prompt', '{"image":32,"video":0}',
            '--default-chat-template-kwargs', '{"preserve_thinking":true,"reasoning_effort":"medium"}',
            '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
            '--tool-call-parser', 'qwen3_xml', '--enable-prompt-tokens-details',
            '--override-generation-config', '{"temperature":0.6,"top_p":0.95,"top_k":20}']
    return cmd


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('/srv/qwen5090/models/qwen3.8-27b-nvidia-nvfp4'))
    p.add_argument('--state', type=Path, default=Path('/srv/qwen5090/v032-candidate'))
    p.add_argument('--ba4-image', action='store_true', help='integration image; BA4 remains disabled')
    p.add_argument('--dynamic-mtp', action='store_true', help='K=3 at 1–2 requests, K=2 at 3–4, K=1 at 5–16')
    a = p.parse_args()
    print(shlex.join(command(a.model, a.state, a.ba4_image, a.dynamic_mtp)))
