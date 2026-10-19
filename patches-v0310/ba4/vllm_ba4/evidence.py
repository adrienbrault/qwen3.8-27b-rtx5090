"""Strict serving PROFILE evidence contract, independent of torch/CUDA."""
import hashlib
import json
import math
from .measurement import positive

BASE_IMAGE = 'sha256:631a6f24ccca2f5470a3d0772d833670a16613c69bb4191663f718e473086f5e'
SERVING_PROTOCOL = 'L1a_BA4-serving-profile-v2'
SHARD_ORDER = 'beta[24*rank:24*(rank+1)],alpha[24*rank:24*(rank+1)]'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def image(value):
    return isinstance(value, str) and value.startswith('sha256:') and sha(value[7:])


def diagnostic_digest(report):
    return digest({k: v for k, v in report.items()
                   if k not in ('serving_evidence', 'verdict', 'reason')})


def serving_reason(report):
    evidence = report.get('serving_evidence', {})
    if evidence.get('protocol') != SERVING_PROTOCOL:
        return 'missing_serving_PROFILE'
    if (evidence.get('rank') != report['rank']
            or evidence.get('fingerprint') != report['fingerprint']
            or evidence.get('diagnostic_sha256') != diagnostic_digest(report)):
        return 'serving_capture_or_hardware_mismatch'
    binding = evidence.get('binding', {})
    if (binding.get('base_image_id') != BASE_IMAGE
            or not image(binding.get('candidate_image_id'))
            or binding['candidate_image_id'] == BASE_IMAGE
            or binding.get('code_sha256') != report['code_sha256']
            or binding.get('benchmark_sha256') != report['benchmark_sha256']
            or any(not sha(binding.get(k)) for k in
                   ('compiler_config_sha256', 'launch_config_sha256', 'source_archive_sha256'))):
        return 'serving_image_or_compiler_binding'
    trials = evidence.get('trials', [])
    if len(trials) != 3:
        return 'need_three_serving_PROFILE_pairs'
    trace_hashes = set()
    for trial in trials:
        totals = []
        for phase in ('control_before', 'candidate', 'control_after'):
            p = trial[phase]
            candidate = phase == 'candidate'
            if (p.get('mode') != 'PROFILE' or p.get('concurrency') != 1
                    or p.get('arm') != ('L1a_BA4' if candidate else 'A_S_L1a')
                    or p.get('image_id') != binding['candidate_image_id' if candidate else 'base_image_id']
                    or p.get('binding_sha256') != digest(binding)
                    or p.get('vllm_compile_capture') is not True
                    or p.get('outputs_correct') is not True
                    or p.get('changed_input_correct') is not True
                    or p.get('graph_replays_per_step') != 4
                    or p.get('pcie_pushes_per_step') != 138
                    or p.get('mtp_bf16_calls_per_step') != 15):
                return 'unverified_serving_graph'
            for kind in ('trace', 'probe', 'harness'):
                artifact = p[kind]
                if not artifact.get('path') or not sha(artifact.get('sha256')):
                    return 'missing_serving_provenance'
            if p['trace']['sha256'] in trace_hashes:
                return 'reused_serving_trace'
            trace_hashes.add(p['trace']['sha256'])
            layers = p['layers']
            if len(layers) != 48 or {l['prefix'] for l in layers} != {l['prefix'] for l in report['layers']}:
                return 'serving_prefixes'
            expected = {l['prefix']: l for l in report['layers']}
            total = 0
            for layer in layers:
                e = expected[layer['prefix']]
                if (layer.get('x_shape') != [4, 5120] or layer.get('logical_M') != 4
                        or layer.get('padded_M') != 4 or layer.get('x_stride') != e['x_stride']
                        or layer.get('x_offset') != e['capture_x_offsets']['4']
                        or layer.get('weight_shape') != [48, 5120]
                        or layer.get('weight_stride') != [5120, 1]
                        or layer.get('weight_sha256') != e['weight_sha256']
                        or layer.get('tp_rank') != report['rank'] or layer.get('tp_size') != 2
                        or layer.get('output_sizes') != [48, 48]
                        or layer.get('shard_order') != SHARD_ORDER
                        or layer.get('calls_per_step') != 1
                        or layer.get('route') != (report['selected'] if candidate else 'served_compiled_baseline')):
                    return 'serving_shape_stride_padding_or_layout'
                kernels = layer['kernels']
                minimum = (2 if report['selected'].startswith('triton_split') else
                           4 if report['selected'].startswith('torch_rows') else 1) if candidate else 1
                if (not kernels or any(not k.get('symbol') or not positive(k.get('count_per_step'))
                                       or not positive(k.get('us_per_step')) for k in kernels)
                        or sum(k['count_per_step'] for k in kernels) < minimum):
                    return 'serving_kernel_or_helper_costs'
                total += sum(k['us_per_step'] for k in kernels)
            if not positive(p.get('b1_us_per_step')) or abs(total - p['b1_us_per_step']) > 1e-6:
                return 'serving_cost_sum_mismatch'
            totals.append(total)
        saving = (totals[0] + totals[2]) / 2 - totals[1]
        if not math.isfinite(saving) or saving < 500:
            return 'serving_saving_below_500us'
    return None
