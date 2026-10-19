"""Dependency-free routing/admission policy; also exercised on the CPU host."""

import math
import statistics

from .measurement import TIMING, positive
from .evidence import SHARD_ORDER, serving_reason, sha

PROTOCOL = "L1a_BA4-v2"
PROFILE_PROTOCOL = "L1a_BA4-profile-qualified-v3"
SUFFIX = ".linear_attn.in_proj_ba"
WEIGHT_SHAPE = (48, 5120)
WEIGHT_STRIDE = (5120, 1)
CALLS = 48
MIN_SAVING_US = 500.0
CANDIDATES = (
    "torch_mm", "torch_rows_mm", "torch_rows_mv",
    "triton_direct", "triton_split128", "triton_split256", "triton_split512",
)
COMPILED_CHECKS = ("allocation_free", "output_aliases_buffer", "correct",
                   "capture_replay_correct", "changed_input_correct")


def weight_reason(prefix, shape, stride, dtype, device_type, capability,
                  tp_size, output_sizes, offset=0):
    if not prefix.endswith(SUFFIX):
        return "prefix"
    if tuple(shape) != WEIGHT_SHAPE:
        return "weight_shape"
    if tuple(stride) != WEIGHT_STRIDE or offset != 0:
        return "weight_stride_or_offset"
    if str(dtype) != "torch.bfloat16":
        return "weight_dtype"
    if device_type != "cuda" or tuple(capability) != (12, 0):
        return "device"  # Exact target SM120; do not infer SM121 compatibility.
    if tp_size != 2 or list(output_sizes) != [48, 48]:
        return "shards"  # Excludes the interleaved Qwen3-Next [96] route.
    return None


def input_reason(shape, stride, dtype, device, expected_device,
                 expected_stride, bias, requires_grad=False, offset=0):
    if bias is not None:
        return "bias"
    if tuple(shape) != (4, 5120):
        return "input_shape"
    if tuple(stride) != tuple(expected_stride):
        return "input_stride"
    if str(dtype) != "torch.bfloat16" or device != expected_device:
        return "input_dtype_or_device"
    if requires_grad or offset % 8:
        return "input_grad_or_alignment"
    return None


def valid_x_stride(stride):
    return (len(stride) == 2 and stride[1] == 1
            and stride[0] >= 5120 and stride[0] % 8 == 0)


def select(results):
    """Pick the fastest correct, graph-checked, allocation-free complete path.

    Ordinary torch paths win exact ties. Results are medians of measured sums,
    never 48 times the fastest/hottest single layer.
    """
    eligible = [r for r in results if r.get("name") in CANDIDATES
                and r.get("correct") is True and r.get("graph_ok") is True
                and r.get("allocation_free") is True
                and math.isfinite(r.get("us_per_step", math.inf))
                and r["us_per_step"] > 0]
    return min(eligible, key=lambda r: (r["us_per_step"],
               CANDIDATES.index(r["name"]))) if eligible else None


def paired_values(result):
    trials = result.get("paired_trials", [])
    if len(trials) != 3:
        raise ValueError("need three complete control/candidate/control trials")
    values = []
    for trial in trials:
        before, candidate, after = (trial[k] for k in
                                    ("control_before_us", "candidate_us", "control_after_us"))
        if not all(positive(v) for v in (before, candidate, after)):
            raise ValueError("invalid paired time")
        saving = (before + after) / 2 - candidate
        if not math.isfinite(saving):
            raise ValueError('nonfinite recomputed saving')
        values.append(saving)
    return values


def winning(result):
    try:
        savings = paired_values(result)
        costs = [t["candidate_us"] for t in result["paired_trials"]]
        return (result.get("paired_savings_us") == savings
                and result.get("trial_us_per_step") == costs
                and result.get("us_per_step") == statistics.median(costs)
                and all(s >= MIN_SAVING_US for s in savings))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def measurement_reason(report):
    replay = report.get('replay', {})
    required = {'distinct_B1_matrices': CALLS, 'B1_weight_bytes': CALLS*48*5120*2,
                'before_bytes_per_layer': 41_943_040, 'after_bytes_per_layer': 90_931_200,
                'stream_bytes_per_step': 6_377_963_520, 'timing': TIMING}
    if any(replay.get(k) != v for k, v in required.items()):
        return 'unverified_cold_replay_protocol'
    sanity = report.get('timing_validation', {})
    single, long = sanity.get('single_work_us', []), sanity.get('sixteen_work_us', [])
    if (len(single) != 3 or len(long) != 3 or not all(positive(v) for v in single + long)
            or statistics.median(long) <= 2 * statistics.median(single)):
        return 'unvalidated_outside_replay_timer'
    for m in ("4", "8", "12", "16", "32"):
        rows = report.get("measurements", {}).get(m, [])
        names = [r.get("name") for r in rows]
        if (len(names) != len(set(names)) or set(names) !=
                {"current", *CANDIDATES, "cublaslt_mm_probe"}):
            return "incomplete_other_M_or_control_rows"
        control_us = next(r for r in rows if r['name'] == 'current').get('us_per_step')
        if not positive(control_us):
            return 'invalid_current_control'
        for row in rows:
            if row["name"] != "current" and "us_per_step" not in row:
                if not row.get("error"):
                    return "unexplained_missing_measurement"
                continue
            if (not positive(row.get("us_per_step")) or row.get("timing") != TIMING
                    or not positive(row.get("b1_trace_us_per_step"))
                    or len(row.get("per_layer_us", [])) != CALLS
                    or not all(positive(v) for v in row["per_layer_us"])
                    or abs(sum(row["per_layer_us"]) - row["b1_trace_us_per_step"]) > 1e-6
                    or len(row.get("kernel_counts", [])) != CALLS
                    or not all(type(n) is int and n > 0 for n in row["kernel_counts"])
                    or not row.get("cuda_symbols")
                    or not sha(row.get("trace_sha256")) or not row.get("trace_path")
                    or not positive(row.get("us_per_call"))
                    or abs(row["us_per_call"] * CALLS - row["b1_trace_us_per_step"]) > 1e-6
                    or not positive(row.get("effective_weight_GBps"))):
                return "invalid_timing_or_trace_schema"
            if (not positive(row.get('all_kernel_us')) or not positive(row.get('profiled_replay_us'))
                    or row['b1_trace_us_per_step'] > row['all_kernel_us']
                    or row['all_kernel_us'] > row['profiled_replay_us'] * 1.1
                    or abs(row['effective_weight_GBps'] - CALLS*48*5120*2 /
                           row['b1_trace_us_per_step'] / 1000) > 1e-6
                    or any(type(row.get(k)) not in (int, float) or row[k] < 0
                           for k in ('max_abs_vs_fp64_bf16', 'max_rel_vs_fp64_bf16'))):
                return 'incomplete_trace_or_numerical_evidence'
            minimum = (2 if row['name'].startswith('triton_split') else
                       int(m) if row['name'].startswith('torch_rows') else 1)
            if any(n < minimum for n in row['kernel_counts']):
                return 'missing_reduction_or_rowwise_helpers'
            if row["name"] == "current":
                if (len(row.get("replay_samples_us", [])) != 3
                        or not all(positive(v) for v in row["replay_samples_us"])
                        or row["us_per_step"] != statistics.median(row["replay_samples_us"])
                        or type(row.get('real_M_capture')) is not bool
                        or (m == '4' and row['real_M_capture'] is not True)
                        or row.get("retained_graph_outputs_correct") is not True):
                    return "invalid_current_graph_control"
            else:
                if any(type(row.get(k)) not in (int, float) or row[k] < 0
                       for k in ('max_abs', 'max_rel')):
                    return 'missing_numerical_evidence'
                savings = paired_values(row)
                costs = [t["candidate_us"] for t in row["paired_trials"]]
                if any(abs(t[k] / control_us - 1) > .2 for t in row['paired_trials']
                       for k in ('control_before_us', 'control_after_us')):
                    return 'unstable_or_mismatched_current_control'
                if (type(row.get('transient_device_bytes')) is not int
                        or row['transient_device_bytes'] < 0
                        or row.get('allocation_free') != (row['transient_device_bytes'] == 0)
                        or row.get('packing_us') != 0
                        or row.get('synchronization_inside_timing') is not False
                        or row.get('scratch_bytes') != 40*int(m)*48*4*CALLS
                        or row.get('scratch_used') != row['name'].startswith('triton_split')):
                    return 'inconsistent_helper_or_allocation_evidence'
                if (row.get("paired_savings_us") != savings
                        or row.get("trial_us_per_step") != costs
                        or row["us_per_step"] != statistics.median(costs)
                        or any(type(row.get(k)) is not bool for k in
                               ("correct", "graph_ok", "allocation_free"))):
                    return "inconsistent_paired_measurement"
    return None


def admission_reason(report, fingerprint, code_sha, rank, benchmark_sha=None, profile_only=False):
    try:
        return _admission_reason(report, fingerprint, code_sha, rank, benchmark_sha, profile_only)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return "malformed_result"


def _admission_reason(report, fingerprint, code_sha, rank, benchmark_sha, profile_only):
    if report.get("protocol") == PROFILE_PROTOCOL:
        if profile_only or report.get("verdict") != "PROFILE_QUALIFIED":
            return "no_profile_qualified_result"
        # Reuse every diagnostic, fingerprint, code, benchmark and layer check.
        diagnostic = dict(report, protocol=PROTOCOL, verdict="PROVISIONAL")
        reason = _admission_reason(diagnostic, fingerprint, code_sha, rank, benchmark_sha, True)
        return reason or profile_qualified_reason(report)
    if report.get("protocol") != PROTOCOL or report.get("verdict") != ("PROVISIONAL" if profile_only else "WIN"):
        return "no_provisional_result" if profile_only else "no_WIN_result"
    if not finite_tree(report):
        return "nonfinite_result"
    if report.get("fingerprint") != fingerprint:
        return "hardware_or_software_changed"
    if report.get("code_sha256") != code_sha:
        return "implementation_changed"
    if benchmark_sha is None:
        from .common import benchmark_sha256
        benchmark_sha = benchmark_sha256()
    if report.get("benchmark_sha256") != benchmark_sha:
        return "benchmark_changed"
    if (report.get("rank") != rank or report.get("real_capture") is not True
            or not sha(report.get("capture_sha256"))
            or report.get("probe_mode") != "eager_diagnostic"):
        return "rank_or_capture"
    if report.get("baseline_impl") != "default_unquantized_gemm":
        return "unsupported_baseline"
    if (report.get("timing") != TIMING
            or report.get("cublaslt_requires_per_call_pin") is not False
            or any(report.get("compiled_check", {}).get(k) is not True
                   for k in COMPILED_CHECKS)):
        return "unverified_timing_or_compiled_path"
    entries = report.get("layers", [])
    if (len(entries) != CALLS or len({e.get("prefix") for e in entries}) != CALLS
            or any(not e.get("prefix", "").endswith(SUFFIX)
                   or not valid_x_stride(e.get("x_stride", []))
                   or e.get("weight_stride") != [5120, 1]
                   or e.get("shard_order") != SHARD_ORDER
                   or not sha(e.get("weight_sha256"))
                   or type(e.get("capture_x_offsets", {}).get("4")) is not int
                   or e["capture_x_offsets"]["4"] < 0 or e["capture_x_offsets"]["4"] % 8
                   for e in entries)):
        return "incomplete_48_layer_capture"
    reason = measurement_reason(report)
    if reason:
        return reason
    rows = report["measurements"]["4"]
    best = select(rows)
    if (best is None or report.get("selected") != best["name"]
            or best.get("compiled_selected") is not True or not winning(best)):
        return "no_qualified_fastest_path"
    lt = next(r for r in rows if r["name"] == "cublaslt_mm_probe")
    if lt.get("us_per_step", math.inf) < best["us_per_step"]:
        return "cublaslt_requires_per_call_pin"
    if not profile_only:
        return serving_reason(report)
    return None


def profile_qualified_reason(report):
    """v3 is admission from repeated complete PROFILE pairs, never fidelity."""
    from .evidence import BASE_IMAGE, digest, image
    evidence = report.get('profile_evidence', {})
    if (evidence.get('protocol') != PROFILE_PROTOCOL
            or evidence.get('diagnostic_sha256') != digest({k: v for k, v in report.items()
                if k not in ('profile_evidence', 'serving_evidence')})):
        return 'profile_diagnostic_changed'
    binding = report.get('serving_evidence', {}).get('binding', {})
    if (binding.get('base_image_id') != BASE_IMAGE
            or not image(binding.get('candidate_image_id'))
            or binding['candidate_image_id'] == BASE_IMAGE
            or binding.get('code_sha256') != report['code_sha256']
            or binding.get('benchmark_sha256') != report['benchmark_sha256']
            or any(not sha(binding.get(k)) for k in
                ('compiler_config_sha256', 'launch_config_sha256', 'source_archive_sha256'))):
        return 'profile_image_or_compiler_binding'
    inputs = evidence.get('provisional_inputs', {})
    if (set(inputs) != {'provisional-rank0.json', 'provisional-rank1.json'}
            or not all(sha(h) for h in inputs.values())):
        return 'profile_provisional_inputs'
    migration = evidence.get('migration', {})
    if (migration.get('target_code_sha256') != report['code_sha256']
            or migration.get('target_image_id') != binding['candidate_image_id']
            or migration.get('source_code_sha256') not in
                (report['code_sha256'], 'f21ed08a57291f0ba0501d45d7e8cbb5b863fb5c68dc6c0e2155cd5e54d517ea')
            or not image(migration.get('source_image_id'))):
        return 'unverified_profile_code_migration'
    pairs = evidence.get('pairs', [])
    if len(pairs) < 2:
        return 'need_two_complete_PROFILE_pairs'
    paths, manifests, units, traces = set(), set(), set(), set()
    for pair in pairs:
        if (not isinstance(pair.get('result_dir'), str) or not pair['result_dir'].startswith('/')
                or pair['result_dir'] in paths or pair.get('unit_identity') in units
                or not sha(pair.get('unit_identity'))):
            return 'reused_PROFILE_unit'
        paths.add(pair['result_dir']); units.add(pair['unit_identity'])
        manifest = pair.get('manifest', {})
        if (not manifest or not all(isinstance(n, str) and n and not n.startswith('/')
                and '..' not in n.split('/') and sha(h) for n, h in manifest.items())
                or pair.get('manifest_sha256') != digest(manifest)
                or pair['manifest_sha256'] in manifests):
            return 'profile_manifest_changed'
        manifests.add(pair['manifest_sha256'])
        if (pair.get('order') != ['A_S_L1a', 'L1a_BA4']
                or pair.get('provisional_inputs') != inputs
                or pair.get('binding') != dict(binding,
                    code_sha256=migration['source_code_sha256'],
                    candidate_image_id=migration['source_image_id'],
                    source_archive_sha256=pair.get('binding', {}).get('source_archive_sha256'))
                or not sha(pair.get('binding', {}).get('source_archive_sha256'))
                or pair.get('gates') != 'PASS'
                or pair.get('aot_cache_decimal_nonce_dirs') != 0):
            return 'incomplete_PROFILE_pair'
        timings = [pair.get('unprofiled')] + [pair.get('profiled', {}).get(str(r)) for r in (0, 1)]
        for timing in timings:
            if (not isinstance(timing, dict) or not positive(timing.get('control_us'))
                    or not positive(timing.get('candidate_us'))
                    or timing.get('saving_us') != timing['control_us'] - timing['candidate_us']
                    or timing['saving_us'] < MIN_SAVING_US):
                return 'insufficient_PROFILE_saving'
        for r in (0,1):
            for key in ('control_trace_sha256','candidate_trace_sha256'):
                h=pair['profiled'][str(r)].get(key)
                if not sha(h) or h in traces:
                    return 'reused_or_missing_PROFILE_trace'
                traces.add(h)
        worker = pair.get('workers', {}).get(str(report['rank']), {})
        if (worker.get('kernel') != report['selected']
                or worker.get('prefixes') != sorted(e['prefix'] for e in report['layers'])
                or not isinstance(worker.get('pid'), str) or not worker['pid'].isdigit()):
            return 'profile_worker_or_kernel_changed'
    return None


def profile_binding_reason(binding, report):
    """Both host and worker bind the v3 source inventory to this rank report."""
    try:
        pairs = report['profile_evidence']['pairs']
        if (set(binding) != {'protocol', 'binding', 'fingerprints', 'sources', 'provisional_inputs'}
                or binding['protocol'] != PROFILE_PROTOCOL
                or binding['binding'] != report['serving_evidence']['binding']
                or binding['fingerprints'][report['rank']] != report['fingerprint']
                or len(binding['fingerprints']) != 2
                or binding['fingerprints'][0]['device_uuid'] == binding['fingerprints'][1]['device_uuid']
                or binding['provisional_inputs'] != report['profile_evidence']['provisional_inputs']
                or binding['sources'] != [{k: p[k] for k in
                    ('result_dir', 'unit_identity', 'manifest_sha256', 'manifest')} for p in pairs]):
            return 'profile_launch_binding_changed'
    except (AttributeError, KeyError, TypeError, IndexError):
        return 'malformed_profile_binding'
    return None
