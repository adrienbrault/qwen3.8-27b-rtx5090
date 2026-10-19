#!/usr/bin/env python3
"""N1 cold 48-layer replay, one GPU, hard wall limit <3 minutes.

Run on each saved TP rank separately. Eager/synthetic diagnostics can never WIN or authorize normal ACTIVE.
Timing events are OUTSIDE capture around whole replays; every projection
is separated from its next reuse by representative large target weight reads.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "L1a_BA4"))
from vllm_ba4.policy import (CALLS, CANDIDATES, PROTOCOL, select, winning,
                            valid_x_stride, weight_reason)
from vllm_ba4.measurement import TIMING, replay_us, trace_layers, validate_replay_timer

# The packet's input_reason accepts exactly (4, 5120); other M rows only
# characterize the underlying kernels and must never select a serving route.
ROUTE_M = 4
INFORMATIONAL_MS = (8, 12, 16, 32)


def select_serving_route(measurements):
    return select(measurements[str(ROUTE_M)])


def compiled_layer(name, prefix, x, weight, out, partial):
    """Build persistent serving state before entering Dynamo, as at model load."""
    from types import SimpleNamespace
    from vllm_ba4.runtime import State
    return SimpleNamespace(weight=weight, prefix=prefix,
                           _sm12x_ba4=State(name, tuple(x.stride()), weight, out, partial))


def compiled_route(layer, x, weight_offset, input_offset):
    from vllm_ba4.runtime import try_apply
    return try_apply(layer, x, None, weight_offset=weight_offset, input_offset=input_offset)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def error_stats(actual, reference):
    import torch
    a, b = actual.double(), reference.double()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return {"max_abs": sys.float_info.max, "max_rel": sys.float_info.max,
                "bit_equal": False, "close": False, "nonfinite": True}
    d = (a - b).abs()
    return {"max_abs": d.max().item(),
            "max_rel": (d / b.abs().clamp_min(1e-6)).max().item(),
            "bit_equal": bool(torch.equal(actual, reference)),
            "close": bool(torch.allclose(a, b, rtol=1 / 64, atol=1 / 256)),
            "nonfinite": False}


def elementwise_error_comparison(actual, current, reference):
    """No element may spend another element's baseline error allowance.

    Absolute FP64-to-BF16 error is compared at each identical reference element,
    without slack. Dividing both errors by that element's same |ref| (or the
    same 1e-6 floor) gives the identical relative-error ordering, even near zero.
    Maxima alone lose this correspondence and are diagnostic only.
    """
    import torch
    a, b, r = actual.double(), current.double(), reference.double()
    nonfinite = not bool(torch.isfinite(a).all() & torch.isfinite(b).all()
                         & torch.isfinite(r).all())
    if nonfinite:
        count, excess = a.numel(), sys.float_info.max
    else:
        increase = (a - r).abs() - (b - r).abs()
        count = int((increase > 0).sum().item())
        excess = increase.clamp_min(0).max().item()
    return {"metric": "abs_error_vs_fp64_bf16_per_element", "nonfinite": nonfinite,
            "worse_elements": count, "max_abs_error_increase": excess}


def record_correctness(path, report, owner, stage, outputs, references, prefixes,
                       trace, baseline_outputs=None, captured_outputs=None):
    """Persist every call and its kernel evidence before any numerical rejection.

    The served baseline defines the comparison: FP64 tolerance is diagnostic
    for it, with only non-finite values fatal (plus captured-output equality).
    Finite candidates pass iff FP64-close OR elementwise no worse than current;
    equality with finite current always passes. Captured equality is unchanged.
    """
    calls = []
    for i, (actual, reference, prefix) in enumerate(zip(outputs, references, prefixes)):
        ref = error_stats(actual, reference)
        call = {"call": i, "prefix": prefix, "vs_fp64_bf16": ref}
        passed = not ref["nonfinite"]
        if baseline_outputs is not None:
            base = error_stats(actual, baseline_outputs[i])
            base_ref = error_stats(baseline_outputs[i], reference)
            comparison = elementwise_error_comparison(actual, baseline_outputs[i], reference)
            no_worse = not comparison["nonfinite"] and comparison["worse_elements"] == 0
            call.update(vs_current=base, current_vs_fp64_bf16=base_ref,
                        elementwise_vs_current=comparison, no_worse_than_current=no_worse)
            passed = (passed and not base["nonfinite"] and not base_ref["nonfinite"]
                      and (ref["close"] or no_worse or base["bit_equal"]))
        if captured_outputs is not None and captured_outputs[i] is not None:
            captured = error_stats(actual, captured_outputs[i])
            call["vs_captured_served"] = captured
            passed = passed and captured["bit_equal"]
        call["passed"] = bool(passed)
        calls.append(call)
    if not calls or len(calls) != len(outputs) or len(calls) != len(references):
        raise ValueError("incomplete correctness inputs")
    result = {"passed": all(c["passed"] for c in calls), "per_call": calls,
              "cuda_symbols": trace["cuda_symbols"],
              "trace_path": trace["trace_path"], "trace_sha256": trace["trace_sha256"]}
    owner.setdefault("correctness", {})[stage] = result
    write_report(path, report)
    if baseline_outputs is None and not result["passed"]:
        if any(c["vs_fp64_bf16"]["nonfinite"] for c in calls):
            raise ValueError("original dispatch or independent reference is non-finite")
        raise ValueError("current dispatch differs from captured served output")
    return result


def print_measurement(m, row):
    if "us_per_step" not in row:
        print("M=%d %-19s unavailable: %s" %
              (m, row["name"], row.get("error", "not qualified")), flush=True)
        return
    print("M=%d %-19s %8.3f B1 trace us/call %9.3f whole replay us/step %8.2f logical GB/s "
          "max_abs=%g max_rel=%g" %
          (m, row["name"], row["us_per_call"], row["us_per_step"],
           row["effective_weight_GBps"], row.get("max_abs", 0), row.get("max_rel", 0)), flush=True)


def worker(args, report):
    import torch
    from vllm_ba4.common import benchmark_sha256, code_sha256, fingerprint, tensor_sha256
    from vllm_ba4.kernels import stream_weights
    from vllm_ba4.runtime import ba4_out

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("one CUDA GPU is required; no measurement made")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("this arm is restricted to SM120")
    report.update(code_sha256=code_sha256(), benchmark_sha256=hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest(), timing=TIMING,
        serving_route_m=[ROUTE_M], selection_m=ROUTE_M, informational_m=list(INFORMATIONAL_MS))
    if report['benchmark_sha256'] != benchmark_sha256():
        raise ValueError('benchmark differs from the packaged measurement implementation')
    deadline = time.monotonic() + args.seconds - 10
    report['timing_validation'] = validate_replay_timer(torch)

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("measurement budget exhausted; keep partial JSON, no activation")

    if args.capture:
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        captured_settings = payload["fingerprint"]
        # The model runner can set these beyond torch's process defaults.
        # Reproduce them before testing the complete hardware/software binding.
        torch.set_float32_matmul_precision(captured_settings["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = captured_settings[
            "bf16_reduced_precision_reduction"]
        preference = captured_settings["blas_preference"].lower()
        library = next((v for v in ("cublaslt", "cublas", "default") if v in preference), None)
        if library is None:
            raise ValueError("unsupported captured BLAS preference: " + preference)
        torch.backends.cuda.preferred_blas_library(library)
        report["fingerprint"] = fingerprint(device)
        if payload["rank"] != args.rank or payload["fingerprint"] != report["fingerprint"]:
            raise ValueError("capture rank/hardware/software/BLAS settings mismatch")
        report.update(real_capture=payload.get("real_capture") is True,
                      baseline_impl=payload.get("baseline_impl"),
                      capture_path=str(args.capture), probe_mode=payload.get("probe_mode"),
                      capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                      captured_matmul_settings_replayed=True)
        entries = payload["layers"]
    else:
        torch.manual_seed(271828 + args.rank)
        report.update(real_capture=False, baseline_impl="default_unquantized_gemm",
                      fingerprint=fingerprint(device))
        entries = [{"prefix": "model.layers.%d.linear_attn.in_proj_ba" % i,
                    "weight": torch.randn(48, 5120).mul_(5120 ** -.5).bfloat16(),
                    "weight_stride": [5120, 1], "weight_offset": 0,
                    "tp_size": 2, "tp_rank": args.rank, "output_sizes": [48, 48],
                    "shard_order": "synthetic_beta24_alpha24", "x": {}, "y": {},
                    "x_stride": {"4": [5120, 1]}, "x_offset": {"4": 0}}
                   for i in range(CALLS)]
    if len(entries) != CALLS or len({e["prefix"] for e in entries}) != CALLS:
        raise ValueError("need 48 distinct, served layer matrices")
    weights = []
    report["layers"] = []
    for e in entries:
        w = e["weight"]
        reason = weight_reason(e["prefix"], w.shape, e["weight_stride"], w.dtype,
                               "cuda", (12, 0), e["tp_size"], e["output_sizes"],
                               e["weight_offset"])
        if reason or e["tp_rank"] != args.rank or not valid_x_stride(e["x_stride"]["4"]):
            raise ValueError("unsupported captured layer: %s %s" % (e["prefix"], reason))
        if report["real_capture"] and ("4" not in e["x"] or "4" not in e["y"]):
            raise ValueError("missing real M4 input/output")
        weights.append(w.to(device))
        report["layers"].append({"prefix": e["prefix"],
                                  "weight_sha256": tensor_sha256(w),
                                  "x_stride": e["x_stride"]["4"],
                                  "observed_M": sorted(e["x"], key=int),
                                  "capture_x_strides": e["x_stride"],
                                  "capture_x_offsets": e["x_offset"],
                                  "weight_stride": e["weight_stride"],
                                  "shard_order": e["shard_order"]})
    write_report(args.json, report)
    # Per GDN layer: FP8 qkvz 8192*5120 before BA; FP8 out 5120*3072 and
    # representative NVFP4 gate/up+down (75,202,560 bytes) after BA. These
    # are distinct allocations for all 48 layers (~6.38 GB), not a hot ring.
    before_bytes = 8192 * 5120
    after_bytes = 5120 * 3072 + 75_202_560
    streams = [(torch.randint(0, 256, (before_bytes,), dtype=torch.uint8, device=device),
                torch.randint(0, 256, (after_bytes,), dtype=torch.uint8, device=device))
               for _ in range(CALLS)]
    sink = torch.empty(((max(before_bytes, after_bytes) + 4095) // 4096,),
                       dtype=torch.int32, device=device)
    stream_weights(streams[0][0], sink)
    stream_weights(streams[0][1], sink)
    weight_bytes = CALLS * 48 * 5120 * 2
    report["replay"] = {"distinct_B1_matrices": CALLS, "B1_weight_bytes": weight_bytes,
                         "stream_bytes_per_step": CALLS * (before_bytes + after_bytes),
                         "before_bytes_per_layer": before_bytes,
                         "after_bytes_per_layer": after_bytes,
                         "stream_kind": "unique random uint8 full reads, surrogate target weights",
                         "timing": TIMING,
                         "decision_scope": "whole replay including streams; B1 cost from separate trace",
                         "relative_error_denominator_floor": 1e-6,
                         "baseline_fp64_policy": "record errors; reject non-finite; require captured equality",
                         "candidate_fp64_policy": "finite AND (FP64-close OR elementwise no-worse than current OR equal to current); finite equality passes even when current is not FP64-close",
                         "rtol": 1 / 64, "atol": 1 / 256}
    write_report(args.json, report)

    def make_graph(xs, outs, partials, name, override=None):
        def call(i):
            if override is not None:
                return override(i)
            if name == "current":
                # Exactly dispatch_unquantized_gemm(auto)'s default on CUDA.
                return torch.nn.functional.linear(xs[i], weights[i], None)
            route = "torch_mm" if name == "cublaslt_mm_probe" else name
            ba4_out(xs[i], weights[i], outs[i], partials[i], route, "")
            return outs[i]
        for i in range(CALLS):
            call(i)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        retained = []
        with torch.cuda.graph(graph):
            for i in range(CALLS):
                stream_weights(streams[i][0], sink)
                retained.append(call(i))
                stream_weights(streams[i][1], sink)
        graph.replay()
        torch.cuda.synchronize()
        return graph, retained

    def sample(g):
        check_deadline()
        return replay_us(torch.cuda, g[0])

    def profile_graph(g, label):
        check_deadline()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            profiled_replay_us = replay_us(torch.cuda, g[0])
        path = args.json.with_suffix("." + label + ".trace.json")
        prof.export_chrome_trace(str(path))
        result = trace_layers(json.loads(path.read_text())["traceEvents"])
        if result['all_kernel_us'] > profiled_replay_us * 1.1:
            raise ValueError('independent trace kernel sum exceeds outside replay interval')
        result['profiled_replay_us'] = profiled_replay_us
        result.update(trace_path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return result

    def profile_calls(action, label):
        # Capture the actual eager/compiled correctness launches, including
        # rejected candidates; graph capture can choose different kernels.
        check_deadline()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            outputs = action()
            torch.cuda.synchronize()
        path = args.json.with_suffix("." + label + ".trace.json")
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        trace = {"cuda_symbols": sorted({e["name"] for e in events if e.get("cat") == "kernel"}),
                 "trace_path": str(path), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if not trace["cuda_symbols"]:
            raise ValueError("correctness profiler did not expose CUDA kernel symbols")
        return outputs, trace

    def inputs(m):
        key = str(m)
        xs, outs, partials, references = [], [], [], []
        for i, e in enumerate(entries):
            stride = e["x_stride"].get(key, e["x_stride"]["4"])
            offset = e["x_offset"].get(key, 0)
            storage = torch.empty(offset + (m - 1) * stride[0] + 5120,
                                  dtype=torch.bfloat16, device=device)
            x = storage.as_strided((m, 5120), stride, offset)
            if key in e["x"]:
                x.copy_(e["x"][key])
            elif "4" in e["x"]:
                x.copy_(e["x"]["4"].repeat((m + 3) // 4, 1)[:m])
            else:
                x.normal_()
            xs.append(x)
            outs.append(torch.empty((m, 48), dtype=x.dtype, device=device))
            partials.append(torch.empty((40, m, 48), dtype=torch.float32, device=device))
            # FP64 computes the same BF16 products with enough precision to
            # independently check the FP32 reduction and the final BF16 cast.
            references.append((x.double() @ weights[i].double().t()).bfloat16())
        return xs, outs, partials, references

    for m in (ROUTE_M, *INFORMATIONAL_MS):
        check_deadline()
        key = str(m)
        scope = "serving_route" if m == ROUTE_M else "informational"
        print("M=%d scope=%s; serving route accepts M=%d only" % (m, scope, ROUTE_M), flush=True)
        xs, outs, partials, references = inputs(m)
        base = make_graph(xs, outs, partials, "current")
        base_samples = [sample(base) for _ in range(3)]
        base_trace = profile_graph(base, "M%d-current" % m)
        current_outputs = [base[1][i].clone() for i in range(CALLS)]
        baseline_checks = [error_stats(current_outputs[i], references[i]) for i in range(CALLS)]
        baseline = {"name": "current", "measurement_scope": scope,
                    "us_per_step": statistics.median(base_samples),
                    "replay_samples_us": base_samples, "timing": TIMING,
                    "us_per_call": base_trace["b1_trace_us_per_step"] / CALLS,
                    "effective_weight_GBps": weight_bytes / base_trace["b1_trace_us_per_step"] / 1000,
                    "retained_graph_outputs_correct": False,
                    "real_M_capture": all(key in e["x"] for e in entries),
                    "max_abs_vs_fp64_bf16": max(e["max_abs"] for e in baseline_checks),
                    "max_rel_vs_fp64_bf16": max(e["max_rel"] for e in baseline_checks),
                    **base_trace}
        if m == 4:
            report["baseline_cuda_symbols"] = base_trace["cuda_symbols"]
            report["wmma128x1_seen"] = any("wmma" in symbol.lower() and "16x16_128x1" in symbol
                                                for symbol in base_trace["cuda_symbols"])
        rows = report["measurements"][key] = [baseline]
        prefixes = [e["prefix"] for e in entries]
        captured = [e["y"][key].to(device) if report["real_capture"] and key in e["y"] else None
                    for e in entries]
        record_correctness(args.json, report, baseline, "graph", current_outputs,
                           references, prefixes, base_trace, captured_outputs=captured)
        baseline["retained_graph_outputs_correct"] = True
        write_report(args.json, report)
        print_measurement(m, baseline)
        for name in (*CANDIDATES, "cublaslt_mm_probe"):
            check_deadline()
            row = {"name": name, "measurement_scope": scope, "correct": False, "graph_ok": False,
                   "allocation_free": False}
            rows.append(row)
            blas_original = torch.backends.cuda.preferred_blas_library()
            route = "torch_mm" if name == "cublaslt_mm_probe" else name
            try:
                if name == "cublaslt_mm_probe":
                    # PyTorch exposes a library preference, not a per-call
                    # cuBLASLt algorithm pin. Capture it for comparison only;
                    # never change the serving process's global BLAS policy.
                    torch.backends.cuda.preferred_blas_library("cublaslt")
                    row["deployable"] = False
                    row["note"] = "advisory library preference; per-call pin required to deploy"
                ba4_out(xs[0], weights[0], outs[0], partials[0], route, "")
                def eager_calls():
                    for i in range(CALLS):
                        ba4_out(xs[i], weights[i], outs[i], partials[i], route, "")
                    return outs
                _, eager_trace = profile_calls(eager_calls, "M%d-%s-eager" % (m, name))
                checked = record_correctness(args.json, report, row, "eager", outs,
                                             references, prefixes, eager_trace,
                                             baseline_outputs=current_outputs)
                checks_base = [c["vs_current"] for c in checked["per_call"]]
                checks_ref = [c["vs_fp64_bf16"] for c in checked["per_call"]]
                row.update(correct=checked["passed"],
                           max_abs=max(e["max_abs"] for e in checks_base),
                           max_rel=max(e["max_rel"] for e in checks_base),
                           bit_equal=all(e["bit_equal"] for e in checks_base),
                           max_abs_vs_fp64_bf16=max(e["max_abs"] for e in checks_ref),
                           max_rel_vs_fp64_bf16=max(e["max_rel"] for e in checks_ref))
                if not row["correct"]:
                    row["error"] = "numerical tolerance or baseline error bound failed"
                    continue
                # Check all layers after warmup, including library workspace.
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for i in range(CALLS):
                    ba4_out(xs[i], weights[i], outs[i], partials[i], route, "")
                torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated() - allocated
                row.update(allocation_free=(extra == 0), transient_device_bytes=extra)
                candidate = make_graph(xs, outs, partials, name)
                trace = profile_graph(candidate, "M%d-%s" % (m, name))
                row["graph_ok"] = record_correctness(
                    args.json, report, row, "graph", candidate[1], references,
                    prefixes, trace, baseline_outputs=current_outputs)["passed"]
                if not row["graph_ok"]:
                    row["error"] = "graph replay correctness failed"
                    del candidate
                    continue
                trials = []
                for _ in range(3):
                    before = sample(base)
                    total = sample(candidate)
                    after = sample(base)
                    trials.append({"control_before_us": before, "candidate_us": total,
                                   "control_after_us": after})
                costs = [t["candidate_us"] for t in trials]
                row.update(us_per_step=statistics.median(costs), timing=TIMING,
                           us_per_call=trace["b1_trace_us_per_step"] / CALLS,
                           trial_us_per_step=costs, paired_trials=trials,
                           paired_savings_us=[(t["control_before_us"] + t["control_after_us"]) / 2
                                              - t["candidate_us"] for t in trials],
                           effective_weight_GBps=weight_bytes / trace["b1_trace_us_per_step"] / 1000,
                           scratch_bytes=partials[0].numel() * 4 * CALLS,
                           scratch_used=name.startswith("triton_split"),
                           packing_us=0, synchronization_inside_timing=False, **trace)
                del candidate
            except (RuntimeError, NotImplementedError) as exc:
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
                # Device errors (illegal access, assert, etc.) are not optional
                # algorithm rejection: synchronization rethrows them and aborts.
                torch.cuda.synchronize()
            finally:
                torch.backends.cuda.preferred_blas_library(blas_original)
                write_report(args.json, report)
                print_measurement(m, row)
        del base
        write_report(args.json, report)

    best = select_serving_route(report["measurements"])
    if best is None:
        report.update(verdict="NO_WIN", reason="no correct allocation-free graph path for M=4; M=8/12/16/32 informational only")
        return
    # Compile the EXACT mutation-only op and persistent output pattern used in
    # serving, for the selected M4 route. Reject functionalization allocations.
    name = best["name"]

    check_deadline()
    xs, outs, partials, references = inputs(ROUTE_M)
    bg = make_graph(xs, outs, partials, "current")
    current_outputs = [a.clone() for a in bg[1]]
    prefixes = [e["prefix"] for e in entries]
    # Keep the original benchmark's common logging prefix: specializing this
    # single compiled callable on 48 different strings exhausts Dynamo's cache.
    # Weights, input strides and persistent buffers remain distinct per layer.
    layers = [compiled_layer(name, prefixes[0], xs[i], weights[i], outs[i], partials[i])
              for i in range(CALLS)]
    # These graph-owned tensors keep their storage/offsets throughout replay.
    offsets = [(weights[i].storage_offset(), xs[i].storage_offset()) for i in range(CALLS)]
    compiled = torch.compile(compiled_route, fullgraph=True)
    outputs = [compiled(layers[i], xs[i], *offsets[i]) for i in range(CALLS)]
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = [compiled(layers[i], xs[i], *offsets[i]) for i in range(CALLS)]
    torch.cuda.synchronize()
    report["compiled_check"] = {
        "allocation_free": torch.cuda.max_memory_allocated() == allocated,
        "output_aliases_buffer": all(a.data_ptr() == b.data_ptr() for a, b in zip(outputs, outs))}
    outputs, compiled_trace = profile_calls(
        lambda: [compiled(layers[i], xs[i], *offsets[i]) for i in range(CALLS)],
        "selected-compiled-eager")
    diagnostics = report.setdefault("compiled_diagnostics", {})
    report["compiled_check"]["correct"] = record_correctness(
        args.json, report, diagnostics, "eager", outputs, references, prefixes,
        compiled_trace, baseline_outputs=current_outputs)["passed"]
    cg = make_graph(xs, outs, partials, name,
                    override=lambda i: compiled(layers[i], xs[i], *offsets[i]))
    trace = profile_graph(cg, "selected-compiled")
    report["compiled_check"]["capture_replay_correct"] = record_correctness(
        args.json, report, diagnostics, "graph", cg[1], references, prefixes,
        trace, baseline_outputs=current_outputs)["passed"]
    # A changed input must be observed on replay; guards against stale results.
    xs[0].neg_()
    cg[0].replay()
    torch.cuda.synchronize()
    changed_trace = profile_graph(cg, "selected-compiled-changed-input")
    changed_reference = (xs[0].double() @ weights[0].double().t()).bfloat16()
    changed_current = torch.nn.functional.linear(xs[0], weights[0])
    report["compiled_check"]["changed_input_correct"] = record_correctness(
        args.json, report, diagnostics, "changed_input", [cg[1][0]],
        [changed_reference], prefixes[:1], changed_trace,
        baseline_outputs=[changed_current])["passed"]
    xs[0].neg_()
    bg = make_graph(xs, outs, partials, "current")
    trials = []
    for _ in range(3):
        before = sample(bg)
        total = sample(cg)
        after = sample(bg)
        trials.append({"control_before_us": before, "candidate_us": total,
                       "control_after_us": after})
    best["bare_op_us_per_step"] = best["us_per_step"]
    costs = [t["candidate_us"] for t in trials]
    best.update(us_per_step=statistics.median(costs), paired_trials=trials,
                paired_savings_us=[(t["control_before_us"] + t["control_after_us"]) / 2
                                   - t["candidate_us"] for t in trials],
                compiled_selected=True, trial_us_per_step=costs,
                us_per_call=trace["b1_trace_us_per_step"] / CALLS,
                effective_weight_GBps=weight_bytes / trace["b1_trace_us_per_step"] / 1000,
                **trace)
    print_measurement(4, dict(best, name=name + " compiled"))
    lt = next(r for r in report["measurements"]["4"] if r["name"] == "cublaslt_mm_probe")
    lt_faster = (lt.get("correct") and lt.get("graph_ok")
                 and lt.get("us_per_step", float("inf")) < best["us_per_step"])
    report["cublaslt_requires_per_call_pin"] = bool(lt_faster)
    qualified = (winning(best) and all(report["compiled_check"].values())
                 and not lt_faster
                 and select_serving_route(report["measurements"])["name"] == name
                 and report["real_capture"])
    report.update(selected=best["name"] if qualified else None,
                  verdict="PROVISIONAL" if qualified else "NO_WIN",
                  reason="M=4 route only; M=8/12/16/32 informational; require both-rank L1a vs L1a_BA4 PROFILE c1 before WIN/ACTIVE" if qualified
                  else "M=4 route only; M=8/12/16/32 informational; require real capture, >=500us whole-replay saving in every pair, compile/graph/no-allocation checks; faster cuBLASLt probe needs scoped per-call pin")
    if qualified:
        from vllm_ba4.policy import admission_reason
        reason = admission_reason(report, report['fingerprint'], code_sha256(), args.rank,
                                  benchmark_sha256(), profile_only=True)
        if reason:
            report.update(verdict='NO_WIN', selected=None, reason=reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--capture", type=Path)
    data.add_argument("--synthetic", action="store_true", help="diagnostic only; cannot admit a layer")
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=170, metavar="30..170")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 30 <= args.seconds <= 170:
        p.error("--seconds must be between 30 and 170")
    if not args.worker:
        # Supervise imports/JIT/driver stalls as well as the timed section.
        # 170s budget + 5s grace, leaving headroom under the required 180s.
        report = {"protocol": PROTOCOL, "rank": args.rank, "verdict": "RUNNING",
                  "selected": None, "measurements": {}}
        write_report(args.json, report)
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     *sys.argv[1:], "--worker"], timeout=args.seconds + 5)
            if result.returncode != 0:
                report = json.loads(args.json.read_text())
                if report["verdict"] == "RUNNING":
                    report.update(verdict="BLOCKED", selected=None,
                                  reason="worker terminated before qualification")
                    write_report(args.json, report)
                    print("VERDICT: BLOCKED (worker terminated; partial JSON preserved)", flush=True)
            return result.returncode
        except subprocess.TimeoutExpired:
            report = json.loads(args.json.read_text())
            report.update(verdict="BLOCKED", selected=None, reason="hard wall limit; no qualification")
            write_report(args.json, report)
            print("VERDICT: BLOCKED (wall limit; partial measurements in %s)" % args.json, flush=True)
            return 2
    report = json.loads(args.json.read_text())
    started = time.monotonic()
    try:
        worker(args, report)
    except Exception as exc:
        report.update(verdict="BLOCKED", selected=None,
                      reason="%s: %s" % (type(exc).__name__, exc))
    report["worker_wall_seconds"] = time.monotonic() - started
    write_report(args.json, report)
    print("VERDICT: %s selected=%s; %s; JSON=%s" %
          (report["verdict"], report.get("selected"), report.get("reason", ""), args.json), flush=True)
    return 0 if report["verdict"] in ("PROVISIONAL", "NO_WIN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
