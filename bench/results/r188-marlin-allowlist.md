# R188, per-layer W4A16 (Marlin) on the MLP projections: a fidelity lever, not a speed lever (2026-09-05 04:22 to 05:44 UTC, `results/2026-09-05-r188-marlin-allowlist`, `scripts/r188-marlin-allowlist.sh`, patch 0139)

[← all results](../RESULTS.md)

The daily runs every NVFP4 GEMM as W4A4. Patch 0139 routes allowlisted layers to the Marlin kernel, which keeps the FP4 weights and uses bf16 activations (W4A16). Seven boots on the daily image plus the patch, 16 sequences, no pcie_ipc: control, all 112 MLP projections, gate_up only, down only, and the three layer thirds. Each boot ran the dense ruler (693 documents, 724,781 positions against the bf16 dump), the agentic ruler (57,972 positions), the greedy decode ruler and the decode probes. Steps per second = tokens per second divided by 1 + 9 times the acceptance per draft token.

| arm | dense PPL gap to bf16 | dense top-1 | agentic PPL gap | agentic top-1 | steps/s code c1 | prose c1 | code c8 | code c16 | KV pin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control (W4A4 everywhere) | +0.667% | 92.74% | +2.748% | 95.63% | 71.9 | 71.1 | 351.0 | 455.3 | 13.98 GB |
| all MLP projections | +0.207% | 94.05% | +1.435% | 96.73% | +1.0% | +1.1% | −15.0% | −21.2% | 13.5 GB |
| gate_up only | +0.575% | 93.63% | +1.690% | 96.29% | +0.3% | +0.4% | −9.1% | −15.1% | 13.98 GB |
| down only | +0.553% | 93.15% | +2.284% | 95.91% | +1.0% | +2.1% | −5.4% | −10.3% | 13.98 GB |
| layers 0 to 18 | +0.762% | 92.93% | +2.572% | 95.56% | −0.3% | +0.6% | −5.6% | −8.9% | 13.98 GB |
| layers 19 to 37 | +0.560% | 93.24% | +2.025% | 96.06% | −0.1% | +0.8% | −5.3% | −8.5% | 13.98 GB |
| layers 38 to 55 | +0.390% | 93.35% | +2.044% | 96.07% | −0.1% | 0.0% | −4.6% | −8.1% | 13.98 GB |

The greedy decode ruler agrees: at 30K context every Marlin arm is closer to bf16 than the control (median absolute delta logprob on agreed tokens 0.0042 to 0.0067 against 0.0074). The activation quantization is the larger part of the remaining gap to bf16, and it is concentrated in the last third of the stack. The all-MLP arm needed the 13.5 GB KV pin (347 MiB free at 13.98 GB). The offline census predicted a c1 win from Marlin's faster small-M GEMMs and the engine shows parity at c1; the c8 and c16 losses follow the census ordering. No engine errors on any arm. Not promoted; the next step is the same allowlist on the Humming W4A16 kernel, which the census has 20 to 30% faster than Marlin at M 80 to 160.
