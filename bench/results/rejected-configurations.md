# Rejected configurations (with numbers)

[← all results](../RESULTS.md)

| | result |
|---|---|
| **`turboquant_4bit_nc`** (4-bit Keys) | **UN-rejected here (the 0/8 was async×spec corruption, [#42655](https://github.com/vllm-project/vllm/issues/42655), not the keys) — then RE-rejected for good on 2026-07-20**: superseded by fp8 KV, and a tier-era re-audition hit a wrong 60K needle + a dead engine under the concurrency killer at a 563K pool. Full arc: [REJECTED.md](../../docs/REJECTED.md) / [HISTORY.md](../../docs/HISTORY.md#turboquant_4bit_nc-became-the-daily-the-asyncspec-reversal). |
| `--async-scheduling` (not passing `--no-async-scheduling`) | c4 552 → 526 on throughput **and** corrupts KV under MTP (0/8 on 4bit_nc, ~10% on k8v4). Rejected — `--no-async-scheduling` is mandatory. |
| **nvfp4-FA2** (FlashInfer FA2 nvfp4 KV) | Builds & runs byte-identical (jethac/vllm + FlashInfer #3684, JIT sm120), but loses to `4bit_nc`: stable pool 184K, decode −8..−23%, tool-eval 82, OOMs at util 0.97, 2-branch dev build + ~15min JIT. Rejected — see [REJECTED.md](../../docs/REJECTED.md). |
| smaller-bit TQ presets `k3v4_nc` / `3bit_nc` | PPL delta vs bf16: k8v4 +1.17% → 4bit_nc +2.71% → k3v4_nc +10.63% → 3bit_nc +20.59%. Every preset below 4bit_nc attacks the keys. Rejected. |
| `--max-num-batched-tokens 4096` (on the TurboQuant NVFP4 config) | prefill 9,607 → 2,556 (**−73%**) for +28K ctx. Rejected **here**. Note: on the current AR + fp8 daily this reverses — `mnbt 4096` neither changes the pool nor costs prefill, and *is* the daily (see [the util sweep](prior-daily-2026-07-18.md#pool-vs-util-util-is-the-only-pool-lever)). |
| `VLLM_TQ_KV_SPLITS=8` | c1 143 → 132, c8 unchanged. Rejected (default 32 is right). |
| froggeric chat template | 4/4 vs bundled 4/4 on a behavioural tool-call probe. No gain. |
| DFlash | 3.3 GB draft model → 1,616-token context. Fatal on 32 GB. |
