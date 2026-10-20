# r27 draftshare: the MTP draft passes take 16 % of a single-stream decode step, and the step is 4.9 % longer in the boot without the `pcie_ipc` all-reduce (2026-10-20, results `2026-10-20-060844-r27-draftshare`, `2026-10-20-082652-r27-draftshare`)

[← all results](../RESULTS.md)

2026-10-20 06:08 to 06:14 UTC and 08:26 to 08:31 UTC. The raw run directories and the test-only hook are on the serving host and are not included in this repository.

## What was run

Each run booted the serving host's launcher once on the experiment port with the BA4 image, 16 sequences, batch-sharded sampling, 96 of 96 BA4 rank-layers active, a pool of 1,391,795 tokens and the stock power limits of 600 / 575 W. A test-only hook, loaded through a `.pth` file without modifying vLLM, recorded two CUDA events around every call of the MTP speculator's `propose()` on each rank: the draft time is the interval between the two events, the step period is the interval between the starts of consecutive calls. [decode_ss.py](../../scripts/decode_ss.py) then decoded 1,024 tokens per stream at temperature 0.6, five runs, code and prose at one stream and code at three streams. Both runs used the same seeded prompts; their step counts are identical (1,568, 2,194 and 1,794).

The two runs differ in two settings. The first run booted the configuration of [scripts/serve-v031-ba4-daily.sh](../../scripts/serve-v031-ba4-daily.sh) without the `pcie_ipc` all-reduce. Its boot log reads `PCIe IPC all-reduce disabled: knob off or stateless group` and `Using ['CUSTOM', 'PYNCCL'] all-reduce backends`: vLLM's own peer-to-peer all-reduce carried the collectives. The second run booted the served configuration of [scripts/serve-v031-nbd-daily.sh](../../scripts/serve-v031-nbd-daily.sh), promoted between the two runs: `PCIe IPC all-reduce enabled`, `Using ['PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends`, and `"disable_eagle_block_drop":true` in the MTP speculative config. Both runs carry the same hook, so its cost is in both.

## Result

Median per step, rank 0 / rank 1:

| shape | `pcie_ipc` | step period ms | draft ms | draft share | decode t/s (median) | acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| code, 1 stream | off | 11.513 / 11.512 | 1.806 / 1.797 | 15.7 / 15.6 % | 260.8 | 0.677 |
| code, 1 stream | on | 10.947 / 10.947 | 1.748 / 1.738 | 16.0 / 15.9 % | 283.0 | 0.700 |
| prose, 1 stream | off | 11.500 / 11.501 | 1.807 / 1.798 | 15.7 / 15.6 % | 206.3 | 0.461 |
| prose, 1 stream | on | 10.951 / 10.951 | 1.748 / 1.739 | 16.0 / 15.9 % | 215.2 | 0.463 |
| code, 3 streams | off | 13.244 / 13.244 | 2.001 / 2.000 | 15.1 / 15.1 % | 213.4 | 0.613 |
| code, 3 streams | on | 12.812 / 12.812 | 1.951 / 1.955 | 15.2 / 15.3 % | 229.7 | 0.651 |

The three MTP draft passes take 1.74 to 1.81 ms of a single-stream step, 15.6 to 16.0 % of it, in both runs.

With `pcie_ipc` on, the single-stream step is 0.57 ms shorter for code (11.51 → 10.95 ms, −4.9 %) and 0.55 ms shorter for prose (−4.8 %), and the 3-stream step 0.43 ms shorter (−3.3 %). The decode rates differ by more than the step because acceptance differs between the runs. This is one boot per arm, two hours apart, not an order-balanced pair, and the second arm also carries the block-drop flag, which [r27-no-block-drop](r27-no-block-drop.md) measured at −0.4 to +2.3 % decode over three boots per arm without prefix revisits; [R185](r185-pcie-ipc-all-reduce.md) measured the paired effect on the vLLM 0.29 route (+4.6 % code and +5.4 % prose single-stream decode, 2026-09-05).

Both arms ran on a host with GPU peer-to-peer. The off arm compares `pcie_ipc` with vLLM's CustomAllreduce; on a host without peer-to-peer CustomAllreduce disables itself as well and NCCL carries every all-reduce, which no round has measured.
