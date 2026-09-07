# R203: speculative decoding on or off does not move the prompt-logprob ladder beyond the boot-to-boot draw (2026-09-06 09:18 to 09:35 UTC, results `2026-09-06-r203-spec-ladder`, [scripts/r203-spec-ladder.sh](../../scripts/r203-spec-ladder.sh), [scripts/ladder_doc_compare.py](../../scripts/ladder_doc_compare.py))

[← all results](../RESULTS.md)

vllm#53488 reports `prompt_logprobs` silently corrupted for a subset of requests under speculative decoding on Qwen3.8-27B (reported with the MTP drafter). Every dense and agentic ruler number in this file since R156 was taken with the served DFlash drafter on, against a bf16 reference dumped with it off. This unit measured the served image twice in the same hour on the served route (16 sequences, `pcie_ipc` all-reduce, batch-sharded sampling, `VLLM_TRITON_FORCE_FIRST_CONFIG=1`): once with no speculative decoding at all (a new `SPEC_METHOD=none` passthrough in the launcher: no drafter, no `--speculative-config`; pool 1,461,692 tokens, attention block 1,424) and once as served (DFlash draft length 7; pool 1,052,277, block 1,552). The control is the R196 daily-image boot of 2026-09-05 (spec on, other compile artifacts): two boots of one configuration.

| arm | dense PPL vs bf16 | dense top-1 | agentic PPL vs bf16 | agentic top-1 | code c1 tok/s |
|---|---|---|---|---|---|
| spec off | +0.791 % | 92.789 % | +2.551 % | 95.582 % | 110.3 |
| spec on (served) | +0.878 % | 92.806 % | +2.560 % | 95.551 % | 243.5 |

Per document (693 dense documents, 724,781 scored positions; 72 agentic documents, 57,972 positions):

| pair | dense docs beyond ±2 % | dense positions moved > 1 nat | dense median doc delta | agentic docs beyond ±2 % |
|---|---|---|---|---|
| on vs off | 92 | 1.31 % | −0.06 % | 3 of 72 |
| on vs R196 boot (control) | 92 | 1.44 % | −0.09 % | 3 of 72 |
| off vs R196 boot (control) | 104 | 1.42 % | −0.03 % | 2 of 72 |

The on-versus-off spread equals the spread between two boots of the same configuration, so the report does not reproduce on the DFlash route and the rulers taken with the drafter on stand. The same numbers calibrate every ladder comparison in this file: two boots of one configuration differ by 0.10 to 0.15 % corpus PPL, about ±2.4 % per document at the 5th and 95th percentiles, and about 1.4 % of positions by more than one nat. A candidate delta inside that band is the per-boot draw (the R193c/R193d finding, now on the prefill ladder). The spec-free boot also shows what the draft slots cost: +39 % pool and 2.8 GB more free VRAM (R200b), against a halved single-stream decode rate.
