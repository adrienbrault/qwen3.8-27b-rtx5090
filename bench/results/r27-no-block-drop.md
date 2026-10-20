# r27 no block drop: with `disable_eagle_block_drop`, 32K prefix revisits match the cold text 12 of 12 times and the warm TTFT falls to 0.071 s (2026-10-20, results `2026-10-20-051649-r27-flags-screen`, `2026-10-20-070708-r27-flags-screen`, `2026-10-20-081135-nbd-promote`)

[← all results](../RESULTS.md)

On 2026-10-20 the user approved adding `"disable_eagle_block_drop":true` to the MTP speculative config of the 27B daily. The served launcher is [scripts/serve-v031-nbd-daily.sh](../../scripts/serve-v031-nbd-daily.sh), which is [scripts/serve-v031-ba4-daily.sh](../../scripts/serve-v031-ba4-daily.sh) plus the flag. The BA4 launcher is the rollback. Image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, the checkpoint, KV pin, sequence limit, draft length, dtypes, tiers and the BA4 layer are unchanged.

The launcher adds the flag on the daily branch and asserts it on the container's arguments, and its UP line reads `MTP head ns3 no-block-drop`. Experiment boots default to the flag and take `NO_BLOCK_DROP=0` for the old behavior. [scripts/serve.sh](../../scripts/serve.sh) sets it by default (`NO_BLOCK_DROP=1`).

The screens ran [scripts/r27-flags-screen.sh](../../scripts/r27-flags-screen.sh), decided by [scripts/r27-flags-decide.py](../../scripts/r27-flags-decide.py). Their probes are [decode_ss.py](../../scripts/decode_ss.py), [reask.py](../../scripts/reask.py) and [warm_identity.py](../../scripts/warm_identity.py). The promotion ran [scripts/r27-nbd-promote.sh](../../scripts/r27-nbd-promote.sh). The values below come from each round's audit log and verdict; the raw records stay in the results directories named here and are not published.

## What the flag changes

With an EAGLE or MTP drafter, vLLM drops the last matching full-attention block of every prefix-cache hit. The drafter's KV slot at the end of the producing request's last chunk was written with that request's lookahead token, and dropping the block makes the drafter recompute it. In the hybrid align mode the scheduler then backs the linear-attention state checkpoint off by one block as well. Each revisit therefore re-prefills one whole 1,472-token block.

With the flag on, the hit reaches the last full block. The one draft-KV slot written with the producer's lookahead token can lower draft acceptance but not change the output, since the target model verifies every token. vLLM v0.31.0 carries the option and logs that trailing prefix-cache block dropping is disabled. Draft PR [vllm#61911](https://github.com/vllm-project/vllm/pull/61911) turns it on by default for hybrid models in align mode with single-lookahead MTP or EAGLE ([THIRD_PARTY.md](../../THIRD_PARTY.md)).

## Round 1: both flags on decode and warm TTFT; decode cells too noisy to decide

The first round ran 2026-10-20 02:39 to 03:04 UTC (results `2026-10-20-023934-r27-flags-screen`). It screened two zero-code options of vLLM v0.31.0's speculative config on the BA4 launcher at the experiment port, booting C P E P C E:

- **C:** control.
- **P:** `"draft_sample_method":"probabilistic"`. The drafter samples, and the rejection sampler accepts with probability min(1, p/q).
- **E:** `"disable_eagle_block_drop":true`.

Each boot ran `decode_ss.py` at 1 and 3 streams on code and prose, sampled at temperature 0.6 with 1,024 forced tokens, 3 runs. It also ran `reask.py`: a 6K and a 32K completions prompt sent three times 2 s apart, with 8 output tokens. Values are medians of the two boots per arm, in per-stream tokens per second and accepted draft tokens per draft token.

| cell | C t/s | P t/s | E t/s | C accept | P accept | E accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code, 1 stream | 258.8 | 267.6 | 243.8 | 0.669 | 0.698 | 0.610 |
| code, 3 streams | 211.7 | 231.4 | 220.2 | 0.605 | 0.691 | 0.647 |
| prose, 1 stream | 190.1 | 210.1 | 214.6 | 0.403 | 0.477 | 0.498 |
| prose, 3 streams | 181.2 | 183.9 | 177.1 | 0.470 | 0.483 | 0.454 |

| prompt | C warm TTFT | P warm TTFT | E warm TTFT | C cold TTFT | E cold TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6K | 0.196 s | 0.196 s | 0.041 s | 0.64 s | 0.64 s |
| 32K | 0.398 s | 0.399 s | 0.225 s | 3.88 s | 3.94 s |

The rules, fixed in the driver before the data:

- **P passes** if acceptance at one stream rises by at least 1.5 points on both kinds, every cell is at least C − 2 %, and both one-stream cells reach at least C + 1 %.
- **E passes** if warm TTFT falls by at least 0.08 s at 32K and does not rise at 6K, and every decode cell is within 2 % of C.

The verdicts were P PASS (+3.4, +9.3, +10.5, +1.5 %) and E FAIL on its decode clause. E changes nothing on a decode step without a prefix hit, yet against C it read −5.8 to +12.9 % in rate and −5.9 to +9.5 points in acceptance. Decode cells of 3 runs on 2 boots per arm therefore carried about ±10 % of boot noise, so neither P's pass nor E's decode failure is evidence. E's warm TTFT drop, 0.155 s at 6K and 0.173 s at 32K with cold TTFT unchanged, was the one clear effect. The rounds below re-measure decode with 10 runs per cell on 3 boots per arm, and test E's effect on the output.

## Round 2: probabilistic drafting trades code for prose

The second round ran 03:37 to 04:16 UTC (results `2026-10-20-033705-r27-flags-screen`). It booted C P C P C P, with 10 decode runs per cell and no reask. Per-boot values are per-stream tokens per second; the median is over the three boots.

| cell | C boots | P boots | C median | P median | Δ | C accept | P accept |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| code, 1 stream | 262.4, 265.9, 260.8 | 261.0, 252.8, 251.1 | 262.4 | 252.8 | −3.7 % | 0.673 | 0.639 |
| code, 3 streams | 212.2, 228.3, 225.1 | 223.4, 229.2, 228.2 | 225.1 | 228.2 | +1.4 % | 0.665 | 0.673 |
| prose, 1 stream | 207.7, 203.8, 206.1 | 219.0, 216.9, 221.6 | 206.1 | 219.0 | +6.3 % | 0.464 | 0.508 |
| prose, 3 streams | 181.7, 182.9, 183.6 | 186.1, 186.5, 189.3 | 182.9 | 186.5 | +2.0 % | 0.476 | 0.488 |

Under the same rule P fails: code at one stream reads −3.7 %, with all three P boots at or below C's lowest. Prose at one stream reads +6.3 %, every P boot above every C boot. Probabilistic drafting stays off. It is reopened for a replay of thinking-on agent turns, whose mix of code and prose these single-prompt cells do not represent.

## Round 3: warm 32K revisits equal the cold text 12 of 12 with the flag, 1 of 12 without

The third round ran 05:16 to 05:32 UTC (results `2026-10-20-051649-r27-flags-screen`) and booted C E C E. [`warm_identity.py`](../../scripts/warm_identity.py) sends each fresh prompt twice, cold then warm. The prompt is filler of the stated length followed by a coding question, sent through chat completions with thinking off, greedy, 256 tokens. The probe then compares the two texts. Each boot ran 10 prompts at 6K and 6 at 32K, so each arm has 20 and 12. TTFTs are medians over the arm's prompts.

| prompt | C warm == cold | E warm == cold | C warm TTFT | E warm TTFT | C cold TTFT | E cold TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6K | 20 / 20 | 20 / 20 | 0.265 s | 0.121 s | 0.56 s | 0.57 s |
| 32K | 1 / 12 | 12 / 12 | 0.275 s | 0.071 s | 3.08 s | 2.92 s |

The rule, fixed before the data: E passes if its identical count at each length is at least C's minus one, and its warm TTFT is below C's. Verdict: PASS. Without the flag, 11 of the 12 warm 32K continuations differ from their cold run, first at characters 10 to 879; with it the warm request reuses every cached block and its text equals the cold text in every prompt. This round does not identify how the re-prefilled block changes the continuation. The same pattern appears in the [BA4 gate](r27-ba4.md#greedy-decode-differs-from-l1a-the-pre-registered-near-tie-rule-failed) (2026-10-19, results `2026-10-19-220845-ba4-lean-gate`): L1a against its own prefix-cached second run agreed on 20 of 20 chunks at no context and 11 of 20 behind 30,000 tokens.

## Round 4: decode without prefix hits is unchanged

The fourth round ran 07:07 to 07:47 UTC (results `2026-10-20-070708-r27-flags-screen`) and booted C E C E C E. Each boot ran `decode_ss.py` with 10 runs per cell at temperature 0.6, with no prefix revisits.

| cell | C boots | E boots | C median | E median | Δ | C accept | E accept |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| code, 1 stream | 261.0, 267.4, 261.8 | 237.6, 267.8, 266.0 | 261.8 | 266.0 | +1.6 % | 0.676 | 0.689 |
| code, 3 streams | 214.0, 231.4, 225.4 | 220.6, 226.0, 224.6 | 225.4 | 224.6 | −0.4 % | 0.659 | 0.655 |
| prose, 1 stream | 209.8, 205.4, 207.4 | 212.1, 211.1, 213.1 | 207.4 | 212.1 | +2.3 % | 0.464 | 0.482 |
| prose, 3 streams | 183.2, 184.9, 180.5 | 183.6, 184.7, 184.5 | 183.2 | 184.5 | +0.7 % | 0.475 | 0.479 |

The rule fixed before the data was |Δ| ≤ 2 % in every cell. It fails by its letter on prose at one stream, where E is 2.3 % faster and all three E boots read above all three C boots. No cell reads slower than −0.4 %.

## The promotion boot

On 2026-10-20 08:11 to 08:16 UTC (results `2026-10-20-081135-nbd-promote`), the promotion script installed the launcher, kept the BA4 launcher as the rollback and booted the installed launcher on the experiment port. The boot read:

- the label `MTP head ns3 no-block-drop` and the flag on the container;
- a pool of 1,391,795 tokens;
- 96 of 96 BA4 rank-layers active;
- the tool smoke at 5 of 5;
- three 32K warm revisits equal to their cold runs, 3 of 3, with warm TTFT 0.071, 0.062 and 0.119 s against cold 3.33 to 3.38 s.

## Not measured

- **Deeper revisits:** revisits beyond 32K tokens and revisits served from the CPU or disk tier.
- **Agent sessions:** multi-turn replays of real agent sessions.
- **Thinking-on revisits:** the identity probe runs with thinking off.
- **The served curves:** the decode, depth and cold-prefill curves and the standard benchmark were measured on the BA4 launcher before this change ([r27-ba4-curves](r27-ba4-curves.md), [r27-ba4-std-bench](r27-ba4-std-bench.md)). They make no prefix-cache hits, which are the only requests where the flag changes what is computed. Round 4's decode-only cells are the measurement on decode.
