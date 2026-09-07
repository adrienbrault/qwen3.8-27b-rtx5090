# Tool-call parser A/B, qwen3_xml vs qwen3_coder: a tie, because both names alias one class (2026-08-26, `results/2026-08-26-parser-ab`)

[← all results](../RESULTS.md)

Community Qwen3.8/5090 stacks commonly ship `--tool-call-parser qwen3_coder`, while this stack ships `qwen3_xml`. Paired same-session tool-eval 69x2 read 91 +- 1.4 vs 91.5 +- 0.7, and the follow-up explains why a tie is required: **both names are registry aliases for the same class** (`qwen3_engine_tool_parser.Qwen3EngineToolParser`, verified in-image and at v0.27.1). The parsers were historically separate and were unified upstream, so either name works.

The 69x4 rerun (xml 88.5 +- 1.7, coder 89.8 +- 2.1) therefore doubles as a same-config repeatability measurement: **12 same-day trials of the identical engine span 118–127/137 points** (mean 90.5, single-trial sigma about 2.0). Practical rule for this eval: differences under about 2 points at 2 trials (about 1.5 at 4) are noise. The `TOOLPARSER` env knob in `scripts/serve-tier-rc4.sh` remains for future non-aliased parsers.
