# Benchmarks

- **[RESULTS.md](RESULTS.md)** — all numbers; the current daily at the top, the 2026-07 archive below.
- [`needle_depth.py`](../scripts/needle_depth.py) — depth-needle probe with cold/warm passes and optional concurrent loaders; the first gate of every audition.
- [`nvfp4kv-gauntlet.sh`](../scripts/nvfp4kv-gauntlet.sh) — the audition driver (boot facts, needles, sean gate, killer, vision, SO, tool-eval, benchy).
- `template_probe.py <base_url> <label>` — behavioural probe: single tool call, **parallel** tool
  calls, chat→tool-result→chat continuation, plain chat turn after tools. This is how you evaluate
  a chat template; a speed benchmark can't.
- `prefix_probe.py <base_url> <label>` — prefix-cache hit rate. **Note:** vLLM's counter reports 0%
  on this hybrid model even when the cache works. Trust the TTFT timing, not the metric.

Throughput: [llama-benchy](https://github.com/eugr/llama-benchy).

```bash
llama-benchy --base-url http://localhost:8020/v1 --model qwen3.8-27b \
  --tokenizer /path/to/model --pp 512 4096 --tg 256 --concurrency 1 2 4 8 \
  --runs 2 --skip-coherence --format md
```

Agentic: [Harbor / Terminal-Bench](https://www.tbench.ai/). Gotcha: terminus-2's litellm runs
**harness-side**, so the credentials must be in harbor's own process env — the `--ae` vars only
reach the task container (keep them for agents that need them there).

```bash
export OPENAI_API_KEY=local
export OPENAI_API_BASE=http://172.17.0.1:8020/v1
export OPENAI_BASE_URL=http://172.17.0.1:8020/v1
harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
  --model openai/qwen3.8-27b \
  --ae OPENAI_API_KEY=local --ae OPENAI_API_BASE=http://172.17.0.1:8020/v1 \
  --ae OPENAI_BASE_URL=http://172.17.0.1:8020/v1 \
  --allow-agent-host 172.17.0.1 --n-concurrent 4 --n-attempts 1
```
