# Benchmarks

[RESULTS.md](RESULTS.md) holds every measurement, newest first. The served configuration is at the top; the Qwen3.6-era archive is at the bottom. [reproduce/](reproduce/README.md) holds the SWE-Bench prediction files and the rescoring script.

## Probes in this repo

| script | what it measures |
|---|---|
| [`../scripts/decode_ss.py`](../scripts/decode_ss.py) | Steady-state decode throughput per content type (code, prose) at a given concurrency, from the engine's own metrics. The decode number quoted for a configuration comes from here. |
| [`../scripts/fidelity.py`](../scripts/fidelity.py), [`fidelity_compare.py`](../scripts/fidelity_compare.py), [`fidelity_ladder.py`](../scripts/fidelity_ladder.py) | Teacher-forced perplexity and top-1 agreement against a reference model over a fixed corpus. The ruler used to pick the checkpoint and to validate the NVFP4 KV path. |
| [`../scripts/decode_fidelity.py`](../scripts/decode_fidelity.py) | The same, through the decode path. Prefill-only rulers cannot see decode-kernel bugs. |
| [`../scripts/needle_depth.py`](../scripts/needle_depth.py) | Depth-needle retrieval with cold and warm passes and optional concurrent loaders. The first gate of every audition. |
| [`../scripts/nvfp4kv-gauntlet.sh`](../scripts/nvfp4kv-gauntlet.sh) | The audition driver: boot facts, needles, the concurrent-loader gate, the 8-stream burst, vision, structured output, tool-eval, llama-benchy. |
| [`../scripts/nvfp4_fa2_harness.py`](../scripts/nvfp4_fa2_harness.py) | Differential test of the FA2 NVFP4 attention kernel against an fp32 reference across 576 shape cells. |
| [`template_probe.py`](template_probe.py) | Chat-template behaviour: single tool call, parallel tool calls, chat after tool results. |
| [`prefix_probe.py`](prefix_probe.py) | Prefix-cache effect by timing a repeated prompt. vLLM's hit counter reads 0 with an external connector on. |

## External tools

Throughput: [llama-benchy](https://github.com/eugr/llama-benchy).

```bash
llama-benchy --base-url http://localhost:8020/v1 --model qwen3.8-27b \
  --tokenizer /path/to/model --pp 2048 --tg 256 --concurrency 1 8 \
  --runs 3 --skip-coherence --format md
```

Tool calling: [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), 69 cases, reported as the mean of 2 or 4 runs with the spread.

Agentic: [SWE-Bench Verified](https://github.com/SWE-bench/SWE-bench) scored with the official harness, and [Terminal-Bench 2.1](https://www.tbench.ai/) through Harbor with the terminus-2 agent. Harbor's litellm runs on the harness side, so the endpoint variables must be in Harbor's own environment; the `--ae` variables only reach the task container.

```bash
export OPENAI_API_KEY=local OPENAI_API_BASE=http://172.17.0.1:8020/v1 OPENAI_BASE_URL=http://172.17.0.1:8020/v1
harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
  --model openai/qwen3.8-27b \
  --ae OPENAI_API_KEY=local --ae OPENAI_API_BASE=http://172.17.0.1:8020/v1 --ae OPENAI_BASE_URL=http://172.17.0.1:8020/v1 \
  --allow-agent-host 172.17.0.1 --n-concurrent 4 --n-attempts 1
```
