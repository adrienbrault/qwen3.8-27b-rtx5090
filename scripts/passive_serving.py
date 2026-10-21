#!/usr/bin/env python3
"""Sample an existing server and GPU counters; sends no inference requests.

Usage on the serving host: python3 passive_serving.py --seconds 60
Output is JSONL on stdout. No files are written on the serving host.
"""
import argparse
import datetime
import json
import subprocess
import time
import urllib.request

KEYS = {
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("seconds and interval must be positive")
    start = time.monotonic()
    previous = None
    tick = 0
    while True:
        row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "elapsed_s": round(time.monotonic() - start, 3)}
        try:
            with urllib.request.urlopen(args.url + "/metrics", timeout=5) as response:
                lines = response.read().decode().splitlines()
            counters = {}
            for line in lines:
                key = line.split("{", 1)[0].split(" ", 1)[0]
                if key in KEYS:
                    counters[key] = counters.get(key, 0) + float(line.rsplit(" ", 1)[1])
            row["metrics"] = counters
            now = time.monotonic()
            if previous:
                dt = now - previous[0]
                row["interval_s"] = dt
                row["rates"] = {
                    key: (value - previous[1][key]) / dt
                    for key, value in counters.items()
                    if key.endswith("_total") and key in previous[1] and value >= previous[1][key]
                }
            previous = (now, counters)
        except Exception as error:
            row["error"] = str(error)
        if tick % 5 == 0:
            query = "index,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm,clocks.mem"
            result = subprocess.run(["nvidia-smi", "--query-gpu=" + query,
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5)
            row["gpu_csv"] = result.stdout.strip().splitlines()
        print(json.dumps(row), flush=True)
        tick += 1
        if time.monotonic() - start >= args.seconds:
            break
        time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - start))))


if __name__ == "__main__":
    main()
