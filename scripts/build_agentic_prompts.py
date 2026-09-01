#!/usr/bin/env python3
"""R156 agentic prompt set — pinned, deterministic, chat-shaped (the DEPLOYED regime).

The raw-text fidelity corpus measured weight fidelity on memorised text with no chat template,
tools or thinking. This set is what the box actually serves: tool-use tasks with a real tool
schema (rendered by the model's own chat template), coding tasks, reasoning, long-form prose.
bf16 generates the reference; quantized arms are teacher-forced on bf16's exact token ids.
Doc-id ranges encode kind: tool 0-99, code 100-199, reason 200-299, prose 300-399.

  build_agentic_prompts.py --out /srv/qwen5090/r156-agentic-prompts.jsonl
"""
import argparse, hashlib, json

TOOLS = [
 {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Create or overwrite a text file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"run_shell","description":"Run a shell command and return stdout/stderr","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout_s":{"type":"integer","default":60}},"required":["cmd"]}}},
 {"type":"function","function":{"name":"search_code","description":"Regex search under a directory","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"dir":{"type":"string"},"glob":{"type":"string"}},"required":["pattern","dir"]}}},
 {"type":"function","function":{"name":"http_get","description":"HTTP GET a URL, return status and body","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
]
SYS_TOOL = "You are an engineering assistant operating on a Linux server. Use the provided tools when they are needed to answer accurately; explain what you did."
TOOL_TASKS = [
 "Check disk usage on / and /storage and tell me whether either is above 80%.",
 "Find every TODO and FIXME under src/ and summarise them grouped by file.",
 "Fetch https://example.internal/status.json and report the service version and uptime.",
 "The nginx container keeps restarting. Look at its last 50 log lines and diagnose.",
 "Create a systemd timer that runs /opt/backup.sh daily at 03:15 and enable it.",
 "Which process is listening on port 8020? Report its PID, user and command line.",
 "Count lines of Python code under app/ excluding tests, and list the five largest files.",
 "Rotate the log file /var/log/app.log: compress it with a date suffix and truncate the original.",
 "Read config.yaml and tell me the database host, port and pool size.",
 "There is a merge conflict in src/api/routes.py. Show me the conflicting hunks.",
 "Check whether the SSL certificate for api.example.com expires within 30 days.",
 "List all Docker images older than 90 days and estimate the space they use.",
 "Grep the codebase for hard-coded AWS keys (AKIA...) and report file and line.",
 "Measure the p95 latency of http://localhost:8080/health over 20 requests.",
 "Write a small Python script to /tmp/dedupe.py that removes duplicate lines from a file, then run it on /tmp/input.txt.",
 "Find which git commit introduced the string 'retry_backoff' in the repo at /srv/app.",
 "Show memory usage per container and flag any above 2 GiB.",
 "Verify that /etc/ssh/sshd_config disables password authentication, and fix it if not.",
 "Tail the last 100 lines of journalctl for the k3s service and summarise errors.",
 "Compare requirements.txt with what pip freeze reports inside the venv at /opt/venv.",
 "Create a cron entry that clears /tmp/cache every Sunday at 04:00.",
 "Look up the current ZFS pool health and scrub status, and explain any warnings.",
 "Download https://example.internal/schema.sql and count the CREATE TABLE statements.",
 "Find files under /storage/data modified in the last 24 hours larger than 1 GiB.",
 "Check that the GPU driver is loaded and report the driver version and each GPU's memory.",
 "Add a .gitignore for a Python project at /srv/newproj and commit it.",
 "Search the Go code under cmd/ for context.Background() calls outside main and list them.",
 "Report the five most frequent 5xx paths from /var/log/nginx/access.log today.",
 "Set the timezone of the machine to Europe/Paris and confirm with the date command.",
 "Find the largest 10 directories under /home and print sizes in human-readable form.",
 "Check whether port 443 on 10.0.0.5 is reachable and how long the TCP handshake took.",
 "Read pyproject.toml and tell me the package name, version and declared Python requirement.",
]
CODE_TASKS = [
 "Write a Python function `parse_duration(s: str) -> int` that accepts strings like '1h30m', '45s', '2d4h' and returns seconds. Include pytest tests covering edge cases.",
 "Implement an LRU cache class in TypeScript with `get`, `set` and a capacity, O(1) operations, plus vitest tests.",
 "Write a Go function that reads a CSV with a header row and returns a slice of maps keyed by column name. Handle quoted fields. Add table-driven tests.",
 "Implement `merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]` in Python with tests, including touching and nested intervals.",
 "Write a Rust function that validates an IPv4 address string without using the std parser, with unit tests.",
 "Write a bash script that safely rotates a set of log files keeping the last N, with dry-run support.",
 "Implement a token-bucket rate limiter in Python usable as a decorator, thread-safe, with tests.",
 "Write a SQL query (PostgreSQL) that returns each customer's most recent order and the number of orders in the last 90 days.",
 "Implement binary search on a rotated sorted array in Python, returning the index or -1, with tests.",
 "Write a Python context manager that atomically writes a file (write to temp, fsync, rename), with tests.",
 "Implement a minimal JSON pretty-printer in C that reads stdin and writes indented output; handle strings with escapes.",
 "Write a Python async function that fetches a list of URLs with a concurrency limit of 5 and per-request timeout, returning results in input order.",
 "Implement Dijkstra's shortest path in Python over an adjacency dict, returning distance and path, with tests.",
 "Write a TypeScript function that deeply merges two plain objects, with arrays replaced not concatenated, and tests.",
 "Implement a simple Markdown table parser in Python that returns a list of dicts, tolerant of alignment rows, with tests.",
 "Write a Python CLI (argparse) that walks a directory and reports duplicate files by SHA-256, with a --delete flag guarded by confirmation.",
]
REASON_TASKS = [
 "A tank fills at 12 L/min and drains at 5 L/min through a leak. It holds 840 L. Starting empty with both running, when is it full? Then the leak is fixed at the 60-minute mark; when is it full now?",
 "Three friends split a restaurant bill. Anna pays 40%, Ben pays 35%, and Cara pays the remaining 30 euros. What was the total bill, and how much did each pay?",
 "A train leaves at 9:00 at 80 km/h. A second train leaves the same station at 9:45 at 110 km/h on the same track. At what time and distance does it catch up?",
 "A rectangle's length is 3 more than twice its width. Its perimeter is 48. Find its area.",
 "You have 8 coins, one counterfeit and lighter. With a balance scale, what is the minimum number of weighings to guarantee finding it, and describe the procedure.",
 "A store marks up an item 25% then offers a 20% discount. Is the final price higher, lower or equal to cost? By what percentage?",
 "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets? Explain the reasoning carefully.",
 "A password must be 4 characters from the digits 0-9 with no repeated digit and must not start with 0. How many are possible?",
 "Water is 11.19% hydrogen by mass. How many grams of hydrogen are in 2.5 kg of water?",
 "A recipe for 6 people needs 450 g flour and 3 eggs. Scale it for 20 people, rounding eggs up, and give flour in grams.",
 "In a class of 30, 18 play football, 15 play basketball and 6 play neither. How many play both?",
 "A car's fuel economy is 6.2 L/100 km. Fuel costs 1.85 EUR/L. What does a 740 km trip cost, and what is the cost per km?",
 "The sum of three consecutive odd integers is 87. Find them, and then the sum of their squares.",
 "A loan of 12,000 at 6% simple annual interest is repaid after 30 months. What total is repaid?",
 "How many trailing zeros does 50! have? Show the method.",
 "Two dice are rolled. What is the probability the sum is prime? List the favourable outcomes.",
]
PROSE_TASKS = [
 "Write a 600-word explainer on how RAID parity protects against disk failure, for a technically curious non-expert.",
 "Draft a clear incident postmortem (blameless) for a 40-minute outage caused by a full disk on a database host.",
 "Write a product-style README introduction for a self-hosted photo backup tool, covering what it does, who it is for and how to start.",
 "Explain speculative decoding in LLM inference to a software engineer who has never worked on ML systems, in about 500 words.",
 "Write a short essay on why home servers are worth the effort, balancing the case for and against.",
 "Write a step-by-step guide for migrating a small Postgres database between two hosts with minimal downtime.",
 "Explain how ZFS scrubs and checksums differ from a traditional fsck, for a sysadmin coming from ext4.",
 "Write a thoughtful review of the trade-offs between tensor parallelism and pipeline parallelism on a two-GPU workstation.",
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    recs = []
    for i, t in enumerate(TOOL_TASKS):
        recs.append({"id": i, "kind": "tool", "tools": TOOLS, "messages": [{"role":"system","content":SYS_TOOL},{"role":"user","content":t}]})
    for i, t in enumerate(CODE_TASKS):
        recs.append({"id": 100+i, "kind": "code", "messages": [{"role":"user","content":t}]})
    for i, t in enumerate(REASON_TASKS):
        recs.append({"id": 200+i, "kind": "reason", "messages": [{"role":"user","content":t}]})
    for i, t in enumerate(PROSE_TASKS):
        recs.append({"id": 300+i, "kind": "prose", "messages": [{"role":"user","content":t}]})
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"prompts": len(recs), "by_kind": {k: sum(1 for r in recs if r["kind"]==k) for k in ("tool","code","reason","prose")}, "sha256": h}))

if __name__ == "__main__":
    main()
