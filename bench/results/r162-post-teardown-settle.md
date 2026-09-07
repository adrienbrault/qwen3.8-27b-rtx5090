# Post-teardown settle: 60 seconds is enough on a healthy box (2026-09-03, `results/2026-09-03-r162-settle`, `scripts/r162-settle-ladder.sh`)

[← all results](../RESULTS.md)

Question: every restart cycle waited 300 seconds between a TP=2 teardown and the next TP=2 boot (gotcha 21). That number came from one bad evening: three chained boots with a 45 second settle all died in kernel warmup right after a kernel-panic reboot, and a 300 second settle booted first try. Is 300 needed?

Method: teardown, wait for both GPUs to report idle (immediate, 2 MiB each), sleep S, boot the served shape on a side port with the launcher's fail-fast on the warmup error signature.

| settle | boot | time to health |
|---|---|---|
| 60 s | first try | 137 s |
| 120 s | first try | 138 s |
| 180 s | first try | 137 s |

Reading: on a healthy box the settle length does not matter and boot time is constant. The 45 second failures happened in the first hour after a kernel panic and did not reproduce, so this does not show that 60 seconds would have cleared that state. It shows the normal case does not need the 300 second wait. Not tested: under 60 seconds, and the post-reboot state.

Applied: routine settles are now 60 seconds in the campaign, pause and restore scripts; 300 seconds remains the retry path after a failed boot. A pause-to-healthy-daily cycle is about 60 + 60 + 140 seconds.
