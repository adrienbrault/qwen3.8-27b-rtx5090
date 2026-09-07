# Boot-to-boot decode spread, no-spec pool ceiling, depth scaling: suffix decoding rejected (2026-08-29, `results/2026-08-29-r115-misc`)

[← all results](../RESULTS.md)

Three identical boots of the daily config read code c1 **190.2 / 176.9 / 177.0** at MTP acceptance 0.613 / 0.570 / 0.554. Within-boot spread is ±2–5 t/s, so the ±7% swing is boot-level state: something nondeterministic at engine start fixes drafting quality for the boot's lifetime. Compare decode A/Bs within one boot, or run ≥3 boots per arm. Booting the same engine without speculative decoding shows the MTP head and draft reservations cost 171K tokens of pool (552,838 vs 381,300). Suffix decoding was measured and rejected at 31.2 t/s on code (see REJECTED.md). Deep-context decode on the promoted daily measured 127.2 t/s at 30K, flat against surface, and 115.5 at 100K, which is +10–12% over the previous generation at depth. The XQA decode path barely bends with context.
