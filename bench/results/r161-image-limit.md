# Image limit per request: raising the count is free, a pixel cap is not (2026-09-03, `results/2026-09-03-r161-images`, `scripts/r161-images.sh`, `scripts/mm_probe.py`)

[← all results](../RESULTS.md)

Question: the daily rejects prompts with more than 4 images (`--limit-mm-per-prompt '{"image":4}'`); an agent client that packed 5 screenshots into one message after compaction hit HTTP 400. What does a higher limit cost, and what does a per-image pixel cap buy? Three cells on the daily shape (RedHat NVFP4, fp8 KV, DFlash2 ns9, TP=2, util 0.92, tier on), fresh boot each: the daily itself (count 4), count 16, and count 16 with `--mm-processor-kwargs '{"max_pixels":1048576}'`. Probe: 12 synthetic 1468×1328 screenshots (one 4-digit code each), cold, warm resend, resend plus one image, then a 512-token generation with and without the images in context.

| cell | encoder budget / profiled items | KV pool | outcome |
|---|---|---|---|
| count 4 (daily) | 16,384 tokens / 1 image | 655,186 | reference |
| count 16 | 16,384 / 1 | 655,186 | identical boot; 12-image probe below |
| count 16 + 1 Mpx cap | 8,192 / 8 | 668,380 | engine OOM during the launcher pre-warm (240 MiB needed, 197 MiB free); no probe |

Why the count is free: vLLM profiles the encoder with `encoder_budget // max_tokens_per_item` images of maximum size. This checkpoint's processor allows 16.7 Mpx per image, so one image is 16,384 tokens and the profile encodes exactly one, whatever the count limit. The count only bounds `max_items_per_prompt` (already capped at `max_model_len // 16384 = 16`). Why the pixel cap is not: with 1,024 tokens per image the profile encodes 8 small images instead of one large one, the activation reserve shrinks, 13K more tokens go to KV, and the serve-time autotune workspace no longer fits at util 0.92. The same failure class as the 2026-07 util-0.98 retirement. A pixel cap would need util 0.90.

Count 16, single stream, 12 screenshots (23,249 prompt tokens, 1,930 per image):

| step | TTFT | prompt tokens | prefix-cache hit | decode | accepted per draft token |
|---|---|---|---|---|---|
| cold | 4.78 s | 23,249 | 0 | 374 t/s (code list) | 0.55 |
| warm resend | 0.58 s | 23,249 | 19,968 (12 of 13 full blocks) | 403 | 0.59 |
| plus one image | 1.00 s | 25,183 | 19,968 | 366 | 0.52 |
| 512-token essay, 12 images in context | 0.57 s | 23,240 | 19,968 | 191 | 0.22 |
| 512-token essay, text-only prompt of similar length | 2.49 s | 20,930 | 0 | 169 | 0.18 |

Reading: cold prefill of 12 screenshots runs at 4.9K tokens/s including the 12 encoder passes, a warm resend costs 0.6 s, and appending one image costs 0.4 s over warm. The prefix cache hits one block fewer than the full-block count on every resend (also 2 of 3 blocks on the 3-image smoke run), so per-image reuse is one 1,664-token block short of the token count. Decode with 12 images in context is not slower than text of the same length; the DFlash2 drafter does not see image embeddings, but acceptance on the image-dependent answer (0.55) is in the normal band. The code-listing answer was cut by the probe's 300-token budget on this cell (6 of 12 codes listed, all correct, in order), so image correctness at 12 images is not gated by this run; the 3-image smoke run against the daily read 3/3.

Not measured: 16 images under 8-stream concurrency with the 16,384-token encoder cache forcing chunked encodes. The daily was restored unchanged; the proposed change is `MMLIMIT='{"image":16,"video":0}'` in the daily launcher, no pixel cap. Host-side cost that applies at any count: `mm_processor_cache_gb` defaults to 4 GiB in both the API process and the engine core.
