# Cost and latency (test split)

Measured 2026-11-25 on Apple M5, 10 cores, Darwin 25.6.0. Local models: one filing per call, after warm-up. LLMs: one API request per filing as the client saw it (includes network and any 429 retries).

| Model | Where | p50 | p95 | p99 | $ per 1K filings | $ per 1K, no prompt caching | Tokens per filing (in / cached / out) | Filings timed |
|---|---|---|---|---|---|---|---|---|
| tfidf | CPU | 0.9 ms | 1.6 ms | 2.3 ms | 0.00 | — | — | 1,220 |
| modernbert_1024 | MPS | 56 ms | 124 ms | 125 ms | 0.00 | — | — | 1,220 |
| modernbert_1024 | CPU | 140 ms | 292 ms | 296 ms | 0.00 | — | — | 200 |
| gpt6luna_zeroshot | API | 779 ms | 1.21 s | 1.65 s | 0.23 | 0.23 | 2,207 / 0 / 17 | 1,220 |
| gpt6luna_fewshot | API | 808 ms | 1.49 s | 2.04 s | 0.19 | 0.58 | 1,374 / 4,340 / 17 | 1,220 |
| haiku55_fewshot | API | 827 ms | 1.44 s | 1.81 s | 0.30 | 0.95 | 2,196 / 7,197 / 18 | 1,220 |
| haiku55_zeroshot | API | 900 ms | 1.54 s | 1.92 s | 0.25 | 0.39 | 2,196 / 1,592 / 17 | 1,220 |
| gpt62sol_zeroshot | API | 1.13 s | 1.59 s | 2.38 s | 4.57 | 4.57 | 2,207 / 0 / 16 | 1,220 |
| sonnet55_fewshot | API | 1.18 s | 1.68 s | 2.04 s | 5.38 | 18.94 | 2,196 / 7,150 / 15 | 1,220 |
| gpt62sol_fewshot | API | 1.21 s | 2.42 s | 2.97 s | 3.27 | 11.59 | 1,335 / 4,380 / 16 | 1,220 |
| sonnet55_zeroshot | API | 1.27 s | 1.94 s | 2.43 s | 4.73 | 7.75 | 2,196 / 1,592 / 15 | 1,220 |

Training time (one-off): modernbert_1024 182.1 min; tfidf 4 s.

Local models have no per-filing charge; the machine is the cost. LLM prices are list prices as logged with each run.
