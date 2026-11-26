# When does fine-tuning beat prompting? SEC 8-K filings

This project classifies SEC Form 8-K filings into the event types they report. A filing can have several, such as a new agreement, an executive change and earnings, so it's a multi-label task. It compares three approaches on the same time-based split, scored on test once:

- a TF-IDF baseline;
- four LLMs, prompted zero- and few-shot, from Anthropic and OpenAI;
- a fine-tuned 150M-parameter encoder (ModernBERT-base).

All three are compared on accuracy, cost and latency. The data is 8,128 public filings from EDGAR (2021–2026), labelled with the Item codes the SEC records, and the label headings are masked out of the text.

**Full method, per-class results and error analysis: [RESULTS.md](RESULTS.md).**

## Results (test set, 1,220 filings from Jun–Nov 2026)

| Model | Micro-F1 | Macro-F1 | All labels right | $ per 1K filings | Latency p50 |
|---|---|---|---|---|---|
| GPT-6.2 Sol, zero-shot | 0.949 | 0.951 | 91.9% | $4.58 | 1.13 s |
| **Hybrid: ModernBERT for short filings, Sol for the rest** | **0.957** | **0.953** | **92.5%** | **$2.76** | 56 ms / 1.1 s |
| Claude Sonnet 5.5, few-shot | 0.907 | 0.901 | 83.7% | $5.38 | 1.18 s |
| Claude Sonnet 5.5, zero-shot | 0.898 | 0.892 | 81.3% | $4.73 | 1.27 s |
| ModernBERT-base, fine-tuned (1,024 tokens) | 0.894 | 0.830 | 82.3% | ~$0 | 56 ms (laptop GPU) |
| TF-IDF + logistic regression | 0.881 | 0.831 | 73.0% | ~$0 | 0.9 ms (CPU) |
| GPT-6 Luna, few-shot | 0.829 | 0.817 | 67.8% | $0.19 | 0.81 s |
| Claude Haiku 5.5, few-shot | 0.822 | 0.770 | 61.0% | $0.30 | 0.83 s |

95% bootstrap confidence intervals, paired tests and the remaining configurations are in [RESULTS.md](RESULTS.md).

## What I found

- **The best frontier LLM is the most accurate single model.** A fine-tuned 150M-parameter encoder matches a mid-tier frontier LLM (Sonnet) on micro-F1, runs locally 20× faster and costs nothing per filing.
- **The fine-tune loses on long filings, not short ones.** On filings under 512 tokens, ModernBERT is right more often than the best LLM (94.7% vs 93.4% of filings fully correct). Over 1,024 tokens it drops to 52%.
- **So route by length.** Fine-tuned model on short filings, LLM on long ones, with the cutoff chosen on validation. On test this was at least as accurate as the LLM alone, for 40% less API cost.
- **Small LLMs lose to TF-IDF** on this task, mostly by over-labelling. Few-shot examples help them, and help frontier models little.
- **Prompt caching makes few-shot cheaper than zero-shot** for GPT-6.2 Sol. Without caching it would cost 3.5× more.
- **8.01 "Other events" is the hardest label for every model.** Whether a company files an announcement as 7.01 or 8.01 is partly a legal choice that the text doesn't reveal.

## How it's checked

- **Time-based split.** Train covers 2021–2025, validation Dec 2025–Jun 2026, test Jun–Nov 2026. Prompts, epochs, thresholds and the hybrid cutoff were all chosen on validation.
- **Label leakage masked.** 8-Ks contain their own "Item 5.02 Departure of Directors…" headings. Every Item code and official item title is replaced with `[ITEM]` before any model sees the text.
- **Contamination check.** Every LLM was rescored on test filings dated after all four models' published knowledge cutoffs. Scores changed by at most 0.006.
- **Uncertainty reported.** Bootstrap CIs on every metric, and paired bootstrap tests against the baseline.
- **Documented shortcuts.** For example, auditor-change filings cite "Item 304 of Regulation S-K", which makes Item 4.01 nearly free for every model.

## Repository

```
src/scrape.py          EDGAR index + download (rate-limited, cached, resumable)
src/build.py           HTML → text, label-leakage masking, labels, time-based split
src/baseline.py        TF-IDF + one-vs-rest logistic regression
src/llm.py             Anthropic / OpenAI classifier: structured output, prompt caching, response cache
src/finetune.py        ModernBERT-base fine-tune (runs on Apple-silicon MPS, CUDA or CPU)
src/evaluate.py        metrics, bootstrap CIs, paired tests, per-class tables, error dumps
src/benchmark.py       cost and latency per model
src/error_analysis.py  cross-model error analysis and the hybrid router
data/index.csv         the exact 8,136 filings used (accession numbers + SEC item codes)
data/labels.json       the 15 labels
results/               every model's predictions, metrics and error dumps
```

Not committed, because they can be rebuilt: the raw filings, the built dataset, LLM response caches and model checkpoints.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
export SEC_USER_AGENT="Your Name your@email.com"   # SEC requires a contact
uv run python src/scrape.py download   # the 8,136 filings in data/index.csv, ~30 min (SEC rate limit)
uv run python src/build.py             # text, masking, labels, split (~10 s)
uv run python src/baseline.py          # TF-IDF baseline (~10 s, no API keys or GPU)
uv run python src/evaluate.py          # rescores every committed prediction file
```

The LLM runs (about $31 in API calls) and the ModernBERT fine-tune (about 3 hours on an M-series MacBook, much less on a CUDA GPU) are optional. The predictions they produced are already in `results/`. Exact commands: [RESULTS.md § Reproduce](RESULTS.md#reproduce).

## Data and licence

- **Data:** filings are public records from [SEC EDGAR](https://www.sec.gov/edgar), accessed under the SEC's [fair-access policy](https://www.sec.gov/os/accessing-edgar-data) (at most 10 requests per second, with a declared User-Agent).
- **Code:** MIT; see [LICENSE](LICENSE).
- **Scope:** an independent personal project, built on personal time and hardware with public data only.
