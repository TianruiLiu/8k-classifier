# Results

**Task:** multi-label classification of SEC Form 8-K filings into the event types they report (8-K Item codes), from the filing text alone.
**Question this project answers:** when does fine-tuning a small model beat prompting a frontier LLM on financial filings, on accuracy, cost and latency?

Every model follows the same protocol: fit on train only, choose settings and one global decision threshold on validation, score test once.

## Findings

1. **The best frontier LLM wins on accuracy.** GPT-6.2 Sol, zero-shot, gets every label right on 91.9% of test filings (micro-F1 0.949). No other single model comes within 4 points.
2. **A fine-tuned 150M-parameter encoder matches a mid-tier frontier LLM for almost nothing.** ModernBERT-base at 1,024 tokens reaches micro-F1 0.894, statistically level with Claude Sonnet 5.5 zero-shot (0.898), with higher subset accuracy (0.823 vs 0.813). It runs at 56 ms per filing on a laptop GPU versus about 1.2 s per API call, with no per-filing charge. Sonnet keeps a clear lead on macro-F1 (0.892 vs 0.830), from the rare classes.
3. **The fine-tune's remaining gap is mostly input length.** On filings that fit in 512 tokens, ModernBERT is fully right more often than Sol (94.7% vs 93.4%). On filings over 1,024 tokens it falls to 52.3% while Sol holds 91.3%. Going from 512 to 1,024 tokens alone raised ModernBERT's test micro-F1 from 0.868 to 0.894.
   - **That suggests a hybrid,** and it works: short filings go to ModernBERT and the rest to Sol, with the cutoff chosen on validation.
   - It is at least as accurate as Sol alone: micro-F1 0.957 vs 0.949 (+0.007 [+0.001, +0.014]), subset accuracy 0.925 vs 0.919 (difference not significant).
   - It costs $2.76 instead of $4.58 per 1K filings.
4. **Small LLMs lose to a TF-IDF baseline.** GPT-6 Luna and Claude Haiku 5.5 score 0.78–0.83 micro-F1 against TF-IDF's 0.881, mostly by over-labelling. Few-shot examples help them most (+0.04 micro), and help the frontier models little or not at all.
5. **Prompt caching decides whether few-shot is affordable.** The 15 examples add 3,500–5,600 tokens per call, depending on the tokenizer. Cached, few-shot Sol is cheaper than zero-shot ($3.27 vs $4.58 per 1K filings); uncached it would cost $11.59.
6. **8.01 ("other events") is the hardest label for every model,** and most of the filings that every model gets wrong involve it. It is a catch-all whose use is partly the filer's judgment, so some of these errors are probably label noise rather than model failures (hand check pending).

## Summary (test, 1,220 filings)

| Model | Micro-F1 [95% CI] | Macro-F1 [95% CI] | Subset acc | $ per 1K filings | p50 latency | Notes |
|---|---|---|---|---|---|---|
| GPT-6.2 Sol, zero-shot | **0.949** [0.940, 0.959] | 0.951 [0.936, 0.964] | **0.919** | 4.58 | 1.13 s | reasoning effort `low` (lowest it accepts) |
| GPT-6.2 Sol, few-shot | 0.948 [0.939, 0.958] | **0.952** [0.937, 0.964] | 0.917 | 3.27 | 1.21 s | |
| *Hybrid: ModernBERT ≤ 512 tokens, else Sol zero-shot* | *0.957* | *0.953* | *0.925* | *2.76* | *56 ms or 1.1 s* | *cutoff chosen on val; see "Error analysis"* |
| Claude Sonnet 5.5, few-shot | 0.907 [0.894, 0.919] | 0.901 [0.879, 0.916] | 0.837 | 5.38 | 1.18 s | |
| Claude Sonnet 5.5, zero-shot | 0.898 [0.885, 0.910] | 0.892 [0.869, 0.908] | 0.813 | 4.73 | 1.27 s | thinking off |
| **ModernBERT-base, fine-tuned, 1,024 tokens** | 0.894 [0.879, 0.908] | 0.830 [0.794, 0.855] | 0.823 | ~0 | 56 ms (laptop GPU), 141 ms (CPU) | 182 min training on an Apple M5 laptop |
| TF-IDF + LR (1–2-grams) | 0.881 [0.870, 0.893] | 0.831 [0.801, 0.854] | 0.730 | ~0 | 0.9 ms (CPU) | fits in ~5 s |
| GPT-6 Luna, few-shot | 0.829 [0.813, 0.845] | 0.817 [0.791, 0.837] | 0.678 | 0.19 | 0.81 s | |
| Claude Haiku 5.5, few-shot | 0.822 [0.808, 0.835] | 0.770 [0.745, 0.793] | 0.610 | 0.30 | 0.83 s | |
| GPT-6 Luna, zero-shot | 0.788 [0.772, 0.802] | 0.781 [0.751, 0.804] | 0.541 | 0.23 | 0.78 s | reasoning effort `none` |
| Claude Haiku 5.5, zero-shot | 0.781 [0.767, 0.795] | 0.748 [0.725, 0.768] | 0.478 | 0.25 | 0.90 s | thinking off |

- **Subset accuracy:** the share of filings where every label is right.
- **Confidence intervals:** 95% bootstrap intervals over test filings, from 1,000 resamples.
- **Differences from TF-IDF:** tested with a paired bootstrap (`results/compare_test.md`).
  - Every LLM row differs from TF-IDF on micro-F1.
  - ModernBERT's +0.013 is borderline: its CI is [+0.000, +0.026].
  - On macro-F1, ModernBERT ties TF-IDF (−0.001) and so does Luna few-shot (−0.013).
- **Costs:** list prices on Nov 24, 2026, applied to logged token counts.
- **Latency:** one filing per call; details are under "Cost and latency".
- **Same inputs for every LLM:** the same frozen prompt, and a JSON schema limited to the 15 labels.
- **Baseline on validation:** TF-IDF scores micro-F1 0.882, macro-F1 0.791, subset accuracy 0.749.

## How configurations were chosen

Test was scored once per configuration. The only decisions made on validation, and what they were:

- **Prompt.** Tuned on a fixed sample of 200 validation filings with Sonnet zero-shot. One variant added an explicit "7.01 vs 8.01" rule. It lowered validation micro-F1 from 0.896 to 0.878 and 8.01 F1 from 0.73 to 0.67, so it was rejected and the original prompt frozen. Its outputs are kept in `results/rejected/`.
- **LLM settings.** Thinking or reasoning was set to the lowest level each model accepts. A short JSON answer doesn't need reasoning, and on Luna reasoning added ~180 output tokens per filing.
- **Few-shot.** One example per label (15 in all) is drawn from train, preferring filings with a single label. Each example is the first 1,000 characters after the cover page.
  - Zero-shot and few-shot were fixed in advance as two configurations of each model, and both are reported.
  - Validation didn't pick between them. On the 200 validation filings, Sonnet few-shot was slightly below zero-shot (0.885 vs 0.896); on test it was slightly above (0.907 vs 0.898).
  - Both gaps sit well inside their CIs. Read it as "few-shot doesn't matter much for Sonnet", not as a gain.
- **Fine-tuning.** Three ModernBERT runs. The final one was chosen on validation micro-F1 (table below).
- **Thresholds.** Each probabilistic model (TF-IDF, ModernBERT) uses one global threshold picked on validation. The LLMs output labels directly.

## Fine-tuning ModernBERT

ModernBERT-base (150M parameters) was fine-tuned with a 15-way sigmoid head and binary cross-entropy loss.

- **Hardware:** an Apple M5 laptop GPU (MPS); no cloud GPU.
- **Optimizer:** AdamW, lr 5e-5, weight decay 0.01, 10% linear warmup, effective batch 8.
- **Checkpoint:** the best epoch by validation micro-F1 is kept.
- **Input:** starts at the first `[ITEM]` marker. The cover page before it is identical on every 8-K, so skipping it spends the token budget on the event itself.

| Run | Max tokens | Epochs | Train filings truncated | Val micro / macro (best epoch) | Test micro / macro | Test subset acc | Training time |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 3 | 42% | 0.890 / 0.799 | 0.862 / 0.761 | 0.757 | 72 min |
| 2 | 512 | 5 | 42% | 0.897 / 0.822 | 0.868 / 0.795 | 0.766 | 121 min |
| **3** | **1,024** | **5** | **16%** | **0.917 / 0.848** | **0.894 / 0.830** | **0.823** | **182 min** |

- **Training curve:** validation scores were still rising at the last epoch in every run (run 3: 0.845, 0.885, 0.910, 0.911, 0.917), so more epochs might still help a little.
- **What longer input fixed:** going from 512 to 1,024 tokens mostly helped classes whose deciding sentence often comes late in the filing. Test F1 by class:

  | Class | 512 tokens | 1,024 tokens |
  |---|---|---|
  | 1.02 | 0.40 | 0.64 |
  | 3.02 | 0.74 | 0.83 |
  | 3.03 | 0.58 | 0.68 |
  | 7.01 | 0.84 | 0.88 |
  | 8.01 | 0.78 | 0.81 |

- **Memory:** run 3 used batch 2 with 4 gradient-accumulation steps, because batch 4 at 1,024 tokens ran out of GPU memory.
- **Not tried:** rare-label loss weighting. It targets two small classes (`other` with 16 test filings, 5.03 with 55), where any gain would sit inside the noise.

## Training-data contamination check

A model that saw these filings during training could recall their Item codes from EDGAR, inflating its score. The test set covers filings dated Jun 2 – Nov 20, 2026, so each model's published knowledge cutoff was checked against that window:

| Model | Knowledge cutoff (published) | Overlaps the test window? |
|---|---|---|
| GPT-6 Luna | May 18, 2026 | No |
| GPT-6.2 Sol | Jun 1, 2026 | No (test starts Jun 2) |
| Claude Sonnet 5.5 | Jun 2026 (month only) | Possibly June |
| Claude Haiku 5.5 | Jun 2026 (month only) | Possibly June |

Every model was then rescored on the 1,042 test filings dated Jul 1, 2026 or later, after all four cutoffs (178 test filings are from June):

| Model | All test (micro / macro) | Jul 1 onward only (micro / macro) |
|---|---|---|
| GPT-6.2 Sol | 0.949 / 0.951 | 0.948 / 0.952 |
| Claude Sonnet 5.5 | 0.898 / 0.892 | 0.898 / 0.894 |
| TF-IDF + LR | 0.881 / 0.831 | 0.878 / 0.826 |
| GPT-6 Luna | 0.788 / 0.781 | 0.789 / 0.784 |
| Claude Haiku 5.5 | 0.781 / 0.748 | 0.783 / 0.754 |

Scores on filings no model could have seen match the full test set to within 0.006, and the ranking is unchanged, so memorized filings do not explain the results. The time-based split is what makes this check possible: with a random split, test filings would be spread across years that every model's training data covers. (ModernBERT-base was released in Dec 2024 and fine-tuned only on this project's train split, which ends Dec 2025.)

Cutoff sources (checked Nov 24, 2026): [Claude models overview](https://platform.claude.com/docs/en/about-claude/models/overview.md), [GPT-6.2 Sol](https://developers.openai.com/api/docs/models/gpt-6.2-sol), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

## Per-class F1 (test)

| Item | Event | Support | TF-IDF | ModernBERT 1,024 | Haiku few-shot | Sonnet zero-shot | Sonnet few-shot | Sol zero-shot |
|---|---|---|---|---|---|---|---|---|
| 1.01 | Entry into a material definitive agreement | 199 | 0.91 | 0.93 | 0.81 | 0.91 | 0.91 | **0.96** |
| 1.02 | Termination of a material definitive agreement | 27 | 0.36 | 0.64 | 0.79 | 0.87 | 0.83 | **0.96** |
| 2.01 | Completion of acquisition or disposition | 35 | 0.74 | 0.71 | 0.70 | 0.89 | 0.89 | **0.96** |
| 2.02 | Results of operations | 389 | 0.96 | **0.99** | 0.98 | 0.98 | **0.99** | **0.99** |
| 2.03 | Creation of a direct financial obligation | 77 | 0.84 | 0.84 | 0.75 | 0.84 | 0.85 | **0.94** |
| 3.01 | Delisting / listing-standard notice | 23 | 0.82 | 0.81 | 0.82 | 0.84 | 0.85 | **0.88** |
| 3.02 | Unregistered sales of equity | 72 | 0.84 | 0.83 | 0.76 | 0.90 | 0.90 | **0.96** |
| 3.03 | Material modification of holder rights | 29 | 0.82 | 0.68 | 0.62 | 0.79 | 0.83 | **0.96** |
| 4.01 | Change in certifying accountant | 17 | **1.00*** | **1.00** | 0.97 | **1.00** | **1.00** | **1.00** |
| 5.02 | Director / officer changes | 220 | 0.93 | 0.93 | 0.94 | 0.98 | 0.98 | **0.99** |
| 5.03 | Amendments to articles or bylaws | 55 | 0.93 | 0.85 | 0.77 | 0.94 | 0.94 | **0.98** |
| 5.07 | Shareholder vote results | 98 | 0.98 | 0.96 | 0.98 | **0.99** | **0.99** | **0.99** |
| 7.01 | Regulation FD disclosure | 341 | 0.88 | 0.88 | 0.74 | 0.86 | 0.89 | **0.93** |
| 8.01 | Other events | 285 | 0.77 | 0.81 | 0.69 | 0.73 | 0.74 | **0.85** |
| other | 11 rare items, pooled (see Data) | 16 | 0.69 | 0.58 | 0.21 | 0.86 | **0.93** | 0.90 |

Classes with under 30 test filings (1.02, 3.01, 3.03, 4.01, `other`) have wide intervals, often ±0.15–0.25. `results/compare_test.md` has every model with 95% CIs per class.
\* Helped by a citation shortcut; see "Known shortcuts".

- **Easy classes:** on 2.02, 4.01, 5.07 and 5.02, every model except Haiku is at 0.93 or above. An LLM adds little here and costs far more per filing.
- **Classes that need reading:** on 1.02 (terminations), 2.01 (completed deals) and `other`, the frontier LLMs gain the most. Telling a termination from a new agreement takes reading what happened to the agreement. Bag-of-words sees the same vocabulary in both, and TF-IDF's 1.02 recall is 0.26.
- **ModernBERT's weak spots:** it lags TF-IDF on 3.03, 5.03 and `other`. Two reasons:
  - Two of these classes are rare in training: 3.03 has 76 training filings and `other` has 88, a mix of 11 unrelated events (5.03 has 201).
  - Some hinge on a phrase that TF-IDF catches anywhere in a long filing, while ModernBERT never sees text past 1,024 tokens.
- **The M1 hypothesis held:** LLM gains are concentrated in 1.02, 2.01 and `other`. On 8.01 only Sol beats the baseline. Sonnet does worse than TF-IDF there, because it routes "other events" to 7.01 (see below).

## Cost and latency

Measured by `src/benchmark.py` on the 1,220 test filings, one filing per call. Full table: `results/benchmark.md`.

| Model | Where it runs | p50 | p95 | $ per 1K filings | $ per 1K without prompt caching |
|---|---|---|---|---|---|
| TF-IDF + LR | laptop CPU | 0.9 ms | 1.6 ms | ~0 | — |
| ModernBERT, 1,024 tokens | laptop GPU (MPS) | 56 ms | 124 ms | ~0 | — |
| ModernBERT, 1,024 tokens | laptop CPU (200-filing sample) | 141 ms | 292 ms | ~0 | — |
| GPT-6 Luna, zero / few-shot | API | 0.78 / 0.81 s | 1.21 / 1.49 s | 0.23 / 0.19 | 0.23 / 0.58 |
| Claude Haiku 5.5, zero / few-shot | API | 0.90 / 0.83 s | 1.54 / 1.44 s | 0.25 / 0.30 | 0.39 / 0.95 |
| GPT-6.2 Sol, zero / few-shot | API | 1.13 / 1.21 s | 1.59 / 2.42 s | 4.58 / 3.27 | 4.58 / 11.59 |
| Claude Sonnet 5.5, zero / few-shot | API | 1.27 / 1.18 s | 1.94 / 1.68 s | 4.73 / 5.38 | 7.75 / 18.94 |

- **Latency:**
  - For local models, it's tokenizing plus one forward pass, one filing at a time, after warm-up.
  - For LLMs, it's the wall time of one request as the client saw it. That includes the network and any SDK retries after a rate-limit error, with 3–8 requests in flight.
  - Batched, ModernBERT runs at 76 ms per filing on MPS (`results/metrics_modernbert_1024.json`).
- **Cost:**
  - **Tokens per call:** about 2,200 input tokens per filing, and every LLM answers in 15–18 output tokens. Cost is therefore almost all input, and prompt caching (the system prompt, plus the few-shot examples when used) is the main lever.
  - **Spend:** the whole project's LLM spend, validation and test, was about $31.
  - **Fine-tuning:** ModernBERT needed no cloud GPU and no API spend. It took 3 hours of laptop time, about as long as sending 10K filings to Sol's API one at a time.
- **What fine-tuning actually costs** isn't compute here. It's labeled data (free from EDGAR for this task, rarely free elsewhere), engineering time, and the remaining accuracy gap to the best LLM.

## Error analysis

From `src/error_analysis.py` (full output, with examples: `results/error_analysis.md`; per-model error dumps: `results/errors_<model>.md`).

**How each model goes wrong.** Each model's misses split into three kinds:
- **Over-labelled:** all the true labels, plus extras.
- **Under-labelled:** a strict subset of the true labels, with nothing extra.
- **Swapped:** at least one label missed and one wrong.

| Model | All right | Over-labelled | Under-labelled | Swapped | Labels predicted per filing (true: 1.54) |
|---|---|---|---|---|---|
| TF-IDF | 73.0% | 10.6% | 10.9% | 5.5% | 1.53 |
| ModernBERT 1,024 | 82.3% | 5.2% | 5.9% | 6.6% | 1.50 |
| Haiku few-shot | 61.0% | 22.8% | 3.1% | 13.1% | 1.78 |
| Sonnet few-shot | 83.7% | 4.8% | 1.0% | 10.6% | 1.60 |
| Sol zero-shot | 91.9% | 0.2% | 0.5% | 7.5% | 1.54 |

- **The small LLMs over-label.** Haiku predicts 7.01 wrongly 216 times out of 1,220 filings: it reads any press-release language as a Regulation FD disclosure.
- **The frontier LLMs almost never over- or under-label.** Their errors are swaps, picking the wrong item of two plausible ones.
- **The trained models (TF-IDF, ModernBERT) err in both directions about equally.** Their threshold is tuned for exactly that balance.

**The main confusion is 8.01 vs 7.01.** Pairs of a true label missed and a wrong label predicted on the same filing:

| True → predicted instead | TF-IDF | ModernBERT | Haiku few-shot | Sonnet few-shot | Sol zero-shot |
|---|---|---|---|---|---|
| 8.01 → 7.01 | 9 | 14 | 44 | 53 | 35 |
| 8.01 → 1.01 | 6 | 7 | 8 | 28 | 13 |
| 7.01 → 8.01 | 8 | 17 | 9 | 8 | 6 |
| 8.01 → 2.03 | 4 | 5 | 16 | 15 | 3 |

Companies often announce an "other event" with a press release. Whether they file it under 7.01 (furnished), 8.01 (filed), or both is a legal choice that the text doesn't always reveal. The LLMs read the definitions and guess 7.01. TF-IDF and ModernBERT learned the filers' actual habits from 5,689 training examples, so they make this mistake far less often.

**Long filings are where the fine-tune loses.** Subset accuracy by length (ModernBERT tokens after the cover page):

| Length | Filings | TF-IDF | ModernBERT 512 | ModernBERT 1,024 | Sol zero-shot |
|---|---|---|---|---|---|
| ≤ 512 tokens | 624 | 86.5% | 93.4% | **94.7%** | 93.4% |
| 513–1,024 | 319 | 72.1% | 75.2% | 84.0% | **89.3%** |
| > 1,024 | 277 | 43.7% | 40.1% | 52.3% | **91.3%** |

On short filings, ModernBERT is as good as the best LLM. The gap is entirely in long, multi-event filings, which are also harder for TF-IDF. Sol reads the whole filing and barely degrades.

**The two approaches fail on different filings.**

| | Sol right | Sol wrong |
|---|---|---|
| ModernBERT right | 955 | 49 |
| ModernBERT wrong | 166 | 50 |

- **Oracle:** picking the right model per filing would reach 95.9% subset accuracy, vs 91.9% for Sol alone.
- **A rule you can actually deploy:** route by filing length, with no labels needed at inference time. The cutoff was chosen on the 200 validation filings Sol scored, then test was scored once:

  | Split | Routed to ModernBERT | Hybrid micro / macro | Hybrid subset | Sol alone micro / macro | Sol alone subset |
  |---|---|---|---|---|---|
  | val, cutoff 512 tokens (chosen) | 64% | 0.958 / 0.879 | 0.925 | 0.941 / 0.915 | 0.905 |
  | val, cutoff 1,024 tokens | 86% | 0.954 / 0.867 | 0.910 | 0.941 / 0.915 | 0.905 |
  | **test, cutoff 512 tokens** | **51%** | **0.957 / 0.953** | **0.925** | 0.949 / 0.951 | 0.919 |

  - **Accuracy on test:** micro-F1 is +0.007 [+0.001, +0.014] (paired bootstrap), and subset accuracy +0.006, which is not significant.
  - **Cost:** $2.76 per 1K filings, 40% less than Sol alone. Long filings use more tokens, so the saving is smaller than the 51% routed share.
  - **Validation sample:** the cutoff was picked on only 200 filings and only two cutoffs were tried, so treat 512 as a reasonable default rather than a tuned optimum.
  - **Macro-F1 on validation:** the hybrid was lower than Sol alone there (0.879 vs 0.915), because ModernBERT is weaker on the rare classes. On test the two were level. Where rare events matter most, prefer the LLM alone.

**40 filings (3.3%) are wrong for every model.**
- **Labels involved:** 30 involve 8.01, 11 involve 7.01 and 10 involve 5.02. Several come from the same company (CIK 802257), which repeatedly files 5.02 together with 8.01.
- **Why they matter:** these are the first place to look for label noise, i.e. cases where the filer's chosen items aren't recoverable from the text.
- **Status:** they are listed with EDGAR links in `results/error_analysis.md`. *Hand check pending; conclusions about label noise wait on it.*

## When to fine-tune and when to prompt

Applies to this task and these models; the numbers are above.

- **Prompt the best frontier model** when accuracy is what matters and volume is modest. It is the only single model above 0.90 subset accuracy, needs no training data, and costs ~$4.6 per 1K filings.
- **Fine-tune a small encoder** when volume is high, latency matters, or data can't leave your machines. You get Sonnet-level micro-F1 at ~20× lower latency and no per-filing cost. It needs a few thousand labeled examples, which EDGAR provides for free here.
- **Don't use a small LLM out of the box.** On this task, Haiku and Luna are below a 5-second TF-IDF baseline. Their only edge is that they need no training data.
- **Start with TF-IDF.** It gets 0.88 micro-F1 in 5 seconds and shows which classes are keyword problems (where nothing else helps much) and which need reading (where LLMs earn their cost).
- **For production, use both:** the fine-tuned model on short filings and the LLM on long ones. On this test set that was at least as accurate as the LLM alone, at 40% lower API cost, with half the filings answered in about 56 ms.

## Data

- **Source:** SEC EDGAR, public. 400 companies sampled at random (seed 42) from EDGAR's ticker list; each company's 8-Ks filed Jan 2021 – Nov 2026, amendments (8-K/A) excluded, at most the 40 most recent per company so heavy filers don't dominate. 8,136 filings downloaded, 0 failed; 8,128 kept.
- **Labels:** the Item codes EDGAR records for each filing, not parsed from the text.
  - **9.01** (financial statements and exhibits) is dropped: it's attached to most filings and says nothing about the event. 8 filings had no other label and were removed.
  - Codes with fewer than 50 filings are pooled into **`other`**: 1.03, 1.04, 2.04, 2.05, 2.06, 4.02, 5.01, 5.04, 5.05, 5.06, 5.08.
  - Result: 15 labels, 1.43 labels per filing on average (1.54 in test).
- **Text:** the primary 8-K document only (exhibits such as press releases are separate files and not included). Visible text only: the inline-XBRL header, hidden elements, scripts and styles are stripped; capped at 100K characters.
- **Split:** time-based, by filing date. Train 5,689 (Jan 2021 – Dec 2025), validation 1,219 (Dec 2025 – Jun 2026), test 1,220 (Jun – Nov 2026). This mirrors real use, predicting future filings, and keeps near-duplicate filings from the same company and week out of both sides. Because of the per-company cap, the data skews recent.

## Label-leakage masking

8-K text contains its own section headings ("Item 5.02 Departure of Directors or Certain Officers..."). Left in, any model reads the answer off the heading. `src/build.py` replaces with `[ITEM]`:

- every Item code ("Item 5.02", "Items 2.02 and 9.01"): 25,203 occurrences;
- every official item title, each part of multi-part titles matched separately, tolerant of punctuation and apostrophe variants: 22,835 occurrences;
- runs of adjacent masks collapsed into one, since the number of title parts (5.02 has four) would still hint at the label.

After masking, 0 filings contain an Item code.

## Known shortcuts (left in on purpose, documented)

- **4.01 and "Item 304".** Auditor-change filings cite "Item 304 of Regulation S-K". That citation is genuine filing text rather than a heading, but it acts as a near-label: it's 4.01's top TF-IDF feature, and every model scores 0.94–1.00 on 4.01. Masking Regulation S-K citations would make 4.01 harder; that ablation was not run.
- **The mask token is a feature.** `[ITEM]` tokenizes to "item", so cross-references like "the information set forth in [ITEM] is incorporated into this [ITEM]" appear among 2.03's top terms. This signals that a filing reports several items, not which ones.
- **7.01 boilerplate.** "Shall not be deemed filed… incorporated by reference" is standard language for furnished (7.01, 2.02) information and ranks high for 7.01.
- **Some memorization.** A date ("april 19") ranks in 2.01's top TF-IDF terms. Minor at this data size.

## Limitations

- **One run per configuration.** Fine-tuning variance across seeds isn't measured, and LLM sampling isn't repeated. The bootstrap CIs cover test-set sampling only.
- **Small LLM tuning sample.** Prompt and settings were tuned on 200 validation filings, where differences under ~0.04 aren't distinguishable.
- **Primary document only.** Press releases attached as exhibits often state the event more plainly, so including them could help every model.
- **One fine-tuned model.** A LoRA-tuned 1–8B decoder, or a longer-context encoder setting (ModernBERT supports 8K tokens), could close the long-filing gap. Neither was run.
- **Latency setup.** Local latency is on one laptop. API latency depends on load, region and rate-limit tier.

## Reproduce

```bash
export SEC_USER_AGENT="Your Name your@email.com"   # SEC requires a contact
uv run python src/scrape.py index        # ~2 min
uv run python src/scrape.py download     # ~30 min at ~5 requests/sec
uv run python src/build.py               # ~10 s
uv run python src/baseline.py            # ~10 s

# LLMs (keys in ANTHROPIC_API_KEY / OPENAI_API_KEY); tune on --split val --n 200 first, then:
uv run python src/llm.py --provider openai --model gpt-6.2-sol --name gpt62sol_zeroshot --effort low --split test --concurrency 3 --price 2 10 0.1
uv run python src/llm.py --provider anthropic --model claude-sonnet-5-5 --name sonnet55_zeroshot --split test --price 2 10 0.1
#   few-shot: --mode fewshot --name <model>_fewshot
#   gpt-6-luna: --effort none --price 0.1 0.5 0.01; claude-haiku-5-5: --price 0.1 0.5 0.01

# fine-tune (~3 h on an Apple M5 laptop; no internet once weights are cached)
uv run python src/finetune.py --epochs 5 --max-len 1024 --batch-size 2 --grad-accum 4 --name modernbert_1024

uv run python src/evaluate.py            # CIs, paired tests, per-class tables, error dumps
uv run python src/benchmark.py           # cost and latency (~2 min)
uv run python src/error_analysis.py      # cross-model error analysis
```

Baseline settings: TF-IDF on 1–2-grams (min_df 3, max_df 0.9, up to 200K features, sublinear tf; 82,592 terms after filtering); one-vs-rest logistic regression (liblinear, C=4, balanced class weights); seed 42.
