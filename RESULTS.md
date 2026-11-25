# Results

**Task:** multi-label classification of SEC Form 8-K filings into the event types they report (8-K Item codes), from the filing text alone.
**Question this project answers:** when does fine-tuning a small model beat prompting a frontier LLM on financial filings, on accuracy, cost and latency?

Every model follows the same protocol: fit on train only, choose one global decision threshold on validation, score test once.

## Summary

| Model | Micro-F1 [95% CI] | Macro-F1 [95% CI] | Subset acc | $ per 1K filings | Notes |
|---|---|---|---|---|---|
| GPT-6.2 Sol, zero-shot | **0.949** [0.940, 0.959] | **0.951** [0.936, 0.964] | **0.919** | 4.58 | reasoning effort `low` (lowest it accepts) |
| Claude Sonnet 5.5, zero-shot | 0.898 [0.885, 0.910] | 0.892 [0.869, 0.908] | 0.813 | 4.73 | thinking off |
| TF-IDF + LR (1–2-grams) | 0.881 [0.870, 0.893] | 0.831 [0.801, 0.854] | 0.730 | ~0 | threshold 0.60 from val; 5,689 train / 1,220 test; fits in ~5 s on a laptop CPU |
| GPT-6 Luna, zero-shot | 0.788 [0.772, 0.802] | 0.781 [0.751, 0.804] | 0.541 | 0.23 | reasoning effort `none` |
| Claude Haiku 5.5, zero-shot | 0.781 [0.767, 0.795] | 0.748 [0.725, 0.768] | 0.478 | 0.24 | thinking off |
| LLM few-shot | — | — | — | — | M2 |
| Fine-tuned small model | — | — | — | — | M3 |

Subset accuracy = share of filings where every label is right. 95% CIs are bootstrap intervals over test filings (1,000 resamples); every difference from TF-IDF above is significant (paired bootstrap, CI excludes 0). Costs are at list prices on Nov 24, 2026, from logged tokens. All LLMs use the same frozen prompt and a JSON schema limited to the 15 labels; per-class tables for every model are in `results/compare_test.md`. Validation scores for the baseline: micro-F1 0.882, macro-F1 0.791, subset accuracy 0.749.

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

Scores on filings no model could have seen match the full test set to within 0.006, and the ranking is unchanged, so memorized filings do not explain the results. The time-based split is what makes this check possible: with a random split, test filings would be spread across years that every model's training data covers.

Cutoff sources (checked Nov 24, 2026): [Claude models overview](https://platform.claude.com/docs/en/about-claude/models/overview.md), [GPT-6.2 Sol](https://developers.openai.com/api/docs/models/gpt-6.2-sol), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

## Per-class F1 (test)

| Item | Event | Precision | Recall | F1 | Test support |
|---|---|---|---|---|---|
| 1.01 | Entry into a material definitive agreement | 0.86 | 0.97 | 0.91 | 199 |
| 1.02 | Termination of a material definitive agreement | 0.58 | 0.26 | **0.36** | 27 |
| 2.01 | Completion of acquisition or disposition | 0.68 | 0.80 | 0.74 | 35 |
| 2.02 | Results of operations | 0.96 | 0.95 | 0.96 | 389 |
| 2.03 | Creation of a direct financial obligation | 0.76 | 0.92 | 0.84 | 77 |
| 3.01 | Delisting / listing-standard notice | 0.75 | 0.91 | 0.82 | 23 |
| 3.02 | Unregistered sales of equity | 0.77 | 0.92 | 0.84 | 72 |
| 3.03 | Material modification of holder rights | 0.78 | 0.86 | 0.82 | 29 |
| 4.01 | Change in certifying accountant | 1.00 | 1.00 | 1.00* | 17 |
| 5.02 | Director / officer changes | 0.96 | 0.90 | 0.93 | 220 |
| 5.03 | Amendments to articles or bylaws | 0.94 | 0.91 | 0.93 | 55 |
| 5.07 | Shareholder vote results | 1.00 | 0.97 | 0.98 | 98 |
| 7.01 | Regulation FD disclosure | 0.90 | 0.86 | 0.88 | 341 |
| 8.01 | Other events | 0.80 | 0.74 | 0.77 | 285 |
| other | 11 rare items, pooled (see below) | 0.90 | 0.56 | **0.69** | 16 |

\* Helped by a citation shortcut; see "Known shortcuts".

## Data

- **Source:** SEC EDGAR, public. 400 companies sampled at random (seed 42) from EDGAR's ticker list; each company's 8-Ks filed Jan 2021 – Nov 2026, amendments (8-K/A) excluded, at most the 40 most recent per company so heavy filers don't dominate. 8,136 filings downloaded, 0 failed; 8,128 kept.
- **Labels:** the Item codes EDGAR records for each filing, not parsed from the text.
  - **9.01** (financial statements and exhibits) is dropped: it's attached to most filings and says nothing about the event. 8 filings had no other label and were removed.
  - Codes with fewer than 50 filings are pooled into **`other`**: 1.03, 1.04, 2.04, 2.05, 2.06, 4.02, 5.01, 5.04, 5.05, 5.06, 5.08.
  - Result: 15 labels, 1.43 labels per filing on average.
- **Text:** the primary 8-K document only (exhibits such as press releases are separate files and not included). Visible text only: the inline-XBRL header, hidden elements, scripts and styles are stripped; capped at 100K characters.
- **Split:** time-based, by filing date. Train 5,689 (Jan 2021 – Dec 2025), validation 1,219 (Dec 2025 – Jun 2026), test 1,220 (Jun – Nov 2026). This mirrors real use, predicting future filings, and keeps near-duplicate filings from the same company and week out of both sides. Because of the per-company cap, the data skews recent.

## Label-leakage masking

8-K text contains its own section headings ("Item 5.02 Departure of Directors or Certain Officers..."). Left in, any model reads the answer off the heading. `src/build.py` replaces with `[ITEM]`:

- every Item code ("Item 5.02", "Items 2.02 and 9.01"): 25,203 occurrences;
- every official item title, each part of multi-part titles matched separately, tolerant of punctuation and apostrophe variants: 22,835 occurrences;
- runs of adjacent masks collapsed into one, since the number of title parts (5.02 has four) would still hint at the label.

After masking, 0 filings contain an Item code.

## Known shortcuts (left in on purpose, documented)

- **4.01 and "Item 304".** Auditor-change filings cite "Item 304 of Regulation S-K", and that citation is 4.01's top feature. It is genuine filing text rather than a heading, but it acts as a near-label. Masking Regulation S-K citations would make 4.01 harder; that ablation is planned for M2.
- **The mask token is a feature.** `[ITEM]` tokenizes to "item", so cross-references like "the information set forth in [ITEM] is incorporated into this [ITEM]" appear among 2.03's top terms. This signals that a filing reports several items, not which ones.
- **7.01 boilerplate.** "Shall not be deemed filed… incorporated by reference" is standard language for furnished (7.01, 2.02) information and ranks high for 7.01.
- **Some memorization.** A date ("april 19") ranks in 2.01's top terms. Minor at this data size.

## What the baseline gets wrong

- **1.02 terminations (F1 0.36, recall 0.26).** Termination filings use the same vocabulary as new agreements ("agreement", "entered into"), so bag-of-words mostly predicts 1.01. Telling them apart takes reading what happened to the agreement.
- **`other` (F1 0.69).** A mix of 11 unrelated rare events with 16 test examples; recall suffers most.
- **8.01 other events (F1 0.77).** A catch-all by definition: dividends, buybacks, litigation, anything without its own item.
- **Classes with under 30 test examples** (1.02, 3.01, 3.03, 4.01, `other`) have noisy F1; confidence intervals come in M2.

**Hypothesis for M2:** an LLM's gains, if any, show up mainly on 1.02, 2.01, 8.01 and `other`, the classes where meaning matters more than keywords. On the easy keyword classes (2.02, 5.02, 5.07) the baseline is already at 0.93–0.98, and an LLM costs far more per document.

## Reproduce

```bash
export SEC_USER_AGENT="Your Name your@email.com"   # SEC requires a contact
uv run python src/scrape.py index        # ~2 min
uv run python src/scrape.py download     # ~30 min at ~5 requests/sec
uv run python src/build.py               # ~10 s
uv run python src/baseline.py            # ~10 s
```

Outputs: `data/dataset.parquet`, `data/labels.json`, `results/metrics_tfidf.json` (settings, threshold sweep, per-class metrics, top terms), `results/predictions_tfidf.parquet` (per-label probabilities for val and test).

Baseline settings: TF-IDF on 1–2-grams (min_df 3, max_df 0.9, up to 200K features, sublinear tf; 82,592 terms after filtering); one-vs-rest logistic regression (liblinear, C=4, balanced class weights); seed 42.
