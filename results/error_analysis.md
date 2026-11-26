# Error analysis (test split, 1,220 filings)

Models: tfidf, modernbert_1024, haiku55_fewshot, sonnet55_fewshot, gpt62sol_zeroshot. Each at its own val-chosen threshold (LLMs: label present or not).

## 1. Error shape

Over-labelled: all true labels plus extra ones. Under-labelled: some true labels, nothing extra. Swapped: at least one missed and one wrong. True labels per filing: 1.54.

| Model | All right | Over-labelled | Under-labelled | Swapped | Labels predicted per filing | No label predicted |
|---|---|---|---|---|---|---|
| tfidf | 73.0% | 10.6% | 10.9% | 5.5% | 1.53 | 23 |
| modernbert_1024 | 82.3% | 5.2% | 5.9% | 6.6% | 1.50 | 3 |
| haiku55_fewshot | 61.0% | 22.8% | 3.1% | 13.1% | 1.78 | 0 |
| sonnet55_fewshot | 83.7% | 4.8% | 1.0% | 10.6% | 1.60 | 0 |
| gpt62sol_zeroshot | 91.9% | 0.2% | 0.5% | 7.5% | 1.54 | 0 |

## 2. Top confusions

Counted per filing: a true label the model missed, paired with each wrong label it predicted on the same filing. Cells are counts on the test set.

| True → predicted instead | tfidf | modernbert_1024 | haiku55_fewshot | sonnet55_fewshot | gpt62sol_zeroshot |
|---|---|---|---|---|---|
| 8.01 → 7.01 | 9 | 14 | 44 | 53 | 35 |
| 8.01 → 1.01 | 6 | 7 | 8 | 28 | 13 |
| 7.01 → 8.01 | 8 | 17 | 9 | 8 | 6 |
| 8.01 → 2.03 | 4 | 5 | 16 | 15 | 3 |
| 8.01 → 3.02 | 4 | 4 | 16 | 5 | 1 |
| 8.01 → 5.02 | 7 | 6 | 6 | 5 | 4 |
| 8.01 → 3.01 | 3 | 4 | 5 | 8 | 6 |
| 8.01 → 2.01 | 2 | 2 | 7 | 7 | 2 |
| 8.01 → 2.02 | 2 | 0 | 5 | 1 | 4 |
| other → 5.02 | 2 | 2 | 2 | 2 | 2 |

False positives / false negatives per label (where each model errs in which direction):

| Label | Support | tfidf | modernbert_1024 | haiku55_fewshot | sonnet55_fewshot | gpt62sol_zeroshot |
|---|---|---|---|---|---|---|
| 1.01 | 199 | 31 / 6 | 23 / 6 | 20 / 50 | 40 / 1 | 15 / 2 |
| 1.02 | 27 | 5 / 20 | 3 / 13 | 8 / 4 | 8 / 2 | 2 / 0 |
| 2.01 | 35 | 13 / 7 | 5 / 13 | 30 / 0 | 9 / 0 | 3 / 0 |
| 2.02 | 389 | 15 / 19 | 5 / 4 | 10 / 2 | 3 / 6 | 5 / 3 |
| 2.03 | 77 | 22 / 6 | 17 / 9 | 42 / 5 | 23 / 3 | 6 / 3 |
| 3.01 | 23 | 7 / 2 | 8 / 2 | 10 / 0 | 8 / 0 | 6 / 0 |
| 3.02 | 72 | 20 / 6 | 14 / 11 | 43 / 1 | 14 / 2 | 3 / 3 |
| 3.03 | 29 | 7 / 4 | 8 / 10 | 22 / 6 | 8 / 3 | 0 / 2 |
| 4.01 | 17 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| 5.02 | 220 | 8 / 22 | 14 / 17 | 22 / 5 | 8 / 0 | 6 / 0 |
| 5.03 | 55 | 3 / 5 | 3 / 12 | 11 / 14 | 1 / 5 | 1 / 1 |
| 5.07 | 98 | 0 / 3 | 5 / 2 | 3 / 0 | 0 / 1 | 1 / 0 |
| 7.01 | 341 | 33 / 48 | 23 / 53 | 216 / 12 | 76 / 10 | 37 / 11 |
| 8.01 | 285 | 51 / 75 | 43 / 60 | 57 / 106 | 14 / 109 | 8 / 70 |
| other | 16 | 1 / 7 | 1 / 9 | 9 / 13 | 0 / 2 | 1 / 2 |

## 3. Subset accuracy by filing length

Length = ModernBERT tokens from the first [ITEM] marker. modernbert_e5 reads 512 tokens, modernbert_1024 reads 1,024; TF-IDF and the LLMs read the whole filing.

| Length | Filings | tfidf | modernbert_e5 | modernbert_1024 | gpt62sol_zeroshot |
|---|---|---|---|---|---|
| ≤ 512 | 624 | 86.5% | 93.4% | 94.7% | 93.4% |
| 513–1,024 | 319 | 72.1% | 75.2% | 84.0% | 89.3% |
| > 1,024 | 277 | 43.7% | 40.1% | 52.3% | 91.3% |

## 4. Where modernbert_1024 and gpt62sol_zeroshot disagree

|  | gpt62sol_zeroshot right | gpt62sol_zeroshot wrong |
|---|---|---|
| modernbert_1024 right | 955 | 49 |
| modernbert_1024 wrong | 166 | 50 |

Subset accuracy if you could pick the right one each time (oracle): 95.9%, vs 91.9% for gpt62sol_zeroshot alone.

**Every model wrong: 40 filings (3.3%).** These are the first place to look for label noise or filer judgment calls. The first 15, to check by hand:

- [`0000899751-26-000088`](https://www.sec.gov/Archives/edgar/data/899751/000089975126000088/) true ['8.01'] | modernbert_1024: ['7.01']; gpt62sol_zeroshot: ['2.02']
  > As previously disclosed, on September 21, 2026, Titan International, Inc. (the “Company”) entered into a Sale and Purchase Agreement (the “Purchase Agreement”) pursuant to which the Company agreed to sell its ITM business. Beginning with the Company’s Quarterly Report on Form 10-Q for the quarter en
- [`0000912767-26-000050`](https://www.sec.gov/Archives/edgar/data/912767/000091276726000050/) true ['8.01', 'other'] | modernbert_1024: ['5.02', '8.01']; gpt62sol_zeroshot: ['5.02', '8.01']
  > As previously disclosed on a Form 8-K filed on August 27, 2024, Matthew J. Missad, the Executive Chairman of UFP Industries, Inc., a Michigan corporation (the “ Company ”), will be retiring from his employment with the Company on December 31, 2026 (the “ Retirement Date ”). As previously disclosed, 
- [`0000950157-26-000850`](https://www.sec.gov/Archives/edgar/data/916076/000095015726000850/) true ['8.01'] | modernbert_1024: ['7.01']; gpt62sol_zeroshot: ['7.01']
  > As previously announced, on June 27, 2026, Martin Marietta Materials, Inc., a North Carolina corporation (“ Martin Marietta ” or the “ Company ”), and LNA Holding SRL, a société à responsabilité limitée organized under the laws of Belgium, entered into a Securities Sale Agreement, pursuant to which 
- [`0001058090-26-000069`](https://www.sec.gov/Archives/edgar/data/1058090/000105809026000069/) true ['7.01'] | modernbert_1024: ['8.01']; gpt62sol_zeroshot: ['8.01']
  > The health and safety of our guests and employees is our highest priority. We are aware that public health authorities such as the Food and Drug Administration and the Minnesota Department of Health are investigating the source of a salmonella outbreak in the supply chain impacting several food serv
- [`0001104659-26-114127`](https://www.sec.gov/Archives/edgar/data/1547546/000110465926114127/) true ['8.01'] | modernbert_1024: ['3.01', '8.01']; gpt62sol_zeroshot: ['3.01']
  > On October 1, 2026, Lument Finance Trust, Inc. (the “Company”) received notice from the New York Stock Exchange (the “NYSE”) that the Company has regained compliance with the minimum share price requirement in Section 802.01C of the NYSE Listed Company Manual, based on the calculation of the Company
- [`0001104659-26-124574`](https://www.sec.gov/Archives/edgar/data/1819928/000110465926124574/) true ['7.01'] | modernbert_1024: ['1.01']; gpt62sol_zeroshot: ['8.01']
  > Under the terms of the previously announced Agreement and Plan of Merger (the “Merger Agreement”), dated as of August 6, 2026, by and among DoubleVerify Holdings, Inc. (the “Company”), Neptune BidCo US Inc., a Delaware corporation (“Parent”), and Wallace Merger Sub Inc., a Delaware corporation and a
- [`0001140361-26-041436`](https://www.sec.gov/Archives/edgar/data/1746129/000114036126041436/) true ['2.01', '7.01'] | modernbert_1024: ['2.01', '7.01', '8.01']; gpt62sol_zeroshot: ['2.01', '8.01']
  > On November 1, 2026, Bank7 Corp. (the “Company”) , consummated the previously announced merger (the “Merger”) pursuant to the Agreement and Plan of Merger, dated as of September 16, 2026 (the “Merger Agreement”), by and between the Company and Century Financial Services Corporation (“Century”), a Ne
- [`0001140361-26-041901`](https://www.sec.gov/Archives/edgar/data/1533743/000114036126041901/) true ['5.02', '8.01'] | modernbert_1024: ['5.02']; gpt62sol_zeroshot: ['5.02', '7.01']
  > Appointment of Gregory J. Flesher On November 2, 2026, the Board approved the appointment of Gregory J. Flesher as Chief Operating Officer of the Company, effective November 4, 2026 (the “ Start Date ”). In connection with Mr. Flesher’s appointment as Chief Operating Officer, the Company entered int
- [`0001185185-26-002793`](https://www.sec.gov/Archives/edgar/data/802257/000118518526002793/) true ['1.01', '3.02', '5.02', '8.01'] | modernbert_1024: ['1.01', '3.02']; gpt62sol_zeroshot: ['1.01', '3.02', '5.02', '7.01']
  > ● On June 28, 2026, Mitesco, Inc., a Nevada corporation (the “Company”) entered into a Common Stock Purchase Agreement (the “Purchase Agreement”) and a Registration Rights Agreement (the “Registration Rights Agreement”) with an institutional investor (the “Investor”), pursuant to which the Investor 
- [`0001185185-26-003072`](https://www.sec.gov/Archives/edgar/data/802257/000118518526003072/) true ['3.02', '5.02', '8.01'] | modernbert_1024: ['3.02', '8.01']; gpt62sol_zeroshot: ['3.02', '5.02', '7.01']
  > Series X Preferred Stock issuances for compensation On July 21, 2026 the Board of Directors by written consent agreed to make certain additions to its executive and Board compensation. The Company has agreed to issue 2,400 shares of its Series X Preferred stock whose total face value is $60,000 to t
- [`0001185185-26-003266`](https://www.sec.gov/Archives/edgar/data/802257/000118518526003266/) true ['1.01', '3.02', '5.02', '8.01'] | modernbert_1024: ['1.01', '3.02', '8.01']; gpt62sol_zeroshot: ['1.01', '3.02', '5.02', '7.01']
  > The Company has entered into an Advisory Agreement with Dawson James Securities allowing them to assist on certain mergers, financing and corporate structuring. The are to receive an issuance of 1,000,000 shares of restricted common stock as consideration, and other fees in the future if successful 
- [`0001185185-26-005236`](https://www.sec.gov/Archives/edgar/data/802257/000118518526005236/) true ['3.02', '5.02', '5.03', '8.01'] | modernbert_1024: ['3.02', '5.03', '5.07', '8.01']; gpt62sol_zeroshot: ['3.02', '5.02', '5.07', '8.01']
  > On November 5, 2026, the majority of the holders of its Series X Preferred shares, with over 54% of votes available, approved the amendment of its Certificate of Designation to increase the authorized shares from five hundred million (500,000,000) to one billion (1,000,000,000). The Board of Directo
- [`0001193125-26-299839`](https://www.sec.gov/Archives/edgar/data/1785056/000119312526299839/) true ['1.01', '8.01'] | modernbert_1024: ['1.01', '2.03', '3.02', '7.01']; gpt62sol_zeroshot: ['1.01', '7.01']
  > Stock Purchase Agreement On July 7, 2026, Interactive Strength Inc., a Delaware corporation (the “Company”), entered into a Stock Purchase Agreement (the “Purchase Agreement”) by and among the Company, STEPR, Inc., a Delaware corporation (“STEPR”), STEPR PTY LTD – ACN 660 939 079, an Australian prop
- [`0001193125-26-333977`](https://www.sec.gov/Archives/edgar/data/1527469/000119312526333977/) true ['2.02', '8.01'] | modernbert_1024: ['2.02', '7.01']; gpt62sol_zeroshot: ['2.02', '7.01']
  > The information disclosed under [ITEM] of this Current Report on Form 8-K is incorporated by reference into this [ITEM] . The information disclosed under this [ITEM] is being “furnished” and shall not be deemed “filed” by Athene Holding Ltd. (the “Company”) for purposes of Section 18 of the Securiti
- [`0001193125-26-340236`](https://www.sec.gov/Archives/edgar/data/1527469/000119312526340236/) true ['2.03', '8.01'] | modernbert_1024: ['1.01', '8.01']; gpt62sol_zeroshot: ['1.01', '2.03']
  > The information required by [ITEM] contained in [ITEM] below is incorporated by reference herein. [ITEM] On August 5, 2026, Athene Holding Ltd. (the “Company”) entered into an Underwriting Agreement (the “Underwriting Agreement”) by and among the Company and Wells Fargo Securities, LLC, Barclays Cap

True labels among the every-model-wrong filings: 8.01 30, 7.01 11, 5.02 10, 1.01 5, 3.02 5, other 3, 2.03 3, 5.07 2, 2.01 1, 5.03 1, 2.02 1, 4.01 1.

## 5. Hybrid: fine-tuned model on short filings, LLM on the rest

Filings of at most N ModernBERT tokens go to modernbert_1024; longer ones to gpt62sol_zeroshot. N chosen on val by subset accuracy.

| Split | Filings | Routed to fine-tune | Hybrid micro / macro | Hybrid subset | gpt62sol_zeroshot micro / macro | gpt62sol_zeroshot subset |
|---|---|---|---|---|---|---|
| val, cut 512 (chosen) | 200 | 64% | 0.958 / 0.879 | 0.925 | 0.941 / 0.915 | 0.905 |
| val, cut 1024 | 200 | 86% | 0.954 / 0.867 | 0.910 | 0.941 / 0.915 | 0.905 |
| **test, cut 512** | 1220 | 51% | 0.957 / 0.953 | 0.925 | 0.949 / 0.951 | 0.919 |

API cost on test: $2.76 per 1K filings, vs $4.58 sending every filing to gpt62sol_zeroshot (40% less; long filings cost more tokens, so the saving is below the routed share).
