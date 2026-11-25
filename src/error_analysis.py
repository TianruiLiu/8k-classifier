"""M4: error analysis across models on the test split, as numbers to write up from.

    uv run python src/error_analysis.py

Sections in results/error_analysis.md:
  1. Error shape: over-labelling vs under-labelling vs swaps, labels predicted per filing.
  2. Top confusions: (true label missed, wrong label predicted instead) pairs on the same filing.
  3. Accuracy by filing length in ModernBERT tokens (shows what truncation costs).
  4. Where models disagree: ModernBERT vs the best LLM, and filings every model gets wrong
     (candidates for label noise / filer judgment calls; listed with EDGAR links to check by hand).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import load_model, matrices, snippet  # noqa: E402

DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results", ROOT / "models"
MAIN = ["tfidf", "modernbert_1024", "haiku55_fewshot", "sonnet55_fewshot", "gpt62sol_zeroshot"]
LENGTH_MODELS = ["tfidf", "modernbert_e5", "modernbert_1024", "gpt62sol_zeroshot"]
FINETUNED, BEST_LLM = "modernbert_1024", "gpt62sol_zeroshot"


def edgar_url(acc: str, cik: int) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def main() -> None:
    labels = json.loads((DATA / "labels.json").read_text())["labels"]
    data = pd.read_parquet(DATA / "dataset.parquet")
    test = data[data["split"] == "test"].set_index("accession")
    names = list(dict.fromkeys(MAIN + LENGTH_MODELS))
    preds = {}
    for n in names:
        df = load_model(n, labels, "test")
        assert len(df) == len(test), f"{n}: {len(df)} test rows"
        Yt, Yp = matrices(df, labels)
        preds[n] = (df["accession"].tolist(), Yt, Yp)
    acc = preds[MAIN[0]][0]
    assert all(p[0] == acc for p in preds.values()), "models scored different filings"
    Yt = preds[MAIN[0]][1]
    L = np.array(labels)
    out = ["# Error analysis (test split, 1,220 filings)", "",
           f"Models: {', '.join(MAIN)}. Each at its own val-chosen threshold (LLMs: label present or not).", ""]

    # 1. Error shape
    rows = []
    for n in MAIN:
        _, _, Yp = preds[n]
        exact = (Yp == Yt).all(1)
        sup = (Yp >= Yt).all(1) & ~exact          # predicted every true label plus extras
        sub = (Yp <= Yt).all(1) & ~exact          # predicted a strict subset (missed some, nothing extra)
        swap = ~exact & ~sup & ~sub
        empty = Yp.sum(1) == 0
        rows.append([n, f"{exact.mean():.1%}", f"{sup.mean():.1%}", f"{sub.mean():.1%}", f"{swap.mean():.1%}",
                     f"{Yp.sum(1).mean():.2f}", int(empty.sum())])
    out += ["## 1. Error shape", "",
            "Over-labelled: all true labels plus extra ones. Under-labelled: some true labels, nothing extra. "
            f"Swapped: at least one missed and one wrong. True labels per filing: {Yt.sum(1).mean():.2f}.", ""]
    out += table(["Model", "All right", "Over-labelled", "Under-labelled", "Swapped", "Labels predicted per filing", "No label predicted"], rows)

    # 2. Top confusions
    out += ["", "## 2. Top confusions", "",
            "Counted per filing: a true label the model missed, paired with each wrong label it predicted on the same filing. "
            "Cells are counts on the test set.", ""]
    pair_counts = {}
    for n in MAIN:
        _, _, Yp = preds[n]
        c = Counter()
        for i in range(len(Yt)):
            missed, wrong = L[Yt[i] & ~Yp[i]], L[Yp[i] & ~Yt[i]]
            for m in missed:
                for w in wrong:
                    c[(m, w)] += 1
        pair_counts[n] = c
    top = Counter()
    for c in pair_counts.values():
        top.update(c)
    pairs = [p for p, _ in top.most_common(10)]
    out += table(["True → predicted instead"] + MAIN, [[f"{m} → {w}"] + [pair_counts[n][(m, w)] for n in MAIN] for m, w in pairs])

    # Per-label false positives vs false negatives (direction of each model's errors)
    out += ["", "False positives / false negatives per label (where each model errs in which direction):", ""]
    rows = []
    for j, l in enumerate(labels):
        rows.append([l, int(Yt[:, j].sum())] + [f"{int((preds[n][2][:, j] & ~Yt[:, j]).sum())} / {int((~preds[n][2][:, j] & Yt[:, j]).sum())}" for n in MAIN])
    out += table(["Label", "Support"] + MAIN, rows)

    # 3. Accuracy by length
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(MODELS / FINETUNED)
    bodies = [t[t.find("[ITEM]"):] if "[ITEM]" in t else t for t in test.loc[acc, "text"]]
    ntok = np.array([len(x) for x in tok(bodies, add_special_tokens=True)["input_ids"]])
    buckets = [("≤ 512", ntok <= 512), ("513–1,024", (ntok > 512) & (ntok <= 1024)), ("> 1,024", ntok > 1024)]
    rows = []
    for label, mask in buckets:
        r = [label, int(mask.sum())]
        for n in LENGTH_MODELS:
            Yp = preds[n][2]
            r.append(f"{(Yp[mask] == Yt[mask]).all(1).mean():.1%}")
        rows.append(r)
    out += ["", "## 3. Subset accuracy by filing length", "",
            "Length = ModernBERT tokens from the first [ITEM] marker. modernbert_e5 reads 512 tokens, modernbert_1024 reads 1,024; "
            "TF-IDF and the LLMs read the whole filing.", ""]
    out += table(["Length", "Filings"] + LENGTH_MODELS, rows)

    # 4. Disagreement
    ft = (preds[FINETUNED][2] == Yt).all(1)
    llm = (preds[BEST_LLM][2] == Yt).all(1)
    all_wrong = np.all([~(preds[n][2] == Yt).all(1) for n in MAIN], axis=0)
    out += ["", f"## 4. Where {FINETUNED} and {BEST_LLM} disagree", ""]
    out += table(["", f"{BEST_LLM} right", f"{BEST_LLM} wrong"],
                 [[f"{FINETUNED} right", int((ft & llm).sum()), int((ft & ~llm).sum())],
                  [f"{FINETUNED} wrong", int((~ft & llm).sum()), int((~ft & ~llm).sum())]])
    out += ["", f"Subset accuracy if you could pick the right one each time (oracle): {(ft | llm).mean():.1%}, "
            f"vs {llm.mean():.1%} for {BEST_LLM} alone.", "",
            f"**Every model wrong: {int(all_wrong.sum())} filings ({all_wrong.mean():.1%}).** "
            "These are the first place to look for label noise or filer judgment calls. The first 15, to check by hand:", ""]
    true_sets = [[str(x) for x in L[Yt[i]]] for i in range(len(Yt))]
    for i in np.where(all_wrong)[0][:15]:
        a = acc[i]
        guesses = "; ".join(f"{n}: {[str(x) for x in L[preds[n][2][i]]]}" for n in (FINETUNED, BEST_LLM))
        out += [f"- [`{a}`]({edgar_url(a, test.loc[a, 'cik'])}) true {true_sets[i]} | {guesses}",
                f"  > {snippet(test.loc[a, 'text'])[:300]}"]
    labels_all_wrong = Counter(l for i in np.where(all_wrong)[0] for l in true_sets[i])
    out += ["", "True labels among the every-model-wrong filings: " +
            ", ".join(f"{l} {c}" for l, c in labels_all_wrong.most_common()) + "."]

    (RESULTS / "error_analysis.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:80]))
    print(f"\n... wrote results/error_analysis.md")


if __name__ == "__main__":
    main()
