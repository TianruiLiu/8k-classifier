"""TF-IDF + one-vs-rest logistic regression baseline for multi-label 8-K classification.

    uv run python src/baseline.py

Protocol (the same one every later model in this project will follow):
  - fit on train only
  - pick one global decision threshold on val (sweep 0.20-0.80, maximize micro-F1)
  - score test once, with that threshold

Outputs:
  results/metrics_tfidf.json       metrics, threshold, settings, data sizes, git commit
  results/predictions_tfidf.parquet  accession, split, true labels, per-label probabilities (val + test)
  results/table_tfidf.md           results table to paste into RESULTS.md
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42

VECTORIZER = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=200_000, sublinear_tf=True)
CLASSIFIER = dict(solver="liblinear", C=4.0, class_weight="balanced", max_iter=2000, random_state=SEED)
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except subprocess.CalledProcessError:
        return "no-commit-yet"


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),   # every label right
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA / "dataset.parquet")
    meta = json.loads((DATA / "labels.json").read_text())
    labels = meta["labels"]

    mlb = MultiLabelBinarizer(classes=labels)
    Y = mlb.fit_transform(df["labels"])
    parts = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}
    print("sizes:", {s: int(m.sum()) for s, m in parts.items()}, f"| {len(labels)} labels")

    t0 = time.perf_counter()
    vec = TfidfVectorizer(**VECTORIZER)
    X_train = vec.fit_transform(df.loc[parts["train"], "text"])
    X_val = vec.transform(df.loc[parts["val"], "text"])
    X_test = vec.transform(df.loc[parts["test"], "text"])
    # Sequential on purpose: with n_jobs>1, joblib hands liblinear read-only memmapped
    # arrays and it fails ("WRITEBACKIFCOPY base is read-only"). ~15 labels fit in seconds anyway.
    clf = OneVsRestClassifier(LogisticRegression(**CLASSIFIER))
    clf.fit(X_train, Y[parts["train"]])
    fit_s = time.perf_counter() - t0
    print(f"vocab {len(vec.vocabulary_):,} terms; fit in {fit_s:.1f}s")

    # Threshold: chosen on val only.
    P_val = clf.predict_proba(X_val)
    sweep = {float(t): f1_score(Y[parts["val"]], (P_val >= t).astype(int), average="micro", zero_division=0) for t in THRESHOLDS}
    threshold = max(sweep, key=sweep.get)
    print(f"val micro-F1 by threshold: " + ", ".join(f"{t:.2f}={f:.3f}" for t, f in sweep.items()))
    print(f"chosen threshold: {threshold:.2f}")

    # Test: scored once.
    t1 = time.perf_counter()
    P_test = clf.predict_proba(X_test)
    pred_ms_per_doc = (time.perf_counter() - t1) / max(1, X_test.shape[0]) * 1000
    Y_test, Yhat_test = Y[parts["test"]], (P_test >= threshold).astype(int)
    val_scores = scores(Y[parts["val"]], (P_val >= threshold).astype(int))
    test_scores = scores(Y_test, Yhat_test)

    prec, rec, f1, support = precision_recall_fscore_support(Y_test, Yhat_test, zero_division=0)
    per_class = {l: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                 for l, p, r, f, s in zip(labels, prec, rec, f1, support)}

    # Top features per class: the quickest leak detector. If a class's top terms are
    # heading words ("item", a title phrase), the masking in build.py missed something.
    terms = np.array(vec.get_feature_names_out())
    top_terms = {l: terms[np.argsort(est.coef_[0])[-12:][::-1]].tolist() for l, est in zip(labels, clf.estimators_)}

    print(f"\nTEST  micro-F1 {test_scores['micro_f1']:.3f} | macro-F1 {test_scores['macro_f1']:.3f} | "
          f"subset acc {test_scores['subset_accuracy']:.3f}   (val micro-F1 {val_scores['micro_f1']:.3f})")
    print(f"\n{'label':7}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>6}   top terms")
    for l in labels:
        c = per_class[l]
        print(f"{l:7}{c['precision']:7.2f}{c['recall']:7.2f}{c['f1']:7.2f}{c['support']:6}   {', '.join(top_terms[l][:6])}")

    # Save everything a later comparison needs.
    metrics = {
        "model": "tfidf_logreg", "git_commit": git_commit(), "seed": SEED,
        "vectorizer": VECTORIZER, "classifier": {**CLASSIFIER, "wrapper": "OneVsRest"},
        "threshold": threshold, "threshold_rule": "global, max val micro-F1 over 0.20-0.80 step 0.05",
        "val_threshold_sweep": sweep,
        "n": {s: int(m.sum()) for s, m in parts.items()}, "labels": labels,
        "val": val_scores, "test": test_scores, "test_per_class": per_class,
        "fit_seconds": round(fit_s, 2), "predict_ms_per_doc": round(pred_ms_per_doc, 3),
        "top_terms": top_terms, "dataset": {k: meta[k] for k in ("split_rule", "rare_codes", "rare_handling", "min_count")},
    }
    (RESULTS / "metrics_tfidf.json").write_text(json.dumps(metrics, indent=2))

    rows = []
    for s, P in (("val", P_val), ("test", P_test)):
        sub = df.loc[parts[s], ["accession", "filing_date", "labels"]].reset_index(drop=True)
        sub.insert(1, "split", s)
        probs = pd.DataFrame(P, columns=[f"p_{l}" for l in labels])
        rows.append(pd.concat([sub, probs], axis=1))
    pd.concat(rows).to_parquet(RESULTS / "predictions_tfidf.parquet", index=False)

    table = [
        "| Model | Micro-F1 | Macro-F1 | Subset acc | Notes |",
        "|---|---|---|---|---|",
        f"| TF-IDF + LR (1–2-grams) | {test_scores['micro_f1']:.3f} | {test_scores['macro_f1']:.3f} | "
        f"{test_scores['subset_accuracy']:.3f} | threshold {threshold:.2f} from val; "
        f"{metrics['n']['train']:,} train / {metrics['n']['test']:,} test; {pred_ms_per_doc:.2f} ms/doc on CPU |",
        "",
        "| Label | Precision | Recall | F1 | Test support |",
        "|---|---|---|---|---|",
        *[f"| {l} | {per_class[l]['precision']:.2f} | {per_class[l]['recall']:.2f} | {per_class[l]['f1']:.2f} | {per_class[l]['support']} |" for l in labels],
    ]
    (RESULTS / "table_tfidf.md").write_text("\n".join(table) + "\n")
    print("\nWrote results/metrics_tfidf.json, results/predictions_tfidf.parquet, results/table_tfidf.md")


if __name__ == "__main__":
    main()
