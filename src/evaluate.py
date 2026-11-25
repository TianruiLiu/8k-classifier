"""Shared evaluation harness: every model is scored the same way.

    uv run python src/evaluate.py                    # all models in results/, test split
    uv run python src/evaluate.py --split val        # validation (use while tuning prompts)
    uv run python src/evaluate.py --models tfidf llm_zeroshot

Input, one file per model: results/predictions_<model>.parquet with columns
  accession, split, labels (true label list), p_<label> for every label.
  Scores are probabilities (TF-IDF) or 1.0/0.0 (an LLM). The decision threshold comes from
  results/metrics_<model>.json ("threshold") if present, else 0.5.

A model may cover only part of a split (e.g. an LLM run on 200 val filings); it is scored
on the filings it has, and paired comparisons use only filings both models have.

Outputs:
  results/compare_<split>.md   summary + per-class F1, all models, with 95% bootstrap CIs
  results/eval_<split>.json    the same numbers, machine-readable
  results/errors_<model>.md    false positives / negatives per class with text snippets (test only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BASELINE = "tfidf"
SEED = 42
SNIPPET_CHARS = 400
ERRORS_PER_TYPE = 8


# ---------- metrics on 0/1 matrices (rows = filings, columns = labels) ----------

def counts(Yt: np.ndarray, Yp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = (Yt & Yp).sum(axis=0)
    fp = (~Yt & Yp).sum(axis=0)
    fn = (Yt & ~Yp).sum(axis=0)
    return tp, fp, fn


def f1(tp, fp, fn) -> np.ndarray:
    # 2tp / (2tp + fp + fn); 0 when a class has no true and no predicted filings (sklearn's zero_division=0)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def metrics(Yt: np.ndarray, Yp: np.ndarray) -> dict:
    tp, fp, fn = counts(Yt, Yp)
    per_class = f1(tp, fp, fn)
    return {
        "micro_f1": float(f1(tp.sum(), fp.sum(), fn.sum())),
        "macro_f1": float(per_class.mean()),
        "subset_accuracy": float((Yt == Yp).all(axis=1).mean()),
        "per_class_f1": per_class,
    }


def bootstrap(Yt, Yp, n_boot, rng, Yp_base=None) -> dict:
    """95% CIs by resampling filings. With Yp_base, also CIs of (model - baseline) on the same resamples."""
    n = len(Yt)
    keys = ("micro_f1", "macro_f1", "subset_accuracy")
    samples = {k: [] for k in keys}
    per_class = []
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics(Yt[idx], Yp[idx])
        for k in keys:
            samples[k].append(m[k])
        per_class.append(m["per_class_f1"])
        if Yp_base is not None:
            b = metrics(Yt[idx], Yp_base[idx])
            for k in keys:
                diffs[k].append(m[k] - b[k])
    ci = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    out = {k: ci(samples[k]) for k in keys}
    pc = np.array(per_class)
    out["per_class_f1"] = [[float(np.percentile(pc[:, j], 2.5)), float(np.percentile(pc[:, j], 97.5))] for j in range(pc.shape[1])]
    if Yp_base is not None:
        out["diff_vs_baseline"] = {k: ci(diffs[k]) for k in keys}
    return out


# ---------- loading ----------

def load_model(name: str, labels: list[str], split: str) -> pd.DataFrame | None:
    path = RESULTS / f"predictions_{name}.parquet"
    df = pd.read_parquet(path)
    df = df[df["split"] == split].sort_values("accession").reset_index(drop=True)
    if df.empty:
        return None
    missing = [l for l in labels if f"p_{l}" not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} lacks score columns for: {', '.join(missing)}")
    meta = RESULTS / f"metrics_{name}.json"
    threshold = json.loads(meta.read_text()).get("threshold", 0.5) if meta.exists() else 0.5
    df.attrs["threshold"] = float(threshold)
    return df


def matrices(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Yt = np.array([[l in set(ls) for l in labels] for ls in df["labels"]], dtype=bool)
    Yp = df[[f"p_{l}" for l in labels]].to_numpy() >= df.attrs["threshold"]
    return Yt, Yp


# ---------- error dump ----------

def snippet(text: str) -> str:
    # Skip the cover page (identical boilerplate on every 8-K): start at the first masked heading.
    i = text.find("[ITEM]")
    body = text[i + len("[ITEM]"):] if i >= 0 else text
    s = body.strip()[:SNIPPET_CHARS]
    return s.replace("|", "/") + ("…" if len(body) > SNIPPET_CHARS else "")


def write_errors(name: str, df: pd.DataFrame, labels: list[str], Yt, Yp, texts: dict[str, str]) -> None:
    lines = [f"# Errors: {name} (test)\n",
             f"Up to {ERRORS_PER_TYPE} false negatives (missed) and false positives (wrongly predicted) per label. "
             "Snippets start after the cover page.\n"]
    for j, l in enumerate(labels):
        fn_idx = np.where(Yt[:, j] & ~Yp[:, j])[0]
        fp_idx = np.where(~Yt[:, j] & Yp[:, j])[0]
        lines.append(f"\n## {l}: {len(fn_idx)} missed, {len(fp_idx)} wrongly predicted\n")
        for kind, idx in (("Missed", fn_idx), ("Wrongly predicted", fp_idx)):
            for i in idx[:ERRORS_PER_TYPE]:
                row = df.iloc[i]
                pred = [labels[k] for k in np.where(Yp[i])[0]]
                lines.append(f"- **{kind}** `{row['accession']}`: true {list(row['labels'])}, predicted {pred}\n"
                             f"  > {snippet(texts.get(row['accession'], ''))}\n")
    (RESULTS / f"errors_{name}.md").write_text("".join(lines))


# ---------- main ----------

def fmt(x: float, ci: list[float]) -> str:
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--models", nargs="*", help="default: every results/predictions_*.parquet")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    labels = json.loads((DATA / "labels.json").read_text())["labels"]
    names = args.models or sorted(f.stem.removeprefix("predictions_") for f in RESULTS.glob("predictions_*.parquet"))
    names = [BASELINE] + [n for n in names if n != BASELINE] if BASELINE in names else names
    loaded = {n: df for n in names if (df := load_model(n, labels, args.split)) is not None}
    if not loaded:
        raise SystemExit(f"No predictions for split '{args.split}'.")

    texts = {}
    if args.split == "test":
        ds = pd.read_parquet(DATA / "dataset.parquet", columns=["accession", "text"])
        texts = dict(zip(ds["accession"], ds["text"]))

    report = {}
    for name, df in loaded.items():
        Yt, Yp = matrices(df, labels)
        m = metrics(Yt, Yp)
        # Sanity check: our F1 must equal sklearn's.
        assert abs(m["micro_f1"] - f1_score(Yt, Yp, average="micro", zero_division=0)) < 1e-9
        assert abs(m["macro_f1"] - f1_score(Yt, Yp, average="macro", zero_division=0)) < 1e-9

        base_aligned = None
        if name != BASELINE and BASELINE in loaded:
            base = loaded[BASELINE]
            common = df["accession"].isin(base["accession"])
            df_c = df[common].reset_index(drop=True)
            df_c.attrs = df.attrs
            base_c = base[base["accession"].isin(df_c["accession"])].reset_index(drop=True)
            base_c.attrs = base.attrs
            assert (df_c["accession"].to_numpy() == base_c["accession"].to_numpy()).all()
            Yt_c, Yp_c = matrices(df_c, labels)
            _, Yp_b = matrices(base_c, labels)
            base_aligned = (Yt_c, Yp_c, Yp_b, metrics(Yt_c, Yp_b))

        rng = np.random.default_rng(SEED)
        cis = bootstrap(Yt, Yp, args.n_boot, rng)
        entry = {"n": len(df), "threshold": df.attrs["threshold"],
                 **{k: m[k] for k in ("micro_f1", "macro_f1", "subset_accuracy")},
                 "ci": {k: cis[k] for k in ("micro_f1", "macro_f1", "subset_accuracy")},
                 "per_class": {l: {"f1": float(m["per_class_f1"][j]), "ci": cis["per_class_f1"][j],
                                   "support": int(Yt[:, j].sum())} for j, l in enumerate(labels)}}
        if base_aligned is not None:
            Yt_c, Yp_c, Yp_b, mb = base_aligned
            paired = bootstrap(Yt_c, Yp_c, args.n_boot, np.random.default_rng(SEED), Yp_base=Yp_b)
            mc = metrics(Yt_c, Yp_c)
            entry["vs_baseline"] = {"n_common": len(Yt_c), **{
                k: {"diff": mc[k] - mb[k], "ci": paired["diff_vs_baseline"][k],
                    "significant": not (paired["diff_vs_baseline"][k][0] <= 0 <= paired["diff_vs_baseline"][k][1])}
                for k in ("micro_f1", "macro_f1", "subset_accuracy")}}
        report[name] = entry
        if args.split == "test":
            write_errors(name, df, labels, Yt, Yp, texts)

    # ---------- write the comparison ----------
    lines = [f"# Model comparison ({args.split})\n",
             f"95% bootstrap CIs over filings ({args.n_boot} resamples, seed {SEED}). "
             f"'vs {BASELINE}' is the paired difference on filings both models scored; * = CI excludes 0.\n\n",
             "| Model | N | Micro-F1 | Macro-F1 | Subset acc | Δ micro vs baseline | Δ macro vs baseline |\n",
             "|---|---|---|---|---|---|---|\n"]
    for name, e in report.items():
        d = e.get("vs_baseline")
        dm = (f"{d['micro_f1']['diff']:+.3f} [{d['micro_f1']['ci'][0]:+.3f}, {d['micro_f1']['ci'][1]:+.3f}]"
              f"{'*' if d['micro_f1']['significant'] else ''}") if d else "—"
        dM = (f"{d['macro_f1']['diff']:+.3f} [{d['macro_f1']['ci'][0]:+.3f}, {d['macro_f1']['ci'][1]:+.3f}]"
              f"{'*' if d['macro_f1']['significant'] else ''}") if d else "—"
        lines.append(f"| {name} | {e['n']:,} | {fmt(e['micro_f1'], e['ci']['micro_f1'])} | "
                     f"{fmt(e['macro_f1'], e['ci']['macro_f1'])} | {fmt(e['subset_accuracy'], e['ci']['subset_accuracy'])} | {dm} | {dM} |\n")
    lines += ["\n## Per-class F1 [95% CI]\n\n",
              "| Label | Support | " + " | ".join(report) + " |\n",
              "|---|---|" + "---|" * len(report) + "\n"]
    first = next(iter(report.values()))
    for l in labels:
        # A model scored on fewer filings (e.g. 200 val) has its own support; show it when it differs.
        cells = [f"{e['per_class'][l]['f1']:.2f} [{e['per_class'][l]['ci'][0]:.2f}, {e['per_class'][l]['ci'][1]:.2f}]"
                 + (f" (n={e['per_class'][l]['support']})" if e["n"] != first["n"] else "") for e in report.values()]
        lines.append(f"| {l} | {first['per_class'][l]['support']} | " + " | ".join(cells) + " |\n")
    (RESULTS / f"compare_{args.split}.md").write_text("".join(lines))

    def _json(o):
        return float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else str(o)
    (RESULTS / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2, default=_json))

    print("".join(lines[:4 + len(report)]))
    written = [f"compare_{args.split}.md", f"eval_{args.split}.json"] + ([f"errors_{n}.md" for n in report] if args.split == "test" else [])
    print("Wrote " + ", ".join(f"results/{w}" for w in written))


if __name__ == "__main__":
    main()
