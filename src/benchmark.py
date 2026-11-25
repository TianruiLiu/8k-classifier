"""M4: cost and latency for every model on the test split, in one table.

    uv run python src/benchmark.py                 # LLMs from logs + local models measured now (~10 min)
    uv run python src/benchmark.py --skip-local    # LLM numbers only (seconds)

What is measured, and how:
  LLMs     Read from data/llm_cache (one JSON per API call, written by src/llm.py). Latency is the
           wall time of one request as the client saw it, including any SDK retries after a 429,
           with 3-8 requests in flight. Cost is tokens x list price, with and without prompt caching.
  TF-IDF   Refit on train (same settings as src/baseline.py, timed), then one filing at a time on CPU.
  ModernBERT  Load the saved checkpoint, then one filing at a time (tokenize + forward pass) on the
           GPU (MPS) for every test filing, and on CPU for a seeded sample. Batch-1 latency is what a
           request-at-a-time service would see; batched throughput is in metrics_<name>.json.

Outputs: results/benchmark.json, results/benchmark.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

LLM_RUNS = ["gpt62sol_zeroshot", "gpt62sol_fewshot", "sonnet55_zeroshot", "sonnet55_fewshot",
            "gpt6luna_zeroshot", "gpt6luna_fewshot", "haiku55_zeroshot", "haiku55_fewshot"]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def lat_stats(seconds: list[float]) -> dict:
    ms = [s * 1000 for s in seconds]
    return {"n": len(ms), "p50_ms": round(statistics.median(ms), 1), "p95_ms": round(pct(ms, 0.95), 1),
            "p99_ms": round(pct(ms, 0.99), 1), "mean_ms": round(statistics.fmean(ms), 1)}


def llm_rows() -> list[dict]:
    rows = []
    for name in LLM_RUNS:
        mpath = RESULTS / f"metrics_{name}.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        recs = [json.loads(f.read_text()) for f in (DATA / "llm_cache" / name / m["prompt_hash"]).glob("*.json")]
        recs = [r for r in recs if r.get("split") == "test" and r.get("error") is None]   # skips the saved prompt file
        price = m["by_split"]["test"].get("prices_per_1m")
        n = len(recs)
        tok = {k: sum(r["usage"][k] for r in recs) / n for k in ("input", "output", "cache_read", "cache_write")}
        row = {"model": name, "kind": "llm", "model_id": m["model_id"], **lat_stats([r["latency_s"] for r in recs]),
               "tokens_per_filing": {k: round(v, 1) for k, v in tok.items()}}
        if price:
            pin, pout, pcache = price["input"], price["output"], price["cache_read"]
            with_cache = (tok["input"] * pin + tok["output"] * pout + tok["cache_read"] * pcache
                          + tok["cache_write"] * pin * 1.25) / 1e6 * 1000
            no_cache = ((tok["input"] + tok["cache_read"] + tok["cache_write"]) * pin + tok["output"] * pout) / 1e6 * 1000
            row.update(cost_per_1k_usd=round(with_cache, 2), cost_per_1k_no_cache_usd=round(no_cache, 2),
                       prices_per_1m=price)
        rows.append(row)
    return rows


def tfidf_row(df: pd.DataFrame, labels: list[str]) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from baseline import CLASSIFIER, VECTORIZER
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    Y = MultiLabelBinarizer(classes=labels).fit_transform(train["labels"])
    t0 = time.perf_counter()
    vec = TfidfVectorizer(**VECTORIZER)
    clf = OneVsRestClassifier(LogisticRegression(**CLASSIFIER)).fit(vec.fit_transform(train["text"]), Y)
    fit_s = time.perf_counter() - t0
    texts = test["text"].tolist()
    for t in texts[:20]:                                         # warm-up
        clf.predict_proba(vec.transform([t]))
    times = []
    for t in texts:
        t0 = time.perf_counter()
        clf.predict_proba(vec.transform([t]))
        times.append(time.perf_counter() - t0)
    return {"model": "tfidf", "kind": "local", "device": "cpu", **lat_stats(times), "train_seconds": round(fit_s, 1),
            "cost_per_1k_usd": 0.0}


def modernbert_rows(df: pd.DataFrame, name: str, cpu_sample: int) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    m = json.loads((RESULTS / f"metrics_{name}.json").read_text())
    max_len = m["config"]["max_len"]
    tok = AutoTokenizer.from_pretrained(MODELS / name)
    texts = [t[t.find("[ITEM]"):] if "[ITEM]" in t else t for t in df.loc[df["split"] == "test", "text"]]
    rows = []
    devices = [("mps", texts)] if torch.backends.mps.is_available() else []
    devices.append(("cpu", random.Random(42).sample(texts, min(cpu_sample, len(texts)))))
    for dev, sample in devices:
        device = torch.device(dev)
        model = AutoModelForSequenceClassification.from_pretrained(MODELS / name).to(device).eval()

        def one(t: str) -> None:
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                torch.sigmoid(model(**enc).logits).cpu()          # .cpu() forces the GPU to finish

        for t in sample[:10]:
            one(t)
        times = []
        for t in sample:
            t0 = time.perf_counter()
            one(t)
            times.append(time.perf_counter() - t0)
        rows.append({"model": name, "kind": "local", "device": dev, **lat_stats(times),
                     "train_seconds": round(m["train_minutes"] * 60), "batched_ms_per_filing_mps": m["test_inference_ms_per_doc"],
                     "cost_per_1k_usd": 0.0, "torch_threads": torch.get_num_threads() if dev == "cpu" else None})
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    return rows


def fmt_ms(x: float) -> str:
    return f"{x/1000:.2f} s" if x >= 1000 else f"{x:.0f} ms" if x >= 10 else f"{x:.1f} ms"


def to_markdown(rows: list[dict], env: dict) -> str:
    out = ["# Cost and latency (test split)", "",
           f"Measured {env['date']} on {env['machine']}. Local models: one filing per call, after warm-up. "
           "LLMs: one API request per filing as the client saw it (includes network and any 429 retries).", "",
           "| Model | Where | p50 | p95 | p99 | $ per 1K filings | $ per 1K, no prompt caching | Tokens per filing (in / cached / out) | Filings timed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        where = r.get("device", "API").upper() if r["kind"] == "local" else "API"
        t = r.get("tokens_per_filing")
        toks = f"{t['input']:,.0f} / {t['cache_read']:,.0f} / {t['output']:.0f}" if t else "—"
        cost = f"{r['cost_per_1k_usd']:.2f}" if "cost_per_1k_usd" in r else "—"
        nc = f"{r['cost_per_1k_no_cache_usd']:.2f}" if "cost_per_1k_no_cache_usd" in r else "—"
        out.append(f"| {r['model']} | {where} | {fmt_ms(r['p50_ms'])} | {fmt_ms(r['p95_ms'])} | {fmt_ms(r['p99_ms'])} | "
                   f"{cost} | {nc} | {toks} | {r['n']:,} |")
    local = [r for r in rows if r["kind"] == "local" and "train_seconds" in r]
    if local:
        out += ["", "Training time (one-off): " + "; ".join(sorted({f"{r['model']} {r['train_seconds']/60:.1f} min" if r['train_seconds'] >= 60
                                                                  else f"{r['model']} {r['train_seconds']:.0f} s" for r in local})) + "."]
    out += ["", "Local models have no per-filing charge; the machine is the cost. LLM prices are list prices as logged with each run."]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--finetuned", default="modernbert_1024")
    p.add_argument("--cpu-sample", type=int, default=200, help="test filings timed on CPU for the fine-tuned model")
    args = p.parse_args()

    rows = llm_rows()
    if not args.skip_local:
        df = pd.read_parquet(DATA / "dataset.parquet")
        labels = json.loads((DATA / "labels.json").read_text())["labels"]
        print("timing TF-IDF ..."); rows.append(tfidf_row(df, labels))
        print(f"timing {args.finetuned} ..."); rows += modernbert_rows(df, args.finetuned, args.cpu_sample)
    rows.sort(key=lambda r: r["p50_ms"])

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    env = {"date": time.strftime("%Y-%m-%d"), "machine": f"{chip}, {os.cpu_count()} cores, {platform.system()} {platform.release()}"}
    (RESULTS / "benchmark.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    md = to_markdown(rows, env)
    (RESULTS / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
