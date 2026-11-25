"""Fine-tune ModernBERT-base for multi-label 8-K classification (runs locally, no internet needed).

    # smoke test (~2 min): a tiny slice, 1 epoch, separate results name
    uv run python src/finetune.py --smoke
    # full run (~1 h on an M5 at 512 tokens, 3 epochs; plug in the laptop):
    uv run python src/finetune.py
    # then score:  uv run python src/evaluate.py

Same protocol as every other model: train on train only, pick the best epoch and one global
decision threshold on val, score test once.

Input text starts at the first [ITEM] marker: the cover page before it is identical
boilerplate on every 8-K, so skipping it lets 512 tokens cover the actual event.

Outputs:
  models/<name>/                      best checkpoint (gitignored)
  results/predictions_<name>.parquet  per-label probabilities for val + test (evaluate.py format)
  results/metrics_<name>.json         threshold, training config, per-epoch val scores, timings
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
MODEL_ID = "answerdotai/ModernBERT-base"
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)


def body(text: str) -> str:
    i = text.find("[ITEM]")
    return text[i:] if i >= 0 else text


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def encode(tok, texts: list[str], max_len: int) -> list[list[int]]:
    return tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]


def batches(ids: list[list[int]], Y: np.ndarray | None, bs: int, pad_id: int, shuffle: bool, rng: random.Random):
    """Yield padded batches. For inference, sort by length so padding stays small."""
    order = list(range(len(ids)))
    if shuffle:
        rng.shuffle(order)
    else:
        order.sort(key=lambda i: len(ids[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
            m[r, :len(ids[i])] = 1
        y = torch.tensor(Y[idx], dtype=torch.float32) if Y is not None else None
        yield idx, x, m, y


@torch.no_grad()
def predict(model, ids, bs, pad_id, device) -> tuple[np.ndarray, float]:
    model.eval()
    probs = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    t0 = time.perf_counter()
    for idx, x, m, _ in batches(ids, None, bs, pad_id, shuffle=False, rng=random.Random(0)):
        logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
        probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    sync(device)
    return probs, (time.perf_counter() - t0) / max(1, len(ids)) * 1000


def best_threshold(Y: np.ndarray, P: np.ndarray) -> tuple[float, dict]:
    sweep = {float(t): f1_score(Y, (P >= t).astype(int), average="micro", zero_division=0) for t in THRESHOLDS}
    return max(sweep, key=sweep.get), sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="modernbert")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step; effective batch = batch-size x grad-accum")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps for linear warmup")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny slice, 1 epoch, results name <name>_smoke")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # weights are already cached; never touch the network
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    set_seed(args.seed)
    device = pick_device()
    name = f"{args.name}_smoke" if args.smoke else args.name

    df = pd.read_parquet(DATA / "dataset.parquet")
    labels = json.loads((DATA / "labels.json").read_text())["labels"]
    if args.smoke:
        args.epochs = 1
        df = pd.concat([g.head(n) for s, g in df.groupby("split") for n in [{"train": 160, "val": 64, "test": 64}[s]]])

    split = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    Y = {s: np.array([[l in set(ls) for l in labels] for ls in d["labels"]], dtype=np.float32) for s, d in split.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.perf_counter()
    ids = {s: encode(tok, [body(t) for t in d["text"]], args.max_len) for s, d in split.items()}
    truncated = {s: float(np.mean([len(x) >= args.max_len for x in v])) for s, v in ids.items()}
    print(f"device {device} | {', '.join(f'{s} {len(v):,}' for s, v in ids.items())} | tokenized in {time.perf_counter()-t0:.0f}s | "
          f"share hitting {args.max_len} tokens: train {truncated['train']:.0%}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), problem_type="multi_label_classification").to(device)
    no_decay = ("bias", "norm")
    groups = [
        {"params": [p_ for n, p_ in model.named_parameters() if not any(k in n.lower() for k in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p_ for n, p_ in model.named_parameters() if any(k in n.lower() for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(ids["train"]) / args.batch_size)        # micro-batches
    total = math.ceil(steps_per_epoch / args.grad_accum) * args.epochs        # optimizer steps
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = MODELS / name
    rng = random.Random(args.seed)
    history, best = [], {"micro_f1": -1.0}
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep, running = time.perf_counter(), 0.0
        for step, (_, x, m, y) in enumerate(batches(ids["train"], Y["train"], args.batch_size, tok.pad_token_id, True, rng), 1):
            logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
            loss = loss_fn(logits.float(), y.to(device))
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 100 == 0 or step == steps_per_epoch:
                el = time.perf_counter() - t_ep
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running/step:.4f} "
                      f"({el/60:.1f} min, ~{el/step*(steps_per_epoch-step)/60:.0f} min left in epoch)")
        if device.type == "mps":
            torch.mps.empty_cache()

        P_val, _ = predict(model, ids["val"], args.batch_size * 2, tok.pad_token_id, device)
        th, _ = best_threshold(Y["val"], P_val)
        micro = f1_score(Y["val"], (P_val >= th).astype(int), average="micro", zero_division=0)
        macro = f1_score(Y["val"], (P_val >= th).astype(int), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_micro_f1": micro,
                        "val_macro_f1": macro, "val_threshold": th, "epoch_minutes": (time.perf_counter() - t_ep) / 60})
        print(f"epoch {epoch}: train loss {running/steps_per_epoch:.4f} | val micro-F1 {micro:.3f}, macro-F1 {macro:.3f} at threshold {th:.2f}")
        if micro > best["micro_f1"]:
            best = {"micro_f1": micro, "epoch": epoch}
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    train_minutes = (time.perf_counter() - train_start) / 60

    # Reload the best epoch, choose the threshold on val, then score test once.
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    P_val, _ = predict(model, ids["val"], args.batch_size * 2, tok.pad_token_id, device)
    threshold, sweep = best_threshold(Y["val"], P_val)
    P_test, ms_per_doc = predict(model, ids["test"], args.batch_size * 2, tok.pad_token_id, device)

    rows = []
    for s, P in (("val", P_val), ("test", P_test)):
        d = split[s][["accession", "filing_date", "labels"]].copy()
        d.insert(1, "split", s)
        rows.append(pd.concat([d.reset_index(drop=True), pd.DataFrame(P, columns=[f"p_{l}" for l in labels])], axis=1))
    pd.concat(rows).to_parquet(RESULTS / f"predictions_{name}.parquet", index=False)

    metrics = {
        "model": name, "base_model": MODEL_ID, "git_commit": git_commit(), "device": str(device),
        "threshold": threshold, "threshold_rule": "global, max val micro-F1 over 0.20-0.80 step 0.05",
        "val_threshold_sweep": sweep, "best_epoch": best["epoch"], "history": history,
        "config": {k: v for k, v in vars(args).items() if k != "smoke"}, "smoke": args.smoke,
        "input": "text from the first [ITEM] marker, truncated to max_len tokens",
        "share_truncated": truncated, "n": {s: len(v) for s, v in ids.items()},
        "train_minutes": round(train_minutes, 1), "test_inference_ms_per_doc": round(ms_per_doc, 2),
        "torch": torch.__version__,
    }
    (RESULTS / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nbest epoch {best['epoch']}, threshold {threshold:.2f} from val | trained in {train_minutes:.1f} min | "
          f"test inference {ms_per_doc:.1f} ms/filing on {device}")
    print(f"Wrote results/predictions_{name}.parquet, results/metrics_{name}.json, models/{name}/. "
          f"Score with: uv run python src/evaluate.py")


if __name__ == "__main__":
    main()
