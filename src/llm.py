"""Classify 8-K filings with an LLM (Anthropic or OpenAI), zero-shot or few-shot.

    # 1) see the prompt and a token estimate, no API calls:
    uv run python src/llm.py --provider anthropic --model claude-sonnet-5-5 --split val --n 200 --dry-run
    # 2) tune on 200 validation filings (fixed sample, seed 42):
    uv run python src/llm.py --provider anthropic --model claude-sonnet-5-5 --split val --n 200
    # 3) freeze the prompt, run the full test split once:
    uv run python src/llm.py --provider anthropic --model claude-sonnet-5-5 --split test
    # then score:  uv run python src/evaluate.py [--split val]

Keys come from the environment: ANTHROPIC_API_KEY / OPENAI_API_KEY (personal accounts).

Every response is cached at data/llm_cache/<name>/<prompt_hash>/<accession>.json, keyed by
everything that affects the answer (provider, model, system prompt, schema, thinking/effort).
Reruns are free, a crash resumes where it stopped, and changing the prompt starts a new folder.

Writes results/predictions_<name>.parquet (the format evaluate.py reads; 1.0/0.0 scores)
and results/metrics_<name>.json (threshold 0.5, failures, tokens, latency, optional cost).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = DATA / "llm_cache"
SEED = 42

# Official Form 8-K item titles plus a one-line trigger, used as label definitions.
# Giving the model the definitions is fair: what was masked is the titles inside each filing.
ITEMS = {
    "1.01": ("Entry into a Material Definitive Agreement", "the company signed or materially amended a significant contract (credit agreements, merger agreements, supply, licensing, leases)"),
    "1.02": ("Termination of a Material Definitive Agreement", "a significant contract was terminated, other than by expiring on its own terms"),
    "1.03": ("Bankruptcy or Receivership", "a bankruptcy filing or receivership"),
    "1.04": ("Mine Safety - Reporting of Shutdowns and Patterns of Violations", "a mine safety shutdown order or pattern-of-violations notice"),
    "1.05": ("Material Cybersecurity Incidents", "a material cybersecurity incident"),
    "2.01": ("Completion of Acquisition or Disposition of Assets", "an acquisition or sale of a significant business or assets has closed"),
    "2.02": ("Results of Operations and Financial Condition", "the company announced or released earnings or financial results for a completed period"),
    "2.03": ("Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement", "the company took on material debt: notes issued, loans drawn, credit facility borrowings"),
    "2.04": ("Triggering Events That Accelerate or Increase a Direct Financial Obligation", "a default or other event accelerated or increased a debt obligation"),
    "2.05": ("Costs Associated with Exit or Disposal Activities", "a restructuring, layoffs or facility closure with material charges"),
    "2.06": ("Material Impairments", "a material impairment charge"),
    "3.01": ("Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing", "an exchange delisting or listing-deficiency notice, or a transfer of listing"),
    "3.02": ("Unregistered Sales of Equity Securities", "the company sold or issued shares or convertibles without SEC registration (private placements, exempt issuances)"),
    "3.03": ("Material Modification to Rights of Security Holders", "changes to the rights of existing security holders (reverse splits, new preferred terms, charter changes affecting holders)"),
    "4.01": ("Changes in Registrant's Certifying Accountant", "the company's independent auditor resigned, was dismissed, or a new one was engaged"),
    "4.02": ("Non-Reliance on Previously Issued Financial Statements", "previously issued financial statements should no longer be relied on (restatement)"),
    "5.01": ("Changes in Control of Registrant", "a change in control of the company"),
    "5.02": ("Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers", "a director or executive officer left, was appointed or elected, or executive compensation arrangements changed"),
    "5.03": ("Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year", "the charter or bylaws were amended, or the fiscal year changed"),
    "5.04": ("Temporary Suspension of Trading Under Registrant's Employee Benefit Plans", "a blackout period for an employee benefit plan"),
    "5.05": ("Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics", "the code of ethics was amended or a waiver was granted"),
    "5.06": ("Change in Shell Company Status", "the company stopped being a shell company"),
    "5.07": ("Submission of Matters to a Vote of Security Holders", "results of a shareholder vote (annual or special meeting)"),
    "5.08": ("Shareholder Director Nominations", "the deadline for shareholder director nominations"),
    "7.01": ("Regulation FD Disclosure", "information furnished under Regulation FD, typically an investor presentation, press release or other public communication that is furnished rather than filed"),
    "8.01": ("Other Events", "any other event the company chooses to report that has no item of its own (e.g. dividends, buybacks, litigation updates, product news)"),
}


def build_system_prompt(labels: list[str], rare: list[str], examples: list[dict]) -> str:
    lines = [
        "You classify SEC Form 8-K filings by the Form 8-K items they report. A filing can report several items.",
        "",
        "Section headings in the filing have been replaced with [ITEM], so infer the items from the content.",
        "Ignore Item 9.01 (Financial Statements and Exhibits); it is not one of the labels.",
        "",
        "Labels:",
    ]
    for l in labels:
        if l == "other":
            continue
        title, trigger = ITEMS[l]
        lines.append(f"- {l}: {title}. Typical case: {trigger}.")
    if "other" in labels:
        rare_desc = "; ".join(f"{c} {ITEMS[c][0]}" for c in rare if c in ITEMS)
        lines.append(f"- other: any of these less common items: {rare_desc}.")
    # Tried and rejected on val (Nov 24): an explicit "7.01 vs 8.01" rule lowered 8.01 F1 from 0.73 to
    # 0.67 and micro-F1 from 0.896 to 0.878 on the same 200 filings. Recorded in RESULTS.md.
    lines += ["", "Return every label that applies, and only those."]
    if examples:
        lines += ["", "Examples (filing excerpt, then its correct labels):"]
        for i, ex in enumerate(examples, 1):
            lines += ["", f"Example {i}:", ex["snippet"], f"Labels: {json.dumps(ex['labels'])}"]
    return "\n".join(lines)


def label_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": labels}}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def body_snippet(text: str, n: int) -> str:
    i = text.find("[ITEM]")   # skip the cover page, which is identical boilerplate on every 8-K
    return (text[i:] if i >= 0 else text)[:n].strip()


def pick_examples(df: pd.DataFrame, labels: list[str], per_label: int, chars: int) -> list[dict]:
    """per_label training filings for each label, preferring single-label ones; fixed seed."""
    rng = random.Random(SEED)
    train = df[df["split"] == "train"]
    chosen, seen = [], set()
    for l in labels:
        has = train[train["labels"].map(lambda ls: l in ls)]
        single = has[has["labels"].map(len) == 1]
        pool = list((single if len(single) >= per_label else has)["accession"])
        rng.shuffle(pool)
        for acc in [a for a in pool if a not in seen][:per_label]:
            row = train[train["accession"] == acc].iloc[0]
            chosen.append({"accession": acc, "labels": list(row["labels"]), "snippet": body_snippet(row["text"], chars)})
            seen.add(acc)
    return chosen


# ---------- providers: each returns (labels, usage dict, stop_reason, raw_text) ----------

class AnthropicProvider:
    def __init__(self, model: str, thinking: str, effort: str | None, max_tokens: int):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.model, self.max_tokens = model, max_tokens
        self.extra: dict = {}
        if thinking == "off":
            # How each current model turns thinking off (see the Claude API docs for your model):
            if self.model.startswith("claude-sonnet-5-5"):
                self.extra["thinking"] = {"type": "between_tools"}   # 'disabled' is a 400 on Sonnet 5.5
            elif self.model.startswith("claude-haiku-5-5"):
                self.extra["thinking"] = {"type": "disabled"}        # accepted at effort high or below
            else:
                print(f"Note: no thinking-off setting configured for {model}; running its default (adaptive).")
        self.output_config: dict = {}
        if effort:
            self.output_config["effort"] = effort

    def settings(self) -> dict:
        return {"thinking": self.extra.get("thinking"), "output_config": self.output_config, "max_tokens": self.max_tokens}

    def classify(self, system: str, text: str, schema: dict):
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text}],
            output_config={**self.output_config, "format": {"type": "json_schema", "schema": schema}},
            **self.extra,
        )
        u = r.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": u.cache_read_input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0}
        raw = next((b.text for b in r.content if b.type == "text"), "")
        if r.stop_reason != "end_turn":   # refusal, max_tokens, ...: don't trust the output
            return None, usage, r.stop_reason, raw
        return json.loads(raw)["labels"], usage, r.stop_reason, raw


class OpenAIProvider:
    """Chat Completions with a strict JSON schema. Check OpenAI's current docs for your model:
    newer reasoning models take max_completion_tokens and reasoning_effort, and reject temperature."""

    def __init__(self, model: str, thinking: str, effort: str | None, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI(max_retries=6)
        self.model, self.max_tokens, self.effort = model, max_tokens, effort

    def settings(self) -> dict:
        return {"reasoning_effort": self.effort, "max_completion_tokens": self.max_tokens}

    def classify(self, system: str, text: str, schema: dict):
        kwargs = {"reasoning_effort": self.effort} if self.effort else {}
        r = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            response_format={"type": "json_schema", "json_schema": {"name": "labels", "schema": schema, "strict": True}},
            **kwargs,
        )
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        # OpenAI's prompt_tokens includes cached tokens; split them so cost math matches Anthropic's fields.
        usage = {"input": u.prompt_tokens - cached, "output": u.completion_tokens, "cache_read": cached, "cache_write": 0}
        choice = r.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
            return None, usage, choice.finish_reason, raw
        return json.loads(raw)["labels"], usage, choice.finish_reason, raw


# ---------- run ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    p.add_argument("--model", required=True, help="exact model ID from the provider's current model list")
    p.add_argument("--name", help="results name (default: <model>_<mode>), e.g. sonnet55_zeroshot")
    p.add_argument("--mode", choices=["zeroshot", "fewshot"], default="zeroshot")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--n", type=int, default=0, help="sample N filings from the split (seed 42); 0 = all")
    p.add_argument("--thinking", choices=["off", "default"], default="off", help="Anthropic only: turn thinking off where the model allows")
    p.add_argument("--effort", default=None, help="Anthropic effort (low/medium/high) or OpenAI reasoning_effort")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--examples-per-label", type=int, default=1)
    p.add_argument("--example-chars", type=int, default=1000)
    p.add_argument("--price", nargs=3, type=float, metavar=("IN", "OUT", "CACHE_READ"),
                   help="$ per 1M tokens, as of today, for the cost estimate (cache writes billed at 1.25x IN)")
    p.add_argument("--dry-run", action="store_true", help="print the prompt and sizes; no API calls")
    args = p.parse_args()

    df = pd.read_parquet(DATA / "dataset.parquet")
    meta = json.loads((DATA / "labels.json").read_text())
    labels, rare = meta["labels"], meta["rare_codes"]
    name = args.name or f"{args.model.replace('/', '_')}_{args.mode}"

    examples = pick_examples(df, labels, args.examples_per_label, args.example_chars) if args.mode == "fewshot" else []
    system = build_system_prompt(labels, rare, examples)
    schema = label_schema(labels)

    rows = df[df["split"] == args.split]
    if args.n:
        rows = rows.sample(n=min(args.n, len(rows)), random_state=SEED)
    rows = rows.sort_values("accession")

    if args.dry_run:
        print(system)
        print(f"\n--- {len(rows)} filings; system prompt {len(system):,} chars (~{len(system)//4:,} tokens); "
              f"documents {int(rows['text'].str.len().sum()):,} chars (~{int(rows['text'].str.len().sum())//4:,} tokens)")
        if examples:
            print(f"few-shot examples: {len(examples)} from train ({', '.join(e['accession'] for e in examples[:3])}, …)")
        return

    provider = (AnthropicProvider if args.provider == "anthropic" else OpenAIProvider)(
        args.model, args.thinking, args.effort, args.max_tokens)
    config = {"provider": args.provider, "model": args.model, "system": system, "schema": schema, "settings": provider.settings()}
    prompt_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = CACHE / name / prompt_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "_config.json").write_text(json.dumps({**config, "examples": [e["accession"] for e in examples]}, indent=2))

    def done(acc: str) -> bool:
        f = cache_dir / f"{acc}.json"
        return f.exists() and json.loads(f.read_text()).get("error") is None   # retry failures on rerun

    todo = [r for r in rows.itertuples() if not done(r.accession)]
    print(f"{name} [{prompt_hash}] {args.split}: {len(rows)} filings, {len(rows) - len(todo)} cached, {len(todo)} to call")

    def call(r) -> tuple[str, bool]:
        t0 = time.perf_counter()
        rec = {"accession": r.accession, "split": args.split, "chars": len(r.text)}
        try:
            out, usage, stop, raw = provider.classify(system, r.text, schema)
            rec.update(usage=usage, stop_reason=stop, raw=raw[:2000])
            if out is None:
                rec.update(labels=[], error=f"stop_reason={stop}")
            else:
                rec.update(labels=sorted({l for l in out if l in labels}), error=None)
        except Exception as e:   # keep going; failures are counted, not dropped
            rec.update(labels=[], error=f"{type(e).__name__}: {e}"[:500])
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        (cache_dir / f"{r.accession}.json").write_text(json.dumps(rec))
        return r.accession, rec["error"] is None

    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call, r) for r in todo]
        for i, f in enumerate(as_completed(futures), 1):
            failed += not f.result()[1]
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done ({failed} failed)")

    # Collect every cached record for this prompt (all splits run so far) into the shared format.
    recs = [json.loads(f.read_text()) for f in cache_dir.glob("*.json") if not f.name.startswith("_")]
    truth = df.set_index("accession")
    pred_rows = []
    for rec in recs:
        row = {"accession": rec["accession"], "filing_date": truth.at[rec["accession"], "filing_date"],
               "split": rec["split"], "labels": list(truth.at[rec["accession"], "labels"])}
        row.update({f"p_{l}": float(l in rec["labels"]) for l in labels})
        pred_rows.append(row)
    pd.DataFrame(pred_rows).sort_values(["split", "accession"]).to_parquet(RESULTS / f"predictions_{name}.parquet", index=False)

    # Run stats, per split.
    stats = {}
    for split in sorted({r["split"] for r in recs}):
        rs = [r for r in recs if r["split"] == split]
        ok = [r for r in rs if r.get("usage")]
        tok = {k: sum(r["usage"][k] for r in ok) for k in ("input", "output", "cache_read", "cache_write")}
        lat = sorted(r["latency_s"] for r in rs if r["error"] is None)
        s = {"n": len(rs), "failures": sum(r["error"] is not None for r in rs),
             "failure_examples": [r["error"] for r in rs if r["error"]][:5],
             "empty_label_lists": sum(r["error"] is None and not r["labels"] for r in rs),
             "tokens": tok, "avg_output_tokens": round(tok["output"] / max(1, len(ok)), 1),
             "latency_p50_s": statistics.median(lat) if lat else None,
             "latency_p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None}
        if args.price:
            pin, pout, pcache = args.price
            cost = (tok["input"] * pin + tok["output"] * pout + tok["cache_read"] * pcache + tok["cache_write"] * pin * 1.25) / 1e6
            s["cost_usd"] = round(cost, 4)
            s["cost_per_1k_filings_usd"] = round(cost / max(1, len(rs)) * 1000, 3)
            s["prices_per_1m"] = {"input": pin, "output": pout, "cache_read": pcache, "as_of": time.strftime("%Y-%m-%d")}
        stats[split] = s

    metrics = {"model": name, "provider": args.provider, "model_id": args.model, "mode": args.mode,
               "prompt_hash": prompt_hash, "threshold": 0.5, "settings": provider.settings(),
               "few_shot_examples": [e["accession"] for e in examples], "run_date": time.strftime("%Y-%m-%d"),
               "by_split": stats}
    (RESULTS / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))

    s = stats[args.split]
    print(f"\n{args.split}: {s['n']} filings, {s['failures']} failures, {s['empty_label_lists']} empty, "
          f"avg output {s['avg_output_tokens']} tokens, p50 {s['latency_p50_s']}s / p95 {s['latency_p95_s']}s"
          + (f", ${s['cost_usd']:.2f} (${s['cost_per_1k_filings_usd']:.2f} per 1K)" if "cost_usd" in s else ""))
    if s["failures"]:
        print("  e.g.", s["failure_examples"][0])
    if s["avg_output_tokens"] > 150:
        print("  Note: output is well above ~50 tokens per filing, so the model is probably thinking. Check --thinking/--effort.")
    print(f"Wrote results/predictions_{name}.parquet and results/metrics_{name}.json. "
          f"Score with: uv run python src/evaluate.py --split {args.split}")


if __name__ == "__main__":
    main()
