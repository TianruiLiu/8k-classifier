"""Turn downloaded 8-K filings into a labeled, leak-masked, time-split dataset.

    uv run python src/build.py                 # all downloaded filings -> data/dataset.parquet
    uv run python src/build.py --limit 500     # quick test on the first 500 index rows

Steps:
  1. Parse each filing's HTML and keep only the visible text (drop the inline-XBRL
     header, hidden elements, scripts and styles).
  2. Mask label leakage: every "Item X.XX" code and every official 8-K item title is
     replaced with [ITEM], so a model can't read the answer off the headings.
  3. Labels come from EDGAR's own Item list (data/index.csv). 9.01 (exhibits) is dropped
     because it is on most filings and says nothing about the event; codes with fewer
     than --min-count examples are folded into "other" (or dropped with --rare drop).
  4. Time-based split: oldest 70% train, next 15% val, newest 15% test.

Outputs: data/dataset.parquet, data/labels.json (label list, counts, masking stats).
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FILINGS_DIR = DATA / "raw" / "filings"
INDEX_CSV = DATA / "index.csv"
OUT_PARQUET = DATA / "dataset.parquet"
OUT_LABELS = DATA / "labels.json"

DROP_CODES = {"9.01"}
MIN_CHARS = 200
MAX_CHARS = 100_000   # a few filings are huge; the first 100K chars hold the event
MASK = " [ITEM] "

# Official 8-K item titles (Form 8-K General Instructions). Titles with several parts
# are split on ";" so each part is masked even when a filer quotes only one of them.
ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def _title_pattern(phrase: str) -> str:
    # Match the title's words in order, allowing any punctuation/whitespace between them,
    # so "Registrant's", "Registrant’s" and "Off-Balance" / "Off Balance" all match.
    words = re.findall(r"[A-Za-z]+", phrase)
    return r"\b" + r"\W+".join(words) + r"\b"


_parts = sorted({p.strip() for t in ITEM_TITLES.values() for p in t.split(";") if p.strip()}, key=len, reverse=True)
TITLE_RE = re.compile("|".join(_title_pattern(p) for p in _parts), re.I)
CODE_RE = re.compile(r"\bItems?\s*\d{1,2}\s*\.\s*\d{2}\b", re.I)
MASK_RUN_RE = re.compile(r"(?:\[ITEM\][\s;,.:()\-–—]*){2,}")
HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
WS_RE = re.compile(r"\s+")


def html_to_text(raw: bytes) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "ix:header", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN_RE):
        tag.decompose()
    return WS_RE.sub(" ", soup.get_text(" ")).strip()


def mask_leaks(text: str) -> tuple[str, int, int]:
    text, n_codes = CODE_RE.subn(MASK, text)
    text, n_titles = TITLE_RE.subn(MASK, text)
    # Collapse runs like "[ITEM] [ITEM] ; [ITEM] ; [ITEM]" into one token: the number of
    # title parts (5.02 has four) would otherwise still hint at the label.
    text = MASK_RUN_RE.sub(MASK, text)
    return WS_RE.sub(" ", text).strip(), n_codes, n_titles


def process(accession: str) -> dict | None:
    path = FILINGS_DIR / f"{accession}.htm"
    if not path.exists():
        return None
    text = html_to_text(path.read_bytes())[:MAX_CHARS]
    masked, n_codes, n_titles = mask_leaks(text)
    return {"accession": accession, "text": masked, "n_codes": n_codes, "n_titles": n_titles,
            "unmasked_chars": len(text)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-count", type=int, default=50, help="codes with fewer examples are rare (default 50)")
    p.add_argument("--rare", choices=["other", "drop"], default="other", help="fold rare codes into 'other', or drop them")
    p.add_argument("--limit", type=int, default=0, help="only the first N index rows (0 = all)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    index = pd.read_csv(INDEX_CSV, dtype={"cik": int, "accession": str, "items": str})
    if args.limit:
        index = index.head(args.limit)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(process, index["accession"], chunksize=32) if r is not None]
    texts = pd.DataFrame(results)
    print(f"{len(index)} filings in index; {len(texts)} downloaded and parsed")
    if texts.empty:
        raise SystemExit("Nothing parsed yet. Is the download running?")

    df = index.merge(texts, on="accession")
    too_short = df["text"].str.len() < MIN_CHARS
    df = df[~too_short].copy()

    # Labels: EDGAR's item list, minus 9.01, with rare codes folded or dropped.
    codes = df["items"].str.split(",").map(lambda xs: sorted({x.strip() for x in xs} - DROP_CODES - {""}))
    counts = Counter(c for cs in codes for c in cs)
    common = {c for c, n in counts.items() if n >= args.min_count}
    rare = sorted(set(counts) - common)
    if args.rare == "other":
        df["labels"] = codes.map(lambda cs: sorted({c if c in common else "other" for c in cs}))
    else:
        df["labels"] = codes.map(lambda cs: [c for c in cs if c in common])
    no_label = df["labels"].map(len) == 0      # e.g. filings whose only item was 9.01
    df = df[~no_label].copy()

    # Time-based split: train on the past, test on the future.
    df = df.sort_values(["filing_date", "accession"]).reset_index(drop=True)
    n = len(df)
    df["split"] = "train"
    df.loc[int(n * 0.70):, "split"] = "val"
    df.loc[int(n * 0.85):, "split"] = "test"

    # Leak check: anything that still looks like an item heading after masking.
    leftover = df["text"].str.contains(r"\bItem\s*\d{1,2}\.\d{2}\b", case=False, regex=True)

    label_list = sorted({l for ls in df["labels"] for l in ls}, key=lambda c: (c == "other", c))
    per_split = {s: Counter(l for ls in g["labels"] for l in ls) for s, g in df.groupby("split")}
    print(f"Dropped {int(too_short.sum())} under {MIN_CHARS} chars and {int(no_label.sum())} with no label left")
    print(f"Kept {n} filings, {len(label_list)} labels; rare codes {'-> other' if args.rare == 'other' else 'dropped'}: {', '.join(rare) or 'none'}")
    print(f"Masked {int(df['n_codes'].sum())} item codes and {int(df['n_titles'].sum())} item titles; "
          f"{int(leftover.sum())} filings still contain an item code (should be 0)")
    print("\nlabel   train    val   test")
    for l in label_list:
        print(f"{l:6} {per_split.get('train', {}).get(l, 0):6} {per_split.get('val', {}).get(l, 0):6} {per_split.get('test', {}).get(l, 0):6}")
    for s in ("train", "val", "test"):
        g = df[df["split"] == s]
        print(f"{s}: {len(g)} filings, {g['filing_date'].min()} to {g['filing_date'].max()}")
    thin = [l for l in label_list if per_split.get("test", {}).get(l, 0) < 5]
    if thin:
        print(f"\nWarning: fewer than 5 test examples for {', '.join(thin)}. Their test F1 will be noisy; say so in RESULTS.md.")

    cols = ["accession", "cik", "ticker", "filing_date", "items", "labels", "split", "text"]
    df[cols].to_parquet(OUT_PARQUET, index=False)
    OUT_LABELS.write_text(json.dumps({
        "labels": label_list, "dropped_codes": sorted(DROP_CODES), "rare_codes": rare, "rare_handling": args.rare,
        "min_count": args.min_count, "counts_all": dict(sorted(counts.items())),
        "per_split": {s: dict(sorted(c.items())) for s, c in per_split.items()},
        "n_filings": n, "split_rule": "time-based 70/15/15 by filing_date",
        "masked_codes": int(df["n_codes"].sum()), "masked_titles": int(df["n_titles"].sum()),
        "filings_with_leftover_codes": int(leftover.sum()),
    }, indent=2))
    print(f"\nWrote {OUT_PARQUET.relative_to(ROOT)} and {OUT_LABELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
