"""Scrape SEC 8-K filings and their Item-code labels from EDGAR.

Two steps, both safe to rerun (everything is cached on disk):

    uv run python src/scrape.py index      # company list -> data/index.csv (one row per 8-K)
    uv run python src/scrape.py download   # data/index.csv -> data/raw/filings/*.htm

SEC requires a User-Agent with your name and a contact email, and allows at most
10 requests/second. Set it once per terminal:

    export SEC_USER_AGENT="Your Name your@email.com"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SUBMISSIONS_DIR = RAW / "submissions"
FILINGS_DIR = RAW / "filings"
INDEX_CSV = DATA / "index.csv"
FAILURES_LOG = DATA / "failures.log"

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

MIN_GAP_S = 0.2          # ~5 requests/sec, half of SEC's limit
MAX_RETRIES = 4
INDEX_FIELDS = ["cik", "ticker", "company", "filing_date", "accession", "primary_doc", "items"]


class Client:
    """requests.Session with SEC's User-Agent, a global rate limit, and retries."""

    def __init__(self, user_agent: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        for attempt in range(MAX_RETRIES):
            wait = MIN_GAP_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))   # back off: 2s, 4s, 8s
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"unreachable: {url}")


def log_failure(url: str, err: Exception) -> None:
    with FAILURES_LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{url}\t{type(err).__name__}: {err}\n")


def load_companies(client: Client) -> list[dict]:
    """All tickers, deduplicated by CIK (some companies have several share classes)."""
    data = client.get(TICKERS_URL).json()
    by_cik: dict[int, dict] = {}
    for row in data.values():
        by_cik.setdefault(int(row["cik_str"]), {"cik": int(row["cik_str"]), "ticker": row["ticker"], "company": row["title"]})
    return sorted(by_cik.values(), key=lambda c: c["cik"])


def load_submissions(client: Client, cik: int) -> dict:
    cache = SUBMISSIONS_DIR / f"CIK{cik:010d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sub = client.get(SUBMISSIONS_URL.format(cik=cik)).json()
    cache.write_text(json.dumps(sub))
    return sub


def eight_ks(sub: dict, since: str, max_per_company: int) -> list[dict]:
    """8-K filings (no amendments) on or after `since` that list at least one Item code."""
    recent = sub.get("filings", {}).get("recent", {})
    rows = []
    for form, items, acc, date, doc in zip(
        recent.get("form", []), recent.get("items", []), recent.get("accessionNumber", []),
        recent.get("filingDate", []), recent.get("primaryDocument", []),
    ):
        if form != "8-K" or date < since or not items or not doc:
            continue
        rows.append({"filing_date": date, "accession": acc, "primary_doc": doc, "items": items})
    return rows[:max_per_company]   # `recent` is newest first


def cmd_index(args: argparse.Namespace, client: Client) -> None:
    companies = load_companies(client)
    random.Random(args.seed).shuffle(companies)
    sample = companies[: args.companies]
    print(f"{len(companies)} companies listed; sampling {len(sample)} (seed {args.seed})")

    n_rows = 0
    with INDEX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for i, c in enumerate(sample, 1):
            try:
                sub = load_submissions(client, c["cik"])
            except Exception as e:  # keep going; one bad company shouldn't stop the run
                log_failure(SUBMISSIONS_URL.format(cik=c["cik"]), e)
                continue
            for row in eight_ks(sub, args.since, args.max_per_company):
                w.writerow({**c, **row})
                n_rows += 1
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} companies, {n_rows} 8-Ks so far")
    print(f"Wrote {n_rows} 8-Ks to {INDEX_CSV.relative_to(ROOT)}")


def cmd_download(args: argparse.Namespace, client: Client) -> None:
    with INDEX_CSV.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not (FILINGS_DIR / f"{r['accession']}.htm").exists()]
    print(f"{len(rows)} filings in index; {len(rows) - len(todo)} already downloaded; {len(todo)} to go "
          f"(~{len(todo) * MIN_GAP_S / 60:.0f} min)")

    ok = failed = 0
    for i, r in enumerate(todo, 1):
        url = FILING_URL.format(cik=int(r["cik"]), acc_nodash=r["accession"].replace("-", ""), doc=r["primary_doc"])
        try:
            body = client.get(url).content
            (FILINGS_DIR / f"{r['accession']}.htm").write_bytes(body)
            ok += 1
        except Exception as e:
            log_failure(url, e)
            failed += 1
        if i % 200 == 0:
            print(f"  {i}/{len(todo)} done ({failed} failed)")
    print(f"Downloaded {ok}, failed {failed}. Failures (if any) are in {FAILURES_LOG.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="build data/index.csv from a random sample of companies")
    pi.add_argument("--companies", type=int, default=400, help="how many companies to sample (default 400)")
    pi.add_argument("--since", default="2021-01-01", help="earliest filing date, YYYY-MM-DD")
    pi.add_argument("--max-per-company", type=int, default=40, help="cap so a few heavy filers don't dominate")
    pi.add_argument("--seed", type=int, default=42)
    pd_ = sub.add_parser("download", help="download every filing listed in data/index.csv")
    pd_.add_argument("--limit", type=int, default=0, help="only the first N rows (0 = all); use 20 for a smoke test")
    args = p.parse_args()

    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua or " " not in ua:
        sys.exit('Set SEC_USER_AGENT first, e.g.  export SEC_USER_AGENT="Your Name your@email.com"')

    for d in (SUBMISSIONS_DIR, FILINGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    client = Client(ua)
    if args.cmd == "index":
        cmd_index(args, client)
    else:
        if not INDEX_CSV.exists():
            sys.exit("No data/index.csv yet. Run the index step first.")
        cmd_download(args, client)


if __name__ == "__main__":
    main()
