"""
Fetch Smogon chaos JSON for all tiers across Gen 1-6.

Smogon publishes monthly usage statistics at https://www.smogon.com/stats/.
The "chaos" JSON contains the richest per-Pokemon data:
    - raw usage + weighted "Viability Ceiling"
    - Abilities, Items, Moves, Spreads distributions
    - Teammates co-occurrence with weights
    - Checks and Counters

We fetch every tier we can find for gens 1-6 and cache to
    data/raw/chaos/{year-month}/gen{N}/{tier}.json

Usage:
    python data_pipeline/fetch_chaos.py              # latest month, all gens 1-6
    python data_pipeline/fetch_chaos.py --month 2025-04
    python data_pipeline/fetch_chaos.py --gens 3 4 5 --rating 1500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import requests

BASE = "https://www.smogon.com/stats"
HEADERS = {"User-Agent": "pokefight-ai team-builder/0.1 (educational)"}

# Per-gen tier candidates. Smogon doesn't publish every tier for every month, so we
# probe each one and skip the 404s. This list intentionally over-shoots.
TIER_CANDIDATES: dict[int, list[str]] = {
    1: ["ou", "ubers", "uu", "nu", "lc", "tradebacksou"],
    2: ["ou", "ubers", "uu", "nu", "lc"],
    3: ["ou", "ubers", "uu", "nu", "lc"],
    4: ["ou", "ubers", "uu", "nu", "lc", "anythinggoes"],
    5: ["ou", "ubers", "uu", "ru", "nu", "pu", "lc", "doublesou"],
    6: [
        "ou", "ubers", "uu", "ru", "nu", "pu", "lc",
        "monotype", "anythinggoes", "doublesou", "battlespotsingles",
    ],
}

# Rating tiers Smogon publishes per format. The fetcher will probe in this order
# and stop on the first that exists (chaos JSON for high-rating ladder is most
# competitive, but low-tier formats only get the 0-rating bucket).
RATING_CANDIDATES = [1825, 1695, 1630, 1500, 0]


def latest_known_month(reference: Optional[date] = None) -> str:
    """Smogon publishes stats ~3-5 days after a month ends. Default to last month."""
    today = reference or date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y}-{m:02d}"


def discover_months(max_lookback: int = 6) -> list[str]:
    """Months to try, newest first. We fall back through previous months if the
    latest one hasn't been published yet (or a specific format is missing)."""
    months = []
    today = date.today()
    y, m = today.year, today.month
    for _ in range(max_lookback):
        m -= 1
        if m == 0:
            y -= 1
            m = 12
        months.append(f"{y}-{m:02d}")
    return months


def chaos_url(month: str, gen: int, tier: str, rating: int) -> str:
    return f"{BASE}/{month}/chaos/gen{gen}{tier}-{rating}.json"


def fetch_one(
    month: str,
    gen: int,
    tier: str,
    rating: int,
    out_dir: Path,
    session: requests.Session,
    *,
    overwrite: bool = False,
) -> Optional[Path]:
    """Try a single (month, gen, tier, rating) and write to disk on success."""
    out_path = out_dir / month / f"gen{gen}" / f"{tier}-{rating}.json"
    if out_path.exists() and not overwrite:
        return out_path

    url = chaos_url(month, gen, tier, rating)
    try:
        r = session.get(url, timeout=30, headers=HEADERS)
    except requests.RequestException as e:
        print(f"  [net] {url} -> {e}", file=sys.stderr)
        return None

    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  [{r.status_code}] {url}", file=sys.stderr)
        return None

    try:
        payload = r.json()
    except json.JSONDecodeError:
        print(f"  [bad json] {url}", file=sys.stderr)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path


def fetch_tier(
    gen: int,
    tier: str,
    months: Iterable[str],
    ratings: Iterable[int],
    out_dir: Path,
    session: requests.Session,
    *,
    overwrite: bool = False,
) -> Optional[Path]:
    """For one (gen, tier), find the first (month, rating) combo that exists."""
    for month in months:
        for rating in ratings:
            path = fetch_one(
                month, gen, tier, rating, out_dir, session, overwrite=overwrite
            )
            if path is not None:
                return path
    return None


def fetch_all(
    gens: Iterable[int],
    months: list[str],
    ratings: list[int],
    out_dir: Path,
    *,
    overwrite: bool = False,
    workers: int = 6,
) -> dict[tuple[int, str], Optional[Path]]:
    """Fetch every available tier for every gen, in parallel."""
    session = requests.Session()
    jobs: list[tuple[int, str]] = []
    for gen in gens:
        for tier in TIER_CANDIDATES.get(gen, []):
            jobs.append((gen, tier))

    results: dict[tuple[int, str], Optional[Path]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                fetch_tier, gen, tier, months, ratings, out_dir, session,
                overwrite=overwrite,
            ): (gen, tier)
            for gen, tier in jobs
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                print(f"  [error] gen{key[0]}{key[1]}: {e}", file=sys.stderr)
                results[key] = None
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gens", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="YYYY-MM (defaults to walking back from latest until something is found)",
    )
    parser.add_argument("--rating", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/raw/chaos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.month:
        if not re.match(r"^\d{4}-\d{2}$", args.month):
            print("--month must be YYYY-MM", file=sys.stderr)
            return 2
        months = [args.month]
    else:
        months = discover_months()

    ratings = [args.rating] if args.rating is not None else RATING_CANDIDATES
    out_dir = Path(args.out)

    print(f"Fetching gens {args.gens}")
    print(f"Months (newest first): {months[:3]}{'...' if len(months) > 3 else ''}")
    print(f"Ratings (preferred first): {ratings}")
    print(f"Output: {out_dir.resolve()}")

    t0 = time.time()
    results = fetch_all(
        args.gens, months, ratings, out_dir,
        overwrite=args.overwrite, workers=args.workers,
    )
    dt = time.time() - t0

    found = sum(1 for v in results.values() if v is not None)
    print(f"\nDone in {dt:.1f}s — {found}/{len(results)} tiers found.")
    for (gen, tier), path in sorted(results.items()):
        status = path.relative_to(Path.cwd()) if path else "MISSING"
        print(f"  gen{gen}{tier:>14}  ->  {status}")
    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
