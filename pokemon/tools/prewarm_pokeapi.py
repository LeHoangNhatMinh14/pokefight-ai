"""
Pre-warm the PokeAPI disk cache.

The synergy recommender calls PokeAPI live for every team member's moves. On a
cold cache the first synergy team build can fire ~1000 HTTP requests and feel
painfully slow. This script walks every Pokemon in `data/processed/pokemon_final.csv`
and calls `get_pokemon_moves` on each one, which (thanks to the CachedSession in
recommenders/recommender.py) populates `.cache/pokeapi.sqlite` with the full
movepool + per-move metadata.

Run this once and every subsequent synergy team build is near-instant -- even
across `python app.py` restarts.

Usage:
    python tools/prewarm_pokeapi.py
    python tools/prewarm_pokeapi.py --limit 50           # quick smoke test
    python tools/prewarm_pokeapi.py --rate 5             # 5 req/sec (default 2)
    python tools/prewarm_pokeapi.py --pokemon charizard salamence
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the repo root importable so we can pull in the recommender module.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from recommenders.recommender import (
    get_pokemon_moves,
    fetch_json,
    _http,
    _CACHE_PATH,
)


def list_pokemon_names(limit: int | None = None) -> list[str]:
    csv_path = ROOT / "data" / "processed" / "pokemon_final.csv"
    df = pd.read_csv(csv_path)
    names = df["name"].astype(str).str.lower().tolist()
    if limit:
        names = names[:limit]
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only pre-warm the first N pokemon (useful for smoke tests).",
    )
    p.add_argument(
        "--rate", type=float, default=2.0,
        help="Max requests per second to PokeAPI (default 2 -- be polite).",
    )
    p.add_argument(
        "--pokemon", nargs="+", default=None,
        help="Explicit list of pokemon to warm. Overrides --limit.",
    )
    args = p.parse_args()

    if args.pokemon:
        names = [n.lower() for n in args.pokemon]
    else:
        names = list_pokemon_names(args.limit)

    print(f"Pre-warming PokeAPI cache for {len(names)} pokemon.")
    print(f"Cache file: {_CACHE_PATH}.sqlite")
    print(f"Session type: {type(_http).__name__}")
    print(f"Rate limit: ~{args.rate:.1f} req/sec\n")

    sleep_s = max(0.0, 1.0 / args.rate) if args.rate > 0 else 0.0
    t0 = time.time()
    ok = 0
    failed: list[tuple[str, str]] = []

    for i, name in enumerate(names, 1):
        try:
            moves = get_pokemon_moves(name)
            print(f"  [{i:>3}/{len(names)}] {name:<20} {len(moves)} moves")
            ok += 1
        except Exception as e:
            print(f"  [{i:>3}/{len(names)}] {name:<20} FAILED: {type(e).__name__}: {e}")
            failed.append((name, str(e)))
        time.sleep(sleep_s)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. {ok}/{len(names)} succeeded.")
    if failed:
        print(f"{len(failed)} failed:")
        for n, err in failed[:10]:
            print(f"  - {n}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Report final cache size so the user can see the work paid off.
    sqlite_file = Path(str(_CACHE_PATH) + ".sqlite")
    if sqlite_file.exists():
        size_kb = sqlite_file.stat().st_size / 1024
        print(f"\nCache file size: {size_kb:.1f} KB at {sqlite_file}")

    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
