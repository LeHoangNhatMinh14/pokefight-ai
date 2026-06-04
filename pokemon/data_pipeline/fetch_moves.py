"""
Pre-fetch all PokeAPI move data so the synergy recommender doesn't have to
do ~400 HTTP requests per team build.

Run this ONCE. Produces:

    data/processed/move_catalog.parquet   one row per move: name, type, power, accuracy, damage_class
    data/processed/pokemon_moves.parquet  one row per (pokemon, move) pair

After this completes, recommender.py can read straight from disk and team builds
become near-instant.

Usage:
    python data_pipeline/fetch_moves.py
    python data_pipeline/fetch_moves.py --workers 12   (more parallelism)
    python data_pipeline/fetch_moves.py --resume       (skip already-fetched)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = "https://pokeapi.co/api/v2"
HEADERS = {"User-Agent": "pokefight-ai/0.1 (move-cache)"}


def fetch_json(session: requests.Session, url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def list_all_move_urls(session: requests.Session) -> list[tuple[str, str]]:
    """Return a list of (move_name, url) for every move in the game."""
    out: list[tuple[str, str]] = []
    url = f"{BASE}/move?limit=2000"
    data = fetch_json(session, url)
    if not data:
        return out
    for entry in data.get("results", []):
        out.append((entry["name"], entry["url"]))
    return out


def fetch_move_detail(session: requests.Session, name: str, url: str) -> dict | None:
    data = fetch_json(session, url)
    if not data:
        return None
    move_type = (data.get("type") or {}).get("name")
    damage_class = (data.get("damage_class") or {}).get("name")
    return {
        "move": name,
        "type": move_type,
        "power": data.get("power"),
        "accuracy": data.get("accuracy"),
        "damage_class": damage_class,
    }


def fetch_pokemon_moves(session: requests.Session, pokemon_name: str) -> list[dict] | None:
    """Return [{"name": pkm, "move": move_name}, ...] for one Pokemon."""
    data = fetch_json(session, f"{BASE}/pokemon/{pokemon_name.lower()}")
    if not data:
        return None
    moves = []
    for m in data.get("moves", []):
        nm = (m.get("move") or {}).get("name")
        if nm:
            moves.append({"name": pokemon_name, "move": nm})
    return moves


def build_move_catalog(workers: int, out_path: Path, resume: bool) -> pd.DataFrame:
    """Phase 1: fetch every move in the game with its details."""
    if resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"  [resume] move catalog has {len(existing)} rows")
    else:
        existing = pd.DataFrame(columns=["move", "type", "power", "accuracy", "damage_class"])

    session = requests.Session()
    print("Listing all moves...")
    move_urls = list_all_move_urls(session)
    print(f"  found {len(move_urls)} moves in PokeAPI")

    known = set(existing["move"]) if not existing.empty else set()
    todo = [(n, u) for n, u in move_urls if n not in known]
    print(f"  fetching {len(todo)} new move details ({len(known)} already cached)")

    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    if not todo:
        return pd.DataFrame(rows)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_move_detail, session, n, u): n for n, u in todo}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is not None:
                rows.append(r)
            if done % 100 == 0 or done == len(todo):
                print(f"    {done}/{len(todo)} moves fetched ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows).drop_duplicates(subset=["move"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} moves -> {out_path}")
    return df


def build_pokemon_moves(workers: int, names: list[str], out_path: Path, resume: bool) -> pd.DataFrame:
    """Phase 2: for each Pokemon, save the list of move names it can learn."""
    if resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        known = set(existing["name"]) if not existing.empty else set()
        print(f"  [resume] pokemon_moves has {len(known)} Pokemon already cached")
    else:
        existing = pd.DataFrame(columns=["name", "move"])
        known = set()

    todo = [n for n in names if n not in known]
    print(f"  fetching moves for {len(todo)} Pokemon ({len(known)} already cached)")

    session = requests.Session()
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []

    if not todo:
        return pd.DataFrame(rows)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_pokemon_moves, session, n): n for n in todo}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                rows.extend(r)
            if done % 50 == 0 or done == len(todo):
                print(f"    {done}/{len(todo)} Pokemon fetched ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows).drop_duplicates(subset=["name", "move"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} (Pokemon, move) pairs -> {out_path}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", default="data/processed/pokemon_final.csv")
    parser.add_argument("--catalog-out", default="data/processed/move_catalog.parquet")
    parser.add_argument("--pokemon-out", default="data/processed/pokemon_moves.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Skip moves/Pokemon already in the cache files (recommended).")
    args = parser.parse_args()

    base = Path(args.base_csv)
    if not base.exists():
        print(f"Missing {base}", file=sys.stderr)
        return 1
    names = pd.read_csv(base)["name"].tolist()
    print(f"Loaded {len(names)} Pokemon from {base}")

    print("\nPhase 1: move catalog")
    build_move_catalog(args.workers, Path(args.catalog_out), args.resume)

    print("\nPhase 2: per-Pokemon move lists")
    build_pokemon_moves(args.workers, names, Path(args.pokemon_out), args.resume)

    print("\nDone. Recommender will use the cache automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
