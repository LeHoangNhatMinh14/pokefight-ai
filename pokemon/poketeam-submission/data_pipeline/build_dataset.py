"""
Parse cached Smogon chaos JSONs into three tidy tables:

    pokemon_meta.parquet   (gen, tier, name, usage, raw_count, viability_ceiling, battles)
    movesets.parquet       (gen, tier, name, kind, value, weight)   long form
    teammates.parquet      (gen, tier, a, b, weight)                long form

`kind` is one of: 'move', 'item', 'ability', 'spread', 'tera', 'happiness'.
Weights are kept as raw counts (Smogon's "weighted" stat = battles * usage).

We also join in the base stats from pokemon_final.csv so downstream code has
types/weaknesses/role/legendary in the same place.

Usage:
    python data_pipeline/build_dataset.py
    python data_pipeline/build_dataset.py --chaos-dir data/raw/chaos --extra data/fixtures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# ---------- name normalization ---------------------------------------------------

# Smogon names use Title Case with spaces/dashes and special chars (e.g. "Mr. Mime",
# "Nidoran-F", "Mime Jr.", "Tapu Koko"). pokemon_final.csv uses lowercase compact
# names. We canonicalize to lowercase with single-hyphen separators where possible.

_NAME_FIXUPS = {
    "nidoran-f": "nidoran-f",
    "nidoran-m": "nidoran-m",
    "farfetchd": "farfetch'd",
    "mr. mime": "mr-mime",
    "mime jr.": "mime-jr",
    "ho-oh": "ho-oh",
    "porygon-z": "porygon-z",
    "porygon2": "porygon2",
    "type: null": "type-null",
    "jangmo-o": "jangmo-o",
    "hakamo-o": "hakamo-o",
    "kommo-o": "kommo-o",
    "tapu koko": "tapu-koko",
    "tapu lele": "tapu-lele",
    "tapu bulu": "tapu-bulu",
    "tapu fini": "tapu-fini",
}


def normalize_name(name: str) -> str:
    """Canonical lowercase form used to join against pokemon_final.csv."""
    n = unicodedata.normalize("NFKC", name).strip().lower()
    if n in _NAME_FIXUPS:
        return _NAME_FIXUPS[n]
    # Strip trailing dots, swap spaces -> hyphens
    n = n.replace(".", "").replace("'", "").replace(":", "")
    n = re.sub(r"\s+", "-", n)
    return n


# ---------- chaos JSON discovery -------------------------------------------------

_PATH_RE = re.compile(r"gen(?P<gen>\d+)(?P<tier>[a-z]+)-(?P<rating>\d+)\.json$")
_TIER_RATING_RE = re.compile(r"(?P<tier>[a-z]+)-(?P<rating>\d+)\.json$")
_GEN_DIR_RE = re.compile(r"gen(?P<gen>\d+)$")


def discover_chaos_files(*roots: Path) -> list[tuple[Path, int, str, int]]:
    """Walk roots looking for chaos files.

    Supports two layouts:
      fixture style:   .../gen3ou-1500.json
      fetcher style:   .../gen3/ou-1500.json   (gen in parent folder)
    """
    found: list[tuple[Path, int, str, int]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            m = _PATH_RE.search(p.name)
            if m:
                found.append((p, int(m["gen"]), m["tier"], int(m["rating"])))
                continue
            m2 = _TIER_RATING_RE.search(p.name)
            parent_match = _GEN_DIR_RE.match(p.parent.name)
            if m2 and parent_match:
                found.append((p, int(parent_match["gen"]), m2["tier"], int(m2["rating"])))
    # When multiple ratings exist for the same (gen, tier) keep the highest rating
    # (more competitive ladder).
    best: dict[tuple[int, str], tuple[Path, int, str, int]] = {}
    for entry in found:
        path, gen, tier, rating = entry
        key = (gen, tier)
        if key not in best or rating > best[key][3]:
            best[key] = entry
    return list(best.values())


# ---------- parsing ---------------------------------------------------------------

def parse_chaos(payload: dict, gen: int, tier: str) -> tuple[list, list, list]:
    """Convert one chaos JSON into (meta_rows, moveset_rows, teammate_rows)."""
    info = payload.get("info", {}) or {}
    battles = int(info.get("number of battles", 0))

    meta_rows: list[dict] = []
    moveset_rows: list[dict] = []
    teammate_rows: list[dict] = []

    data = payload.get("data", {}) or {}
    for raw_name, entry in data.items():
        name = normalize_name(raw_name)
        raw_count = float(entry.get("Raw count", 0) or 0)
        usage = float(entry.get("usage", 0) or 0)
        vc = entry.get("Viability Ceiling") or [0, 0, 0, 0]
        viability_top = float(vc[0]) if vc else 0.0

        meta_rows.append({
            "gen": gen, "tier": tier, "name": name,
            "raw_count": raw_count,
            "usage": usage,
            "viability_top": viability_top,
            "battles": battles,
        })

        for kind_key, kind_label in [
            ("Moves", "move"),
            ("Items", "item"),
            ("Abilities", "ability"),
            ("Spreads", "spread"),
            ("Tera Types", "tera"),
            ("Happiness", "happiness"),
        ]:
            dist = entry.get(kind_key, {}) or {}
            for value, weight in dist.items():
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    continue
                if w <= 0:
                    continue
                moveset_rows.append({
                    "gen": gen, "tier": tier, "name": name,
                    "kind": kind_label, "value": str(value), "weight": w,
                })

        for partner_raw, weight in (entry.get("Teammates", {}) or {}).items():
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            if w == 0:
                continue
            teammate_rows.append({
                "gen": gen, "tier": tier,
                "a": name, "b": normalize_name(partner_raw),
                "weight": w,
            })

    return meta_rows, moveset_rows, teammate_rows


def load_all(chaos_files: Iterable[tuple[Path, int, str, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metas, moves, teams = [], [], []
    for path, gen, tier, _rating in chaos_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  [skip bad json] {path}: {e}", file=sys.stderr)
            continue
        m, mv, tm = parse_chaos(payload, gen, tier)
        metas.extend(m)
        moves.extend(mv)
        teams.extend(tm)
        print(f"  parsed gen{gen}{tier}: {len(m)} mons, {len(mv)} moveset rows, {len(tm)} teammate edges")

    meta_df = pd.DataFrame(metas)
    move_df = pd.DataFrame(moves)
    team_df = pd.DataFrame(teams)
    return meta_df, move_df, team_df


# ---------- normalization with pokemon_final.csv ---------------------------------

def join_stats(meta_df: pd.DataFrame, base_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(base_csv)
    base["name"] = base["name"].apply(normalize_name)
    cols = [c for c in [
        "name", "type1", "type2", "hp", "attack", "defense",
        "sp_attack", "sp_defense", "speed", "is_legendary",
        "weaknesses", "role",
    ] if c in base.columns]
    return meta_df.merge(base[cols], on="name", how="left")


def report_missing(meta_df: pd.DataFrame) -> None:
    """Show Pokemon in chaos data that we couldn't join to pokemon_final.csv."""
    if "type1" not in meta_df.columns:
        return
    missing = meta_df[meta_df["type1"].isna()]["name"].unique()
    if len(missing):
        print(f"  [warn] {len(missing)} chaos names didn't join to base CSV:")
        for n in sorted(missing)[:20]:
            print(f"          - {n}")


# ---------- main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chaos-dir", type=str, default="data/raw/chaos")
    parser.add_argument(
        "--extra", type=str, nargs="*", default=["data/fixtures"],
        help="Extra directories of chaos JSON files (e.g. fixtures).",
    )
    parser.add_argument("--base-csv", type=str, default="data/processed/pokemon_final.csv")
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()

    roots = [Path(args.chaos_dir)] + [Path(p) for p in (args.extra or [])]
    chaos_files = discover_chaos_files(*roots)
    if not chaos_files:
        print("No chaos JSON files found. Run data_pipeline/fetch_chaos.py first.",
              file=sys.stderr)
        return 1

    print(f"Found {len(chaos_files)} (gen, tier) chaos files:")
    for path, gen, tier, rating in sorted(chaos_files, key=lambda x: (x[1], x[2])):
        print(f"  gen{gen}{tier:>14} (rating {rating})  <-  {path}")

    meta_df, move_df, team_df = load_all(chaos_files)
    if meta_df.empty:
        print("Parsed no rows.", file=sys.stderr)
        return 1

    meta_df = join_stats(meta_df, Path(args.base_csv))
    report_missing(meta_df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(out / "pokemon_meta.parquet", index=False)
    move_df.to_parquet(out / "movesets.parquet", index=False)
    team_df.to_parquet(out / "teammates.parquet", index=False)

    print(f"\nWrote:")
    print(f"  {out/'pokemon_meta.parquet'}  ({len(meta_df):,} rows)")
    print(f"  {out/'movesets.parquet'}      ({len(move_df):,} rows)")
    print(f"  {out/'teammates.parquet'}     ({len(team_df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
