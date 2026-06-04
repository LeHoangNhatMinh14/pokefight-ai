"""
Precompute synergy teams for every (pokemon, playstyle, max_legendaries) combo.

This is the offline twin of the live synergy recommender. Synergy mode is
fully deterministic (same dataset + same inputs = same team) so there's no
reason to redo the ~3.6s lookahead loop on every web request when we can
bake the answers into a parquet once.

Output: data/processed/synergy_teams.parquet
Schema (one row per team slot):
  pokemon, playstyle, max_legendaries, slot_index,
  name, type1, type2, hp, attack, defense, sp_attack, sp_defense, speed,
  role, weaknesses (list[str]), recommended_moves (list[str]),
  is_legendary, reason

Sibling file `synergy_teams.manifest.json` records the SHA-256 of the input
CSV so the app can warn if the precomputed table is stale.

Usage:
    python tools/precompute_synergy_teams.py                  # full build
    python tools/precompute_synergy_teams.py --limit 8        # smoke test
    python tools/precompute_synergy_teams.py --resume         # restart safe
    python tools/precompute_synergy_teams.py --workers 4      # cap parallelism

Expected runtime on a typical laptop (8 cores, warm PokeAPI sqlite cache):
    ~1.5h for the full ~11,500 combos at lookahead_depth=1.

After it finishes, `app.py` will serve synergy requests from the parquet in
~1ms each. If the input CSV changes, just rerun this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

# Repo root on sys.path so the package imports work when invoked as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logging_config import configure_logging  # noqa: E402
from recommenders.recommender import (  # noqa: E402
    add_movesets_to_team,
    add_reasons_to_team,
    load_data,
    recommend_teammates,
)

configure_logging()
logger = logging.getLogger("precompute_synergy_teams")


PLAYSTYLES = ["balanced", "offense", "stall", "tank"]
MAX_LEG_CHOICES = [0, 1, 2, 6]
LOOKAHEAD_DEPTH = 1

DISPLAY_COLUMNS = [
    "name", "type1", "type2", "hp", "attack", "defense",
    "sp_attack", "sp_defense", "speed", "role", "weaknesses",
    "recommended_moves", "is_legendary", "reason",
]

OUT_COLUMNS = [
    "pokemon", "playstyle", "max_legendaries", "slot_index", *DISPLAY_COLUMNS,
]


# ---- multiprocessing worker ------------------------------------------------

# Each worker process lazily caches its own copy of the dataset so the
# CSV is parsed once per worker, not once per task.
_DF_CACHE: pd.DataFrame | None = None


def _df() -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = load_data()
    return _DF_CACHE


def _one_team(args: tuple[str, str, int]) -> tuple[str, str, int, list[dict], str | None]:
    """Compute a single team. Returns (pokemon, playstyle, max_leg, rows, err)."""
    pokemon, playstyle, max_leg = args
    try:
        df = _df()
        team = recommend_teammates(
            pokemon, df,
            playstyle=playstyle,
            max_legendaries=max_leg,
            lookahead_depth=LOOKAHEAD_DEPTH,
        )
        team = add_movesets_to_team(team, df)
        team = add_reasons_to_team(team, playstyle)
        # Replace NaNs with None so parquet doesn't choke on mixed lists/floats.
        team = team.where(team.notna(), None)

        rows: list[dict] = []
        for i, r in enumerate(team[DISPLAY_COLUMNS].to_dict(orient="records")):
            r["pokemon"] = pokemon
            r["playstyle"] = playstyle
            r["max_legendaries"] = max_leg
            r["slot_index"] = i
            rows.append(r)
        return pokemon, playstyle, max_leg, rows, None
    except Exception as e:  # noqa: BLE001 -- worker isolation
        return pokemon, playstyle, max_leg, [], f"{type(e).__name__}: {e}"


# ---- driver ----------------------------------------------------------------

def _fingerprint(csv_path: Path) -> str:
    h = hashlib.sha256()
    h.update(csv_path.read_bytes())
    return h.hexdigest()[:16]


def _flush(out_path: Path, new_rows: list[dict]) -> None:
    """Append new_rows to the parquet on disk (full rewrite for simplicity)."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows, columns=OUT_COLUMNS)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    # Drop duplicates so a re-run never multiplies rows.
    combined = combined.drop_duplicates(
        subset=["pokemon", "playstyle", "max_legendaries", "slot_index"],
        keep="last",
    )
    combined.to_parquet(out_path, index=False)


def _already_done(out_path: Path) -> set[tuple[str, str, int]]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path)
    # A combo is "done" only when all 6 slots are present.
    counts = (
        df.groupby(["pokemon", "playstyle", "max_legendaries"])["slot_index"]
        .count()
    )
    return {tuple(k) for k, n in counts.items() if n >= 6}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--output", default="data/processed/synergy_teams.parquet")
    p.add_argument("--resume", action="store_true",
                   help="Skip combos already present in the output parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="Only compute the first N combos (for smoke tests)")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="Flush rows to parquet every N completed teams")
    p.add_argument("--playstyles", nargs="+", default=PLAYSTYLES)
    p.add_argument("--max-legendaries", type=int, nargs="+", default=MAX_LEG_CHOICES)
    args = p.parse_args()

    csv_path = Path("data/processed/pokemon_final.csv")
    if not csv_path.exists():
        logger.error("missing %s", csv_path)
        return 1
    fp = _fingerprint(csv_path)
    logger.info("input fingerprint=%s", fp)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")

    df = load_data()
    all_pokemon = sorted(df["name"].astype(str).str.lower().tolist())
    logger.info("dataset rows=%d distinct=%d", len(df), len(all_pokemon))

    combos: list[tuple[str, str, int]] = [
        (pkmn, ps, ml)
        for pkmn in all_pokemon
        for ps in args.playstyles
        for ml in args.max_legendaries
    ]
    total_target = len(combos)

    if args.resume:
        done = _already_done(out_path)
        combos = [c for c in combos if c not in done]
        logger.info("resume: done=%d remaining=%d", len(done), len(combos))

    if args.limit is not None:
        combos = combos[: args.limit]

    if not combos:
        logger.info("nothing to do.")
        return 0

    logger.info(
        "computing %d combos with %d workers (lookahead=%d)",
        len(combos), args.workers, LOOKAHEAD_DEPTH,
    )

    rows_buffer: list[dict] = []
    errors: list[tuple[str, str, int, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        iterator = (_one_team(c) for c in combos)
    else:
        # `spawn` is required on Windows and is also safer on macOS.
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers)
        iterator = pool.imap_unordered(_one_team, combos, chunksize=4)

    try:
        for i, (pkmn, ps, ml, rows, err) in enumerate(iterator, 1):
            if err:
                errors.append((pkmn, ps, ml, err))
                logger.warning("error %s/%s/leg=%d: %s", pkmn, ps, ml, err)
            else:
                rows_buffer.extend(rows)

            if i % args.checkpoint_every == 0:
                _flush(out_path, rows_buffer)
                rows_buffer = []
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(combos) - i) / rate if rate else float("inf")
                logger.info(
                    "progress %d/%d (%.0f%%) %.2f teams/s ETA=%.0fs errors=%d",
                    i, len(combos), 100 * i / len(combos), rate, eta, len(errors),
                )
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    _flush(out_path, rows_buffer)

    manifest = {
        "input_csv": str(csv_path),
        "fingerprint": fp,
        "lookahead_depth": LOOKAHEAD_DEPTH,
        "playstyles": list(args.playstyles),
        "max_legendaries_set": list(args.max_legendaries),
        "combos_attempted": len(combos),
        "combos_in_table_after_run": total_target,
        "completed_at": pd.Timestamp.now().isoformat(),
        "errors_sample": errors[:10],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "done. wrote %s and %s. errors=%d total=%.1fs",
        out_path, manifest_path, len(errors), time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
