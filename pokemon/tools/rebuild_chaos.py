"""
One-shot rebuild of the Smogon chaos pipeline.

Runs, in order:
    1. data_pipeline/fetch_chaos.py     -- download chaos JSON for Gen 1-6
       (skippable with --skip-fetch if your data/raw/chaos/ is already populated)
    2. data_pipeline/build_dataset.py   -- parse JSON into the three parquet files
    3. data_pipeline/build_embeddings.py -- SVD over teammate co-occurrence
    4. model/train_viability.py         -- retrain the LightGBM viability reranker

After step 4, this script also normalizes model/viability.lgb to LF line endings.
LightGBM written on Windows can end up with CRLF line terminators, which the
C++ loader rejects with "Model format error, expect a tree here." Doing the
conversion here means you never have to think about it again.

Usage:
    python tools/rebuild_chaos.py
    python tools/rebuild_chaos.py --skip-fetch          # use raw JSON already on disk
    python tools/rebuild_chaos.py --skip-fetch --skip-train  # just rebuild parquets
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LGB_MODEL = ROOT / "model" / "viability.lgb"


def run(cmd: list[str], *, label: str) -> None:
    print(f"\n{'='*60}\n{label}\n  $ {' '.join(cmd)}\n{'='*60}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        sys.exit(f"!! step failed ({label}) with exit {res.returncode}")
    print(f"-- {label} done in {dt:.1f}s")


def normalize_lgb_to_lf(path: Path) -> None:
    """Convert CRLF -> LF in place if needed. Keeps a .crlf.bak first time."""
    if not path.exists():
        print(f"!! {path} not found; skipping CRLF normalization")
        return
    data = path.read_bytes()
    if b"\r\n" not in data:
        print(f"-- {path.name} already has LF line endings")
        return
    backup = path.with_suffix(path.suffix + ".crlf.bak")
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"-- saved CRLF backup: {backup.name}")
    path.write_bytes(data.replace(b"\r\n", b"\n"))
    print(f"-- normalized {path.name} to LF endings "
          f"({len(data)} -> {path.stat().st_size} bytes)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip Smogon JSON fetch; use whatever is already on disk.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip the LightGBM training step.")
    p.add_argument("--gens", nargs="+", type=int, default=None,
                   help="Gens to fetch (e.g. --gens 4 5 6). Default: 1-6.")
    p.add_argument("--embeddings-dim", type=int, default=32,
                   help="Embedding dimension for SVD (default 32).")
    args = p.parse_args()

    py = sys.executable
    t_start = time.time()

    if not args.skip_fetch:
        fetch_cmd = [py, "data_pipeline/fetch_chaos.py"]
        if args.gens:
            fetch_cmd += ["--gens"] + [str(g) for g in args.gens]
        run(fetch_cmd, label="Step 1/4: fetch_chaos.py")
    else:
        print("\n== Step 1/4: SKIPPED (--skip-fetch) ==")

    run([py, "data_pipeline/build_dataset.py"],
        label="Step 2/4: build_dataset.py")

    run([py, "data_pipeline/build_embeddings.py", "--dim", str(args.embeddings_dim)],
        label="Step 3/4: build_embeddings.py")

    if not args.skip_train:
        run([py, "-m", "model.train_viability"],
            label="Step 4/4: model.train_viability")
        # Always normalize after training, since LightGBM on Windows can write CRLF.
        normalize_lgb_to_lf(LGB_MODEL)
    else:
        print("\n== Step 4/4: SKIPPED (--skip-train) ==")
        # Still normalize whatever model is on disk -- the existing file might
        # have stale CRLF endings.
        normalize_lgb_to_lf(LGB_MODEL)

    total = time.time() - t_start
    print(f"\n{'='*60}\nAll done in {total:.1f}s\n{'='*60}")
    print("Try it:")
    print("  python recommenders/chaos_recommender.py salamence --gen 3 --tier ou")
    print("  python recommenders/chaos_recommender.py rayquaza  --gen 6 --tier anythinggoes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
