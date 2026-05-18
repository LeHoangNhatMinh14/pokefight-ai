"""
Build a teammate-co-occurrence embedding per (gen, tier, Pokemon).

For each metagame we form a square matrix M[a, b] = log(1 + weight(a paired with b)).
We row-normalize, then run truncated SVD to get a dense vector per Pokemon.

Distance in this space approximates "Pokemon that play similar roles on similar teams"
and inner products approximate teammate compatibility — both useful as features for
the viability model and as a fast similarity index for the recommender.

Output:
    data/embeddings/embeddings.parquet     (gen, tier, name, dim_0..dim_{k-1})
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def build_one(team_df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """team_df is the teammates table for a single (gen, tier)."""
    if team_df.empty:
        return pd.DataFrame()

    names = sorted(set(team_df["a"]) | set(team_df["b"]))
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    M = np.zeros((n, n), dtype=np.float32)

    for row in team_df.itertuples(index=False):
        i, j = idx[row.a], idx[row.b]
        w = math.log1p(max(0.0, float(row.weight)))
        # Symmetrize: chaos teammate weights from a->b and b->a should already
        # agree but we OR them just in case.
        M[i, j] = max(M[i, j], w)
        M[j, i] = max(M[j, i], w)

    # Row-normalize so each Pokemon has unit "outgoing weight" (so embeddings
    # capture *who you pair with* not *how popular you are*).
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms

    k = min(dim, n - 1) if n > 1 else 1
    if k <= 0:
        return pd.DataFrame()

    svd = TruncatedSVD(n_components=k, random_state=0)
    emb = svd.fit_transform(M)  # (n, k)

    # Pad to requested dim with zeros if metagame is too small.
    if k < dim:
        emb = np.hstack([emb, np.zeros((n, dim - k), dtype=np.float32)])

    df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(dim)])
    df.insert(0, "name", names)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=str, default="data/processed/teammates.parquet")
    parser.add_argument("--out", type=str, default="data/embeddings/embeddings.parquet")
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    team_df = pd.read_parquet(args.inp)
    if team_df.empty:
        print("teammates.parquet is empty.")
        return 1

    out_rows = []
    for (gen, tier), part in team_df.groupby(["gen", "tier"]):
        emb = build_one(part, args.dim)
        if emb.empty:
            continue
        emb.insert(0, "tier", tier)
        emb.insert(0, "gen", gen)
        out_rows.append(emb)
        print(f"  gen{gen}{tier}: {len(emb)} pokemon -> {args.dim}-d embeddings")

    if not out_rows:
        print("No embeddings produced.")
        return 1

    result = pd.concat(out_rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    print(f"\nWrote {len(result):,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
