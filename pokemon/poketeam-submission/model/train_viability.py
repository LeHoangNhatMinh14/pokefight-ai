"""
Train a LightGBM team-viability classifier.

Positives:
    For each (gen, tier), we sample 6-Pokemon teams by Gibbs-style chaining over
    the chaos teammate distribution: pick a seed by usage; then repeatedly pick
    the next member proportional to its teammate weights with one of the current
    team members. This reproduces the statistical structure of high-rated real
    teams better than just listing top-usage mons.

Negatives:
    For each positive we generate ~1 random team from the same metagame's
    Pokemon pool, plus a few "hard" negatives that share the anchor but pick
    teammates with the *lowest* teammate weights (so the model learns the
    teammate-co-occurrence signal, not just popularity).

The model is small (200 trees, depth 5) — the goal is a reranker on top of the
beam search, not a standalone oracle.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.features import FEATURE_NAMES, FeatureBundle, team_features


# ---------- team sampling ---------------------------------------------------------

def _weighted_choice(rng: random.Random, names: list[str], weights: list[float]) -> str:
    total = sum(weights)
    if total <= 0:
        return rng.choice(names)
    r = rng.random() * total
    acc = 0.0
    for n, w in zip(names, weights):
        acc += w
        if acc >= r:
            return n
    return names[-1]


def sample_positive(
    bundle: FeatureBundle, gen: int, tier: str, size: int, rng: random.Random,
) -> list[str]:
    """Greedy chaos sample: seed by usage^1, expand by sum of teammate weights."""
    pool = bundle.names_in(gen, tier)
    if len(pool) < size:
        size = len(pool)
    if size == 0:
        return []

    # Seed
    usages = []
    for n in pool:
        r = bundle.meta_row(gen, tier, n)
        usages.append(float((r or {}).get("usage", 0.0) or 0.0))
    team = [_weighted_choice(rng, pool, usages)]

    while len(team) < size:
        candidates = [n for n in pool if n not in team]
        if not candidates:
            break
        weights = []
        for c in candidates:
            w = 0.0
            for member in team:
                w += bundle.teammate_weight(gen, tier, member, c)
                w += bundle.teammate_weight(gen, tier, c, member)
            weights.append(w + 1e-3)  # small epsilon so we never fully exclude
        team.append(_weighted_choice(rng, candidates, weights))
    return team


def sample_random_negative(
    bundle: FeatureBundle, gen: int, tier: str, size: int, rng: random.Random,
) -> list[str]:
    pool = bundle.names_in(gen, tier)
    if len(pool) < size:
        size = len(pool)
    return rng.sample(pool, size)


def sample_hard_negative(
    bundle: FeatureBundle, gen: int, tier: str, anchor: str, size: int,
    rng: random.Random,
) -> list[str]:
    """Anchor + the lowest-teammate-weight choices in the same tier."""
    pool = bundle.names_in(gen, tier)
    candidates = [n for n in pool if n != anchor]
    scored = []
    for c in candidates:
        w = bundle.teammate_weight(gen, tier, anchor, c)
        w += bundle.teammate_weight(gen, tier, c, anchor)
        scored.append((w, c))
    scored.sort()
    # Take from the bottom but with a little randomness
    bottom = [c for _, c in scored[: max(size * 3, size + 2)]]
    rng.shuffle(bottom)
    return [anchor] + bottom[: size - 1]


# ---------- training -------------------------------------------------------------

def build_training_set(
    bundle: FeatureBundle,
    *, team_size: int = 6, n_positive_per_tier: int = 500, seed: int = 0,
):
    rng = random.Random(seed)
    X, y = [], []
    meta = bundle.meta[["gen", "tier"]].drop_duplicates().values.tolist()

    for gen, tier in meta:
        pool = bundle.names_in(gen, tier)
        if len(pool) < 3:
            continue
        size = min(team_size, len(pool))
        n_pos = n_positive_per_tier
        for _ in range(n_pos):
            pos = sample_positive(bundle, gen, tier, size, rng)
            X.append(team_features(bundle, gen, tier, pos)); y.append(1)
            neg = sample_random_negative(bundle, gen, tier, size, rng)
            X.append(team_features(bundle, gen, tier, neg)); y.append(0)
            anchor = pos[0] if pos else rng.choice(pool)
            hard = sample_hard_negative(bundle, gen, tier, anchor, size, rng)
            X.append(team_features(bundle, gen, tier, hard)); y.append(0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(out_dir: Path, bundle: FeatureBundle, *, seed: int = 0):
    print("Building training set...")
    X, y = build_training_set(bundle, seed=seed)
    print(f"  X: {X.shape}, positives: {int(y.sum())}, negatives: {int((1-y).sum())}")
    if len(np.unique(y)) < 2:
        raise SystemExit("Not enough variety in fixtures to train (need >= 2 classes).")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.85)
    tr, va = idx[:split], idx[split:]

    train_ds = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATURE_NAMES)
    valid_ds = lgb.Dataset(X[va], label=y[va], feature_name=FEATURE_NAMES, reference=train_ds)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
    }
    model = lgb.train(
        params, train_ds, num_boost_round=400,
        valid_sets=[train_ds, valid_ds], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
    )

    p_va = model.predict(X[va])
    auc = roc_auc_score(y[va], p_va) if len(np.unique(y[va])) >= 2 else float("nan")
    print(f"\nValidation AUC: {auc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "viability.lgb"))
    (out_dir / "viability_features.json").write_text(json.dumps(FEATURE_NAMES))
    print(f"Saved -> {out_dir/'viability.lgb'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=str, default="data/processed/pokemon_meta.parquet")
    parser.add_argument("--teammates", type=str, default="data/processed/teammates.parquet")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/embeddings.parquet")
    parser.add_argument("--out", type=str, default="model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bundle = FeatureBundle(
        meta=pd.read_parquet(args.meta),
        teammates=pd.read_parquet(args.teammates),
        embeddings=pd.read_parquet(args.embeddings),
    )
    train(Path(args.out), bundle, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
