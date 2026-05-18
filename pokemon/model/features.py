"""
Feature extraction for the team-viability model.

A "team" is a list of (gen, tier, name) tuples (size 6). Given the chaos-derived
dataset + the embedding table + the base stats, we produce a fixed-length feature
vector summarizing the team's competitive viability signals.

The feature names are stable strings (no positional ambiguity), so the same
extractor is used at training time and at inference time inside the recommender.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

ALL_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
]


# ---------- safe parsing helpers -------------------------------------------------

def _parse_listish(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# ---------- bundle that the recommender / trainer share ------------------------

@dataclass
class FeatureBundle:
    meta: pd.DataFrame          # per (gen, tier, name) stats / usage / role / types
    teammates: pd.DataFrame     # long-form (gen, tier, a, b, weight)
    embeddings: pd.DataFrame    # per (gen, tier, name) -> dim_0..dim_{k-1}

    def __post_init__(self) -> None:
        self.meta = self.meta.copy()
        if "weaknesses" in self.meta.columns:
            self.meta["weaknesses"] = self.meta["weaknesses"].apply(_parse_listish)
        # Indices for O(1) lookup
        self._meta_idx = self.meta.set_index(["gen", "tier", "name"], drop=False)
        self._emb_dim_cols = [c for c in self.embeddings.columns if c.startswith("dim_")]
        self._emb_idx = self.embeddings.set_index(["gen", "tier", "name"], drop=False)
        # teammate weight lookup: (gen,tier,a,b) -> weight
        if not self.teammates.empty:
            self._tm_idx = (
                self.teammates.set_index(["gen", "tier", "a", "b"])["weight"].to_dict()
            )
        else:
            self._tm_idx = {}

    # ---- lookups ----
    def meta_row(self, gen: int, tier: str, name: str) -> dict | None:
        key = (gen, tier, name)
        if key in self._meta_idx.index:
            return self._meta_idx.loc[key].to_dict()
        return None

    def embed(self, gen: int, tier: str, name: str) -> np.ndarray | None:
        key = (gen, tier, name)
        if key in self._emb_idx.index:
            return self._emb_idx.loc[key, self._emb_dim_cols].to_numpy(dtype=np.float32)
        return None

    def teammate_weight(self, gen: int, tier: str, a: str, b: str) -> float:
        return float(self._tm_idx.get((gen, tier, a, b), 0.0))

    def names_in(self, gen: int, tier: str) -> list[str]:
        if (gen, tier) in self._meta_idx.index.droplevel(2).unique():
            sub = self._meta_idx.loc[(gen, tier)]
            if isinstance(sub, pd.Series):
                return [sub["name"]]
            return sub["name"].tolist()
        return []


# ---------- feature extraction --------------------------------------------------

FEATURE_NAMES: list[str] = [
    "team_size",
    "avg_usage", "min_usage", "max_usage",
    "avg_viability",
    "pairwise_teammate_logweight_mean", "pairwise_teammate_logweight_min",
    "pairwise_teammate_fraction_nonzero",
    "pairwise_embedding_cos_mean", "pairwise_embedding_cos_min",
    "type_unique_count",
    "weakness_concentration",  # max same-weakness count across team
    "weakness_unique_count",
    "role_unique_count",
    "stat_hp_mean", "stat_hp_std",
    "stat_atk_mean", "stat_spa_mean", "stat_speed_mean",
    "stat_def_mean", "stat_spd_mean",
    "is_legendary_count",
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def team_features(
    bundle: FeatureBundle,
    gen: int,
    tier: str,
    names: Iterable[str],
) -> np.ndarray:
    names = list(names)
    rows = [bundle.meta_row(gen, tier, n) for n in names]
    rows = [r for r in rows if r is not None]
    n = len(rows)

    feats: dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}
    feats["team_size"] = float(n)
    if n == 0:
        return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)

    usages = np.array([float(r.get("usage", 0.0) or 0.0) for r in rows])
    feats["avg_usage"] = float(usages.mean())
    feats["min_usage"] = float(usages.min())
    feats["max_usage"] = float(usages.max())
    feats["avg_viability"] = float(
        np.mean([float(r.get("viability_top", 0.0) or 0.0) for r in rows])
    )

    # Pairwise teammate weights (in log space for numerical sanity)
    pair_logweights = []
    nonzero = 0
    embs = []
    for r in rows:
        e = bundle.embed(gen, tier, r["name"])
        embs.append(e if e is not None else np.zeros(
            len(bundle._emb_dim_cols), dtype=np.float32))
    for i in range(n):
        for j in range(i + 1, n):
            w = bundle.teammate_weight(gen, tier, rows[i]["name"], rows[j]["name"])
            w += bundle.teammate_weight(gen, tier, rows[j]["name"], rows[i]["name"])
            pair_logweights.append(np.log1p(w))
            if w > 0:
                nonzero += 1
    if pair_logweights:
        feats["pairwise_teammate_logweight_mean"] = float(np.mean(pair_logweights))
        feats["pairwise_teammate_logweight_min"] = float(np.min(pair_logweights))
        feats["pairwise_teammate_fraction_nonzero"] = (
            nonzero / len(pair_logweights)
        )

    # Pairwise embedding cosine
    pair_cos = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_cos.append(_cos(embs[i], embs[j]))
    if pair_cos:
        feats["pairwise_embedding_cos_mean"] = float(np.mean(pair_cos))
        feats["pairwise_embedding_cos_min"] = float(np.min(pair_cos))

    # Type diversity
    type_set = set()
    for r in rows:
        if r.get("type1"):
            type_set.add(str(r["type1"]).lower())
        t2 = r.get("type2")
        if t2 and not (isinstance(t2, float) and np.isnan(t2)):
            type_set.add(str(t2).lower())
    feats["type_unique_count"] = float(len(type_set))

    # Weakness concentration
    weak_counts: dict[str, int] = {}
    for r in rows:
        for w in (r.get("weaknesses") or []):
            weak_counts[w] = weak_counts.get(w, 0) + 1
    feats["weakness_concentration"] = float(max(weak_counts.values(), default=0))
    feats["weakness_unique_count"] = float(len(weak_counts))

    # Role diversity
    roles = {str(r.get("role", "")).lower() for r in rows if r.get("role")}
    feats["role_unique_count"] = float(len(roles))

    # Stat means / std
    def _stat(name):
        return np.array([float(r.get(name, 0) or 0) for r in rows], dtype=np.float32)
    feats["stat_hp_mean"] = float(_stat("hp").mean())
    feats["stat_hp_std"] = float(_stat("hp").std())
    feats["stat_atk_mean"] = float(_stat("attack").mean())
    feats["stat_spa_mean"] = float(_stat("sp_attack").mean())
    feats["stat_speed_mean"] = float(_stat("speed").mean())
    feats["stat_def_mean"] = float(_stat("defense").mean())
    feats["stat_spd_mean"] = float(_stat("sp_defense").mean())

    feats["is_legendary_count"] = float(
        sum(1 for r in rows if bool(r.get("is_legendary", False)))
    )

    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
