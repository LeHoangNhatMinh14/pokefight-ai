"""End-to-end smoke test for the chaos pipeline (runs on fixtures, no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(cmd):
    print("\n$ " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit("command failed: " + " ".join(cmd))
    print(res.stdout[-400:])


def main():
    repo = Path(__file__).resolve().parents[1]
    run([sys.executable, "data_pipeline/build_dataset.py"])
    meta = pd.read_parquet(repo / "data/processed/pokemon_meta.parquet")
    moves = pd.read_parquet(repo / "data/processed/movesets.parquet")
    teams = pd.read_parquet(repo / "data/processed/teammates.parquet")
    assert len(meta) >= 20
    assert len(moves) >= 200
    assert len(teams) >= 100

    run([sys.executable, "data_pipeline/build_embeddings.py", "--dim", "8"])
    emb = pd.read_parquet(repo / "data/embeddings/embeddings.parquet")
    assert len(emb) >= 20

    run([sys.executable, "-m", "model.train_viability"])
    assert (repo / "model/viability.lgb").exists()

    from chaos_recommender import ChaosRecommender
    rec = ChaosRecommender.load()
    cases = [
        ("salamence", {"tyranitar", "blissey", "metagross"}),
        ("snorlax",   {"tauros", "chansey"}),
        ("tauros",    {"snorlax", "chansey"}),
    ]
    for fav, expected in cases:
        team = rec.build_team(fav)
        names = {m.name for m in team.members}
        assert fav in names, "anchor missing for " + fav + ": " + str(names)
        assert (names & expected), (
            "no expected teammates for " + fav + ": got " + str(names)
        )
        for slot in team.members:
            assert slot.moves, slot.name + " has no moves"
        print("  ok: " + fav + " -> " + ", ".join(sorted(names)))
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
