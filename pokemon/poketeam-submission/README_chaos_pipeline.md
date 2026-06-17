# Smogon Chaos Team Builder

A team-recommender that uses Smogon usage stats ("chaos JSON") to build a
competitively viable team around any favorite Pokemon across Gens 1-6, all
tiers (OU, UU, RU, NU, PU, Ubers, LC, Monotype, AG, etc).

## Pipeline

```
fetch_chaos.py        # download chaos JSON for every (gen, tier) we can find
   |
   v
build_dataset.py      # parse JSON -> pokemon_meta.parquet, movesets.parquet, teammates.parquet
   |
   v
build_embeddings.py   # SVD over teammate co-occurrence -> embeddings.parquet
   |
   v
model/train_viability.py   # LightGBM: positives via Gibbs sample of chaos teammates,
                           #           negatives random + hard. Output: model/viability.lgb
   |
   v
chaos_recommender.py  # beam search seeded by favorite, expanded by teammate weights,
                      # reranked by viability model, builds moves/items/ability/spread
                      # from chaos distributions.
   |
   v
app.py                # Flask UI with mode=chaos and metagame picker
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Fetch real Smogon chaos data (~140 metagames, ~2 min)
python data_pipeline/fetch_chaos.py

# 3. Build the dataset, embeddings, and viability model
python data_pipeline/build_dataset.py
python data_pipeline/build_embeddings.py
python -m model.train_viability

# 4. Try it from the CLI
python chaos_recommender.py garchomp --gen 4 --tier ou
python chaos_recommender.py greninja           # auto-detect best metagame

# 5. Run the web app
python app.py
# open http://localhost:5000
```

## Smoke test (no network required)

The repo ships with small fixtures in `data/fixtures/` so the pipeline can be
verified without fetching anything from Smogon:

```bash
python tests/test_pipeline.py
```

## Data layout

```
data/
  fixtures/                   # ships with the repo, used for smoke tests
    gen1ou-1500.json
    gen3ou-1500.json
  raw/chaos/                  # produced by fetch_chaos.py
    2025-04/gen3/ou-1500.json
    ...
  processed/                  # produced by build_dataset.py
    pokemon_final.csv         # base stats / types / weaknesses / role (pre-existing)
    pokemon_meta.parquet      # per (gen, tier, name) usage / viability / stats
    movesets.parquet          # long: (gen, tier, name, kind in {move,item,ability,spread,tera,happiness}, value, weight)
    teammates.parquet         # long: (gen, tier, a, b, weight)
  embeddings/
    embeddings.parquet        # per (gen, tier, name) -> dim_0..dim_31
model/
  viability.lgb               # trained LightGBM team-viability classifier
  viability_features.json     # ordered feature names (must match features.py)
```

## Why this design

- The chaos JSON's `Teammates` field already encodes "who plays well with whom"
  from millions of real high-rated ladder games. A pure-ML model that ignored
  it would be relearning what is already a perfect ground-truth signal, so we
  use it directly as the search heuristic.
- The viability LightGBM model is only a *reranker* on candidate teams — it
  catches global team-level issues (weakness overlap, role imbalance) that
  greedy teammate search misses.
- All names are canonicalized to lowercase compact form on parse so chaos
  data joins cleanly with `pokemon_final.csv`.
- Where chaos data is missing for a pokemon, the Flask app transparently falls
  back to the existing type/role synergy recommender so the user always gets a
  team.
