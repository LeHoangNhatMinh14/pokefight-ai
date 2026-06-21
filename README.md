# PokéTeam AI

A data-driven competitive Pokémon team recommender. Give it your favourite Pokémon, and it builds you a six-member team using real Smogon ladder usage data and a trained machine learning model.

## What it does

Given one Pokémon as input, PokéTeam AI returns a complete six-Pokémon team built around it for competitive play. The system uses a **search-plus-rerank architecture**:

1. **Beam search** proposes candidate teams using Smogon's published teammate co-occurrence weights as a heuristic.
2. **A trained LightGBM classifier** (~200 trees, depth 5) scores each candidate based on 22 team-level features.
3. **The highest-scoring candidate** wins and gets enriched with moves, items, abilities, and EV spreads sampled from chaos distributions.

If chaos data is unavailable for the chosen Pokémon, the app silently falls back to a rules-based synergy recommender so the user always gets a team.

## Tech stack

- **Python 3.10+**
- **Flask** — web application
- **LightGBM** — the trained classifier (gradient-boosted decision trees)
- **scikit-learn** — SVD embeddings, evaluation metrics, baseline model comparisons
- **pandas + NumPy + PyArrow** — data manipulation and parquet storage
- **requests-cache** — SQLite-backed HTTP cache for PokéAPI fair-use compliance

## Installation

```bash
git clone https://github.com/LeHoangNhatMinh14/pokefight-ai.git
cd pokefight-ai/pokemon
pip install -r requirements.txt
```

The repo expects Python 3.10 or later. A conda environment is recommended but not required.

## Quick start

Build all data artefacts and train the model from scratch (one-time, takes a few minutes):

```bash
python tools/rebuild_chaos.py
```

Pre-warm the PokéAPI cache so the first synergy build isn't slow (optional but recommended):

```bash
python tools/prewarm_pokeapi.py
```

Precompute synergy teams for instant lookups (optional, takes ~1-2 hours):

```bash
python tools/precompute_synergy_teams.py
```

Run the web app:

```bash
python app.py
```

Then open `http://localhost:5000` in a browser.

## Running the tests

```bash
python tests/test_pipeline.py
```

Rebuilds the pipeline from fixture data, trains a fresh model, and asserts that known anchors (Salamence, Snorlax, Tauros) return their textbook competitive partners.

## Project structure

```
pokemon/
├── app.py                       # Flask entry point
├── requirements.txt
├── README.md                    # this file
├── README_chaos_pipeline.md     # detailed pipeline docs
├── logging_config.py            # logging setup
│
├── data/
│   ├── raw/chaos/               # raw Smogon JSON (regenerable, gitignored)
│   ├── processed/               # parquet tables
│   ├── embeddings/              # SVD embeddings
│   ├── fixtures/                # small test data for the smoke test
│   └── cache/                   # PokeAPI SQLite cache (regenerable, gitignored)
│
├── data_pipeline/
│   ├── fetch_chaos.py           # downloads Smogon chaos JSON
│   ├── build_dataset.py         # parses JSON to parquets
│   └── build_embeddings.py      # SVD over teammate matrix
│
├── model/
│   ├── train_viability.py       # trains the LightGBM model
│   ├── features.py              # the 22 team-level features
│   ├── viability.lgb            # the trained model
│   └── viability_features.json  # feature name registry
│
├── recommenders/
│   ├── chaos_recommender.py     # beam search + reranker
│   ├── recommender.py           # synergy fallback recommender
│   └── synergy_scorer.py        # type/role scoring
│
├── tools/
│   ├── rebuild_chaos.py         # runs the whole pipeline end-to-end
│   ├── prewarm_pokeapi.py       # warms the PokeAPI cache
│   └── precompute_synergy_teams.py  # bakes synergy answers to parquet
│
├── tests/
│   └── test_pipeline.py         # end-to-end smoke test
│
├── notebooks/
│   └── poketeam_report.ipynb    # main project report (run end-to-end)
│
└── templates/, static/          # Flask UI
```

## How it works in more detail

The main project documentation lives in two places:

- **`notebooks/poketeam_report.ipynb`** — the runnable report. Walks through the data, embeddings, labelling strategy, model training, scoring mechanics (including a real decision tree from the trained model), evaluation (ROC AUC, accuracy, confusion matrix, score distribution), inference demo, and synergy fallback. If you only read one thing, read this.
- **`README_chaos_pipeline.md`** — detailed documentation of the data pipeline stages and how they connect.

## Data sources and licensing

PokéTeam AI uses three external data sources:

- **Smogon chaos JSON** (`smogon.com/stats/`) — publicly published aggregate ladder usage statistics across 14 metagames (Gen 1-6, OU/Ubers/LC/NU/Doubles OU). No personal data. Credit given here and in the project documentation.
- **PokéAPI** (`pokeapi.co`) — free public REST API for Pokémon data. The project complies with their fair-use policy by caching responses locally via a persistent SQLite store.
- **Internal `pokemon_final.csv`** — base table with stats, types, weaknesses, roles, and legendary flags for ~1,500 Pokémon.

Pokémon names, types, and sprites are trademarks of Nintendo and Game Freak. This project is personal and educational — non-commercial use only.

## Limitations to know about

A few things worth flagging:

- **The model's high accuracy (~97.8%) measures statistical resemblance to real high-rated teams, not actual battle outcomes.** Win-rate data is not published by Smogon in a clean format. See the notebook's section 8c and section 11 for the full honest interpretation.
- **The chaos data covers Generations 1-6 only.** No support for newer Pokémon, abilities, or formats.
- **The model inherits the metagame orthodoxy of Smogon's high-rated player base** (rating buckets 1500 / 1630 / 1825). Documented openly in the impact assessment but not mitigated.
- **The Flask app uses the built-in development server with `debug=True`.** Local-only, not safe to expose to a network without adding authentication.

## Acknowledgements

- **Smogon University** for publishing the chaos statistics that this project would be impossible without.
- **PokéAPI maintainers** for the free public API.
- **The HBO-i ICT Research Methods framework** which structured the methodology behind this project.

## A note on AI-assisted documentation

Many of the project's documentation files (the notebook prose, the impact assessment, the model comparison document, this README) were drafted in collaboration with Claude, an AI assistant. The code, the technical decisions, the experimental results, and the reflections are mine; the prose and structure are AI-assisted. This is documented openly across the project's Learning Outcome 6 evidence.

---

Built as part of the *Minor AI for Society* at Fontys, 2025-2026.
