"""Pokemon team recommenders.

This package groups the two recommender backends used by the Flask app:

- ``recommender``: type / role synergy recommender (no Smogon data needed).
- ``chaos_recommender``: Smogon "chaos" usage-data recommender (Gen 1-6, all
  tiers) with a LightGBM viability reranker.
- ``synergy_scorer``: shared helper used by the synergy recommender.
"""
