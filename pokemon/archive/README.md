# Archive

Files in this folder are no longer used by the running app but are kept around
for reference. None of these are imported by `app.py` or either recommender.

- `showdown_data.py` — older Smogon scraper, replaced by
  `data_pipeline/fetch_chaos.py`.
- `stats_index.html` — saved Smogon directory listing (just an `index.html`
  dump from `smogon.com/stats/2026-04/`). Not the Flask template.
- `smogon_gen9ou_data.csv` — Gen 9 OU usage CSV, not used by the Gen 1-6
  pipeline.
- `pokemon_data_with_type_effectiveness.csv` — older derived CSV; the current
  app reads `data/processed/pokemon_final.csv`. Regenerable from the notebook
  in `notebooks/pokemon_data.ipynb`.
- `data/raw/gen9ou-1825.txt` — leftover Gen 9 OU raw text dump, not consumed
  by the chaos pipeline (which fetches JSON, not txt).
