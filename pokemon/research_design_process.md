# Building a Pokémon Team Recommender — Research & Design Process

When I started this project, I genuinely didn't know what I was doing. My knowledge of AI and machine learning was close to zero, and a lot of the early groundwork was laid by asking Claude and ChatGPT to explain concepts to me — what a recommendation system even is, how APIs work, what a pipeline looks like. That process of reading, asking, and slowly understanding is actually where the Library research began, even if I didn't recognise it as research at the time.

The rest of this writeup is organised by the five DOT strategies (Library, Field, Lab, Showroom, Workshop) and within each one I name the specific canonical methods I applied, what I actually did for that method, and where it shows up in the codebase.

---

## Library — Building on Existing Knowledge

### Available product analysis

Before writing any logic, I surveyed the existing landscape of Pokémon team-building tools to understand what was already solved and where a new project could fit. The competitive scene already has several established builders:

- **Pokémon Showdown Teambuilder** — the canonical manual builder, baked into the official Showdown battle simulator. It gives you perfect stats, format options, and instant testing, but it's a blank slate — you have to know what you're building.
- **Pikalytics Team Builder** — data-driven meta tool that imports/exports Showdown teams, with suggested sets, EV spreads, moves, items, abilities, and tournament-ready meta picks for the current VGC format.
- **My Pokémon Team** — broad-coverage builder supporting X&Y through Scarlet/Violet, with team checklists and search filters.
- **AI Team Builder**, **Poké Team Builder**, **BattleWise AI**, **Sueter Team Builder** — newer AI-assisted builders that use LLM chat to suggest teams, EV spreads, and offensive/defensive coverage from a natural-language brief.

The conclusion of that survey shaped my own scope: the manual builders (Showdown) and the meta data viewers (Pikalytics) already do their jobs well, and the LLM chat tools cover the conversational angle. What I didn't see was a builder that takes the Smogon chaos JSON's actual *teammate co-occurrence* data and uses it as a search heuristic — i.e. a tool that asks "what do players who use this Pokémon actually pair it with on the ladder?" and answers that question quantitatively across many gens and tiers at once. That gap became the project's niche.

### Competitive analysis

Building on the product survey, the competitive analysis was about *how* each tool makes its recommendations and where my approach differs:

| Tool                  | Recommendation mechanism                                | Data source                                |
| --------------------- | ------------------------------------------------------- | ------------------------------------------ |
| Showdown Teambuilder  | None — manual                                           | n/a                                        |
| Pikalytics            | Top-usage display, manual selection                     | Smogon usage stats (current VGC)           |
| My Pokémon Team       | Filter + checklist                                      | Game data only                             |
| AI Team Builder etc.  | LLM prompt → natural-language team                      | Whatever the LLM was trained on            |
| **This project**      | Beam search over teammate weights, LightGBM reranker    | Smogon chaos JSON, ~140 metagames, Gens 1–6 |

The key differentiator is the search heuristic itself. Pikalytics shows you the top usage but doesn't *build* a team for you; LLM tools build a team but the reasoning is opaque and not grounded in measured ladder behaviour; my project uses the actual `Teammates` field from the chaos JSON as a hard signal during beam search, with the LightGBM reranker scoring the resulting candidate teams for global viability. That's a different category of tool — closer to a learning-to-rank pipeline than a generator.

### Best good and bad practices

Two-stage recommendation architecture (candidate generation followed by reranking) is the established production pattern for recommender systems, used by YouTube, Twitter Ads, and most modern recsys deployments. The pattern is to use a fast, coarse first stage to narrow millions of items down to a manageable shortlist, then apply a slower, more expensive ranker to that shortlist with richer features. The two stages have different optimisation targets — retrieval optimises for recall of the top-N, while the reranker optimises for a precision-oriented metric like NDCG.

I applied this directly. Beam search over the chaos teammate weights is the candidate-generation stage (fast, broad, coarse), and the LightGBM team-viability classifier is the reranker (slower, evaluating each candidate team holistically). The README captures the reasoning:

```text
- The chaos JSON's `Teammates` field already encodes "who plays well with whom"
  from millions of real high-rated ladder games. A pure-ML model that ignored
  it would be relearning what is already a perfect ground-truth signal, so we
  use it directly as the search heuristic.
- The viability LightGBM model is only a *reranker* on candidate teams — it
  catches global team-level issues (weakness overlap, role imbalance) that
  greedy teammate search misses.
```

In code, the reranker only ever fires *after* the beam-search step has produced candidate teams:

```python
# Rerank by viability model
scored: list[tuple[list[str], float]] = []
for team, _ in new_beams:
    feats = team_features(self.bundle, int(gen), str(tier), team)
    if self.model is not None:
        p = float(self.model.predict(feats.reshape(1, -1))[0])
    else:
        # Fallback: just use the pairwise teammate weight sum
        p = float(feats[FEATURE_NAMES.index(
            "pairwise_teammate_logweight_mean")]) / 12.0
    scored.append((team, p))
```

The other piece of best-practices research was on the *move recommendation* side. The `TYPE_TO_MOVES` table codifies competitive community knowledge about which moves are genuinely strong per type — the same knowledge that informs Smogon set guides and tier list discussions:

```python
TYPE_TO_MOVES: dict[str, list[dict]] = {
    "fire":     [{"move": "flamethrower",  "power":  90, "accuracy": 100},
                 {"move": "fire-blast",    "power": 110, "accuracy":  85},
                 {"move": "flare-blitz",   "power": 120, "accuracy": 100},
                 {"move": "heat-wave",     "power":  95, "accuracy":  90}],
    "electric": [{"move": "thunderbolt",   "power":  90, "accuracy": 100},
                 {"move": "thunder",       "power": 110, "accuracy":  70},
                 {"move": "wild-charge",   "power":  90, "accuracy": 100},
                 {"move": "volt-switch",   "power":  70, "accuracy": 100}],
    # ... 16 more types, four strong attacking moves each
}
```

### Design pattern research

Three established design patterns sit underneath the project:

**Matrix factorisation for collaborative filtering.** SVD is one of the most common factorisation techniques in recommender systems — it decomposes a user-item (here, Pokémon-Pokémon) matrix into latent feature vectors that capture how items relate. I applied this in `data_pipeline/build_embeddings.py` by building a teammate co-occurrence matrix per metagame and running `TruncatedSVD` to get a 32-dimensional embedding per Pokémon. Distance in this space approximates "Pokémon that play similar roles on similar teams":

```python
# build_embeddings.py — for each (gen, tier) metagame
for row in team_df.itertuples(index=False):
    i, j = idx[row.a], idx[row.b]
    w = math.log1p(max(0.0, float(row.weight)))
    M[i, j] = max(M[i, j], w)
    M[j, i] = max(M[j, i], w)

# Row-normalize so each Pokemon has unit "outgoing weight" (so embeddings
# capture *who you pair with* not *how popular you are*).
norms = np.linalg.norm(M, axis=1, keepdims=True)
norms[norms == 0] = 1.0
M = M / norms

svd = TruncatedSVD(n_components=k, random_state=0)
emb = svd.fit_transform(M)
```

**Learning-to-rank with LightGBM.** LightGBM ships a dedicated `LGBMRanker` and is widely used for reranking in information retrieval and recommendation. Its leaf-wise tree growth and efficient training make it the default GBDT for production reranking workloads. My project uses LightGBM as a binary classifier rather than a LambdaRank objective because positive/negative team-level labels were easier to construct than per-team relevance grades — but the underlying pattern (small GBDT used to refine a search shortlist) is the same.

**Beam search.** Beam search is a sequence-generation pattern: at each step you keep the top-K partial sequences instead of greedily expanding only the best one. I applied it to incrementally build a 6-mon team, scoring partial teams by summed teammate weight and a viability rerank. The width of 8 is small enough to be fast, wide enough to escape the local maxima a pure greedy expansion would fall into.

### Community research

The Smogon community is itself part of the research input. The chaos JSON format is documented in the official Smogon University Usage Statistics Discussion Thread on the Smogon forums, and the file format itself is documented at the `pkmn/stats` project. The teammate metric is `P(X|Y) - P(X)` — i.e. how much more likely a teammate is than its base usage would predict — which is exactly the signal my beam search exploits. Open-source parsers like `smogon-usage-parser` and `SmogonUsageParser` on GitHub demonstrate how the community has already standardised access to this data.

### Literature study

A directly relevant prior work is Lukas Schaub's "Play Pokémon like a Data Scientist — Part 4: Probabilistic Team Building" on Medium, which approaches the same problem from a probabilistic-modelling angle. Combined with general recsys references on two-stage architectures (Towards AI, ApplyingML), the literature confirmed that the candidate-generation + reranker split was the right shape for the project rather than a single end-to-end model.

### Not applied: SWOT analysis, Expert interview

For honesty: I did not perform a structured SWOT, and I did not interview a domain expert. Both would strengthen the writeup if added later.

---

## Field — Real Player Behaviour as Data

### Document analysis

The chaos JSON had to be understood before it could be used. Each metagame file contains usage frequencies, moveset distributions, item/ability/spread distributions, and the all-important teammate co-occurrence weights. `data_pipeline/build_dataset.py` parses these into three tidy tables — `pokemon_meta`, `movesets`, `teammates` — that the rest of the pipeline consumes. That parsing step is document analysis in the DOT sense: figuring out the shape of an external document and restructuring it into a usable schema.

### Exploratory data analysis (ML)

Before the schema was finalised I had to look at the raw data — which gens and tiers had enough samples, what the weight distributions looked like, which Pokémon had moveset data and which didn't. The schema in the README reflects what survived that exploration:

```text
data/processed/
  pokemon_meta.parquet      # (gen, tier, name, usage, raw_count, viability, stats)
  movesets.parquet          # long: (gen, tier, name, kind in {move,item,ability,spread,tera,happiness}, value, weight)
  teammates.parquet         # long: (gen, tier, a, b, weight)
```

The "long" format with `kind` as a column came from EDA — moves, items, abilities and spreads all share the same `(value, weight)` shape, so storing them in a single long table is cleaner than four separate tables.

### Domain modelling

The three-table schema *is* the domain model. A `(gen, tier)` is a metagame; each metagame contains Pokémon with usage and stats (`pokemon_meta`), a set of weighted move/item/ability/spread distributions per Pokémon (`movesets`), and a weighted directed graph of teammate co-occurrence (`teammates`). Every downstream component — the embeddings build, the viability features, the beam search — is expressed in terms of this model.

### Field gaps

I did not run interviews, surveys, observation, focus groups, or stakeholder analysis. The Field research is entirely behavioural-data-driven, not interview-driven. This is a legitimate gap if the assessment expects user-facing Field work.

---

## Lab — Testing and Validation

### Unit test

`tests/test_pipeline.py` is the lab artefact for this. It runs the full pipeline on shipped fixtures (no network required) and asserts that known competitive knowledge survives the pipeline:

```python
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
```

Strictly speaking this is closer to a *system test* than a per-function unit test, but the DOT canonical list doesn't separate "smoke test" so I'm filing it under Unit test as the closest fit.

### Model evaluation (ML) and Model validation (ML)

`model/train_viability.py` holds out 15% of the training data and reports ROC-AUC on it before saving the model:

```python
rng = np.random.RandomState(seed)
idx = rng.permutation(len(X))
split = int(len(X) * 0.85)
tr, va = idx[:split], idx[split:]

# ... train model ...

p_va = model.predict(X[va])
auc = roc_auc_score(y[va], p_va) if len(np.unique(y[va])) >= 2 else float("nan")
print(f"\nValidation AUC: {auc:.4f}")
```

ROC-AUC was the right metric here because the task is binary (viable team / not viable) and the class distribution is balanced by construction (one positive plus two negatives per anchor). The split is single-fold rather than k-fold; that's a known weakness if you want tight confidence intervals.

### Non-functional test

The performance investigation around the Pikachu slowdown is a non-functional test in the DOT sense: not "does it produce the right output" but "does it produce it fast enough." I timed `recommend_teammates` and `add_movesets_to_team`, profiled with `cProfile`, traced the latency to the live PokéAPI move-detail loop (~100 sequential HTTP calls per Pokémon, ~600 per team), and replaced it with an offline lookup. Before:

```python
# BEFORE: ~100 HTTP requests per Pokemon, ~600 per team
def get_pokemon_moves(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    data = fetch_json(url)
    moves = []
    for move_entry in data["moves"]:
        move_data = fetch_json(move_entry["move"]["url"])  # one call per move
        # ...filter and collect...
    return tuple(moves)
```

After:

```python
# AFTER: lookup against TYPE_TO_MOVES, zero network
def recommend_moves(pokemon_name, df, max_moves=4):
    pokemon_row = df[df["name"].str.lower() == pokemon_name.lower()].iloc[0]
    pokemon_types = [pokemon_row["type1"].lower()]
    if has_type2(pokemon_row.get("type2")):
        pokemon_types.append(pokemon_row["type2"].lower())

    selected, used_types = [], set()
    for typ in pokemon_types:
        _take(typ, selected, used_types)
    for typ in _COVERAGE_TYPE_ORDER:
        if len(selected) >= max_moves: break
        if typ in used_types: continue
        _take(typ, selected, used_types)
    return pd.DataFrame(selected)
```

Re-measuring after the change confirmed the improvement. Isolate, measure, change, measure again — the canonical non-functional-test loop.

### Data analytics

The chaos JSON itself is a body of measured behavioural data. Pipeline-level analytics on it — usage distributions per tier, top-usage tables, which tiers have enough samples for embeddings to be meaningful — informed every downstream design decision. `data_pipeline/build_dataset.py`'s aggregations and the fixture inspection that drives `test_pipeline.py`'s assertion thresholds (≥20 meta rows, ≥200 moveset rows, ≥100 teammate rows) are both data-analytics artefacts.

### Lab gaps

No A/B test was run (I don't have two teams of users to split). No security test, hardware validation, computer simulation, or inferential statistics in the strict sense.

---

## Workshop — Building, Designing, Iterating

### Prototyping

The Flask app (`app.py` + `templates/index.html` + `static/style.css`) is the central prototype artefact. It's the surface where the pipeline becomes a usable product, and the place where most of the iteration happened. Two recent iteration moments:

The chaos→synergy fallback originally surfaced as a confusing red error. After using the form a few times I realised it should be a friendly informational notice instead:

```python
# app.py — chaos request failed (no Smogon data), fall back to synergy mode
except ValueError:
    notice = (
        f"'{pokemon_name.title()}' has no Smogon usage data — "
        "showing a type-synergy team instead. The metagame dropdown "
        "doesn't apply here."
    )
    mode = "synergy"
```

Typing a full Pokémon name into a plain text input was clumsy, so the input now drives a live filtered dropdown:

```javascript
function filterNames(query) {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const prefix = [], substring = [];
    for (const name of allNames) {
        const lower = name.toLowerCase();
        if (lower.startsWith(q)) prefix.push(name);
        else if (lower.includes(q)) substring.push(name);
        if (prefix.length >= MAX_RESULTS) break;
    }
    return prefix.concat(substring).slice(0, MAX_RESULTS);
}
```

### Decomposition

The system is decomposed into four pipeline stages with explicit dependencies:

```text
fetch_chaos.py        # download chaos JSON for every (gen, tier) we can find
   |
   v
build_dataset.py      # parse JSON -> pokemon_meta / movesets / teammates parquet
   |
   v
build_embeddings.py   # SVD over teammate co-occurrence -> embeddings.parquet
   |
   v
model/train_viability.py   # LightGBM team-viability classifier -> viability.lgb
   |
   v
chaos_recommender.py  # beam search + reranker + slot fill
   |
   v
app.py                # Flask UI
```

Each stage is independently runnable and produces a typed artefact (a parquet or model file) consumed by the next. That's decomposition in the DOT sense — chosen to make the system inspectable at each stage.

### IT architecture sketching

The diagram above (in the README) is the architecture sketch. It existed before any code did. It set the contract — "build_dataset produces three parquets; build_embeddings consumes the teammates parquet and produces a fourth; train_viability consumes all four; chaos_recommender loads them at runtime" — and everything else was implemented to that contract.

### Code review

Code review happened informally but repeatedly during the iterative build. The autocomplete refactor, the disk-cache layer for PokéAPI, the transition from live PokéAPI calls to the offline TYPE_TO_MOVES table, and the chaos→synergy fallback notice were all moments where existing code was re-examined and improved rather than left as-is. Each one was prompted by hitting a friction point, looking at what was already there, and deciding what to change.

### Root cause analysis

The Pikachu slowdown was a textbook root-cause-analysis exercise: a user-facing symptom ("selecting Pikachu takes forever") whose first plausible cause was the beam search, then turned out to be the chaos→synergy fallback, which in turn was caused by the PokéAPI move-detail loop being called ~600 times per team build. Each layer was checked off before moving to the next, and the actual fix happened at the deepest layer (replace the live API with the offline table), not the topmost (don't optimise the beam search — it wasn't the bottleneck).

### Algorithm design

Beam search with a beam width of 8 and a candidate pool of 20 was chosen as the balance between coverage and computational cost. For generating training data, **Gibbs-style sampling** was used to build positive examples — repeatedly picking the next team member proportional to teammate weight, which mirrors how a knowledgeable player constructs a team incrementally:

```python
def sample_positive(bundle, gen, tier, size, rng):
    """Greedy chaos sample: seed by usage^1, expand by sum of teammate weights."""
    pool = bundle.names_in(gen, tier)
    usages = [float((bundle.meta_row(gen, tier, n) or {}).get("usage", 0.0) or 0.0)
              for n in pool]
    team = [_weighted_choice(rng, pool, usages)]
    while len(team) < size:
        candidates = [n for n in pool if n not in team]
        weights = []
        for c in candidates:
            w = 0.0
            for member in team:
                w += bundle.teammate_weight(gen, tier, member, c)
                w += bundle.teammate_weight(gen, tier, c, member)
            weights.append(w + 1e-3)
        team.append(_weighted_choice(rng, candidates, weights))
    return team
```

Hard negatives pair an anchor with its *lowest*-weight teammates so the model has to learn the teammate-co-occurrence signal rather than just popularity:

```python
def sample_hard_negative(bundle, gen, tier, anchor, size, rng):
    """Anchor + the lowest-teammate-weight choices in the same tier."""
    pool = bundle.names_in(gen, tier)
    scored = []
    for c in (n for n in pool if n != anchor):
        w  = bundle.teammate_weight(gen, tier, anchor, c)
        w += bundle.teammate_weight(gen, tier, c, anchor)
        scored.append((w, c))
    scored.sort()
    bottom = [c for _, c in scored[: max(size * 3, size + 2)]]
    rng.shuffle(bottom)
    return [anchor] + bottom[: size - 1]
```

### Gap analysis

Documenting the gaps was itself a workshop method — listing what's missing (no Showroom artefacts, no A/B test, no expert interview) is gap analysis. The "Showroom" section below is the explicit output of that exercise.

### Workshop gaps

No hackathon, no co-creation with another developer, no formal multi-criteria decision matrix, no business case exploration, no requirements prioritisation document. The build was solo and iterative rather than facilitated.

---

## Showroom — The Gap

The Showroom strategy is genuinely underdeveloped. None of the seven Showroom methods on the canonical list — Peer review, Product review, Pitch, Benchmark test, Ethical check, Guideline conformity analysis, Static program analysis — were applied. The Flask UI is the natural artefact to take into a Showroom session: it could be presented to a competitive player for a **Product review**, pitted against Pikalytics or BattleWise AI for a **Benchmark test**, or checked against accessibility/UI guidelines for a **Guideline conformity analysis**. The infrastructure for any of these exists; the sessions themselves don't. That's the most actionable item on the remaining-work list.

---

## Reflection

Looking back, the research didn't follow a clean sequence. Library, Lab, and Workshop were happening at the same time, often in response to each other. I'd read something, try to implement it, test whether it worked, and go back to reading when it didn't. The chaos JSON was the single biggest turning point — moving from synthetic stat-based logic to real competitive data changed what the system was capable of and changed what I understood about how recommendation systems can work. The two-stage candidate-generation-plus-reranker architecture, which I now know is the production-standard pattern for recsys, emerged from following the data rather than from picking a design upfront — but the fact that it converged on the standard pattern is itself reassuring.

The personal knowledge gap I started with never fully closed, but it got a lot smaller, and most of that closure came from building something real and having to confront what wasn't working.

---

## Sources

Library research drew on these external references:

- [Pokémon Showdown Teambuilder](https://play.pokemonshowdown.com/teambuilder) — the canonical manual builder
- [Pikalytics Team Builder](https://www.pikalytics.com/team) — data-driven meta tool with VGC stats
- [My Pokémon Team](https://mypokemonteam.com/) — multi-gen team builder
- [AI Team Builder](https://aiteambuilder.com/) — LLM-driven team suggestions
- [Poké Team Builder](https://poketeambuilder.app/) — AI builder with Showdown export
- [BattleWise AI](https://www.battlewiseai.com/team-builder) — current-meta AI builder
- [Sueter Team Builder](https://sueter-team-builder.vercel.app/) — AI builder with coverage analysis
- [Play Pokemon like a Data Scientist — Part 4: Probabilistic Team Building (Lukas Schaub)](https://medium.com/@lukasschaub/play-pokemon-like-a-data-scientist-part-4-probabilistic-team-building-5e4b9fd9a3bf) — directly comparable academic-style writeup
- [Official Smogon University Usage Statistics Discussion Thread (mk.2)](https://www.smogon.com/forums/threads/official-smogon-university-usage-statistics-discussion-thread-mk-2.3508502/) — chaos JSON format discussion
- [pkmn/stats OUTPUT.md](https://github.com/pkmn/stats/blob/main/stats/OUTPUT.md) — formal chaos JSON format documentation
- [smogon-usage-parser (GitHub)](https://github.com/GriffinLedingham/smogon-usage-parser) — community parser reference
- [Recommender Systems: From Theory to Production (Towards AI)](https://pub.towardsai.net/recommender-systems-from-theory-to-production-0f92bd85dcff) — two-stage recsys architecture
- [One-Stop Guide for Production Recommendation Systems (Medium)](https://medium.com/@zaiinn440/one-stop-guide-for-production-recommendation-systems-9491f68d92e3) — candidate gen + reranking patterns
- [Real-time Retrieval for Recommendations (ApplyingML)](https://applyingml.com/resources/real-time-recommendations/) — retrieval/reranking optimisation targets
- [Matrix Factorization: The Bedrock of Collaborative Filtering Recommendations (Shaped)](https://www.shaped.ai/blog/matrix-factorization-the-bedrock-of-collaborative-filtering-recommendations) — SVD in recommender systems
- [Leveraging LightGBM Ranker for Efficient Large-Scale News Recommendation Systems (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3687151.3687156) — LightGBM as reranker
- [Learning-to-rank with LightGBM (Medium)](https://tamaracucumides.medium.com/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574) — LTR with LightGBM in practice
