import ast
import logging
from functools import lru_cache
from pathlib import Path
import pandas as pd
import requests
from typing import List, Dict, Optional
from .synergy_scorer import PokemonSynergyScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent disk cache for PokeAPI calls.
#
# PokeAPI returns essentially static data (move power/accuracy/type don't
# change), so we cache responses to a local SQLite file. This makes synergy
# mode go from "1000+ HTTP calls on first run" to "0 HTTP calls on every run
# after the first" -- the cache survives Flask restarts.
#
# Falls back gracefully to plain `requests` if `requests_cache` isn't
# installed yet (e.g. before `pip install -r requirements.txt` has been run).
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_CACHE_PATH = _CACHE_DIR / "pokeapi"

try:
    import requests_cache  # type: ignore

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _http = requests_cache.CachedSession(
        cache_name=str(_CACHE_PATH),
        backend="sqlite",
        # PokeAPI data is effectively immutable; never expire.
        expire_after=None,
        # Only cache successful responses.
        allowable_codes=(200,),
        # Limit to PokeAPI to avoid accidentally caching anything else.
        allowable_methods=("GET",),
    )
    logger.info("PokeAPI disk cache enabled at %s.sqlite", _CACHE_PATH)
except ImportError:
    _http = requests.Session()
    logger.warning(
        "requests_cache not installed -- PokeAPI calls will not persist "
        "between restarts. Run `pip install -r requirements.txt` to enable "
        "the disk cache."
    )


def parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def has_type2(value):
    return value is not None and not pd.isna(value) and value != ""


def load_data():
    df = pd.read_csv("data/processed/pokemon_final.csv")

    if "weaknesses" in df.columns:
        df["weaknesses"] = df["weaknesses"].apply(parse_list)

    return df


# ---------------------------------------------------------------------------
# Precomputed synergy teams.
#
# `tools/precompute_synergy_teams.py` bakes the synergy recommender's output
# for every (pokemon, playstyle, max_legendaries) combo into a parquet file.
# When that file exists, the Flask app can return a team in ~1ms instead of
# running the ~3.6s lookahead loop. Missing combos fall back to live compute.
# ---------------------------------------------------------------------------

_PRECOMP_PATH = Path("data/processed/synergy_teams.parquet")
_PRECOMP_DF: Optional[pd.DataFrame] = None
_PRECOMP_LOAD_TRIED = False


def _load_precomputed() -> Optional[pd.DataFrame]:
    """Memoised loader. Returns None if the parquet hasn't been built yet."""
    global _PRECOMP_DF, _PRECOMP_LOAD_TRIED
    if _PRECOMP_DF is not None:
        return _PRECOMP_DF
    if _PRECOMP_LOAD_TRIED:
        return None
    _PRECOMP_LOAD_TRIED = True
    if not _PRECOMP_PATH.exists():
        logger.info("precomputed synergy teams not found at %s", _PRECOMP_PATH)
        return None
    try:
        df = pd.read_parquet(_PRECOMP_PATH)
        # Normalise key columns so lookups are case-insensitive.
        df["pokemon"] = df["pokemon"].astype(str).str.lower()
        df["playstyle"] = df["playstyle"].astype(str).str.lower()
        _PRECOMP_DF = df
        n_combos = (
            df[["pokemon", "playstyle", "max_legendaries"]].drop_duplicates().shape[0]
        )
        logger.info(
            "loaded precomputed synergy teams: %d slots across %d combos",
            len(df), n_combos,
        )
        return df
    except Exception as e:
        logger.warning("failed to load precomputed teams: %s", e)
        return None


def get_precomputed_team(
    pokemon: str, playstyle: str, max_legendaries: int,
) -> Optional[pd.DataFrame]:
    """Return the precomputed team for this combo, or None if not built.

    Returns a DataFrame already shaped like the live synergy recommender's
    output (same columns), sorted by slot order. Callers should treat None
    as "fall back to the live recommender".
    """
    pre = _load_precomputed()
    if pre is None:
        return None
    rows = pre[
        (pre["pokemon"] == pokemon.lower())
        & (pre["playstyle"] == playstyle.lower())
        & (pre["max_legendaries"] == int(max_legendaries))
    ]
    if rows.empty:
        return None
    return rows.sort_values("slot_index").reset_index(drop=True)


def get_pokemon_names():
    df = load_data()
    return df["name"].head(10).tolist()


def find_pokemon(pokemon_name):
    df = load_data()
    result = df[df["name"].str.lower() == pokemon_name.lower()]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


def recommend_teammates(selected_pokemon: str, df: pd.DataFrame, playstyle: str = "balanced",
                       team_size: int = 6, max_legendaries: int = 0,
                       use_synergy: bool = True, lookahead_depth: int = 0) -> pd.DataFrame:
    """
    Recommend Pokemon teammates using synergy-based scoring with optional lookahead.

    Args:
        selected_pokemon: Name of the starting Pokemon
        df: Pokemon DataFrame
        playstyle: "balanced", "offense", "stall", "tank"
        team_size: Target team size
        max_legendaries: Maximum legendary Pokemon allowed
        use_synergy: Whether to use synergy scoring
        lookahead_depth: How many picks ahead to evaluate. Default 0 (greedy)
            because lookahead is a Python-loop blow-up: with 720 candidates and
            depth=1 the pick phase takes ~3.6s, vs ~0.2s greedy. Set to 1 only
            when you specifically want a slightly higher-quality team and can
            tolerate the wait.

    Returns:
        DataFrame with recommended team
    """
    selected_row = df[df["name"].str.lower() == selected_pokemon.lower()]

    if selected_row.empty:
        raise ValueError(f"{selected_pokemon} not found in dataset.")

    # Initialize synergy scorer
    synergy_scorer = PokemonSynergyScorer(df) if use_synergy else None

    # Start with selected Pokemon
    team = [selected_row.iloc[0].to_dict()]
    available_pokemon = df[df["name"].str.lower() != selected_pokemon.lower()].copy()

    # Apply legendary filter
    legendary_count = sum(1 for member in team if member.get("is_legendary", False))
    if legendary_count >= max_legendaries:
        available_pokemon = available_pokemon[~available_pokemon["is_legendary"]]

    while len(team) < team_size and not available_pokemon.empty:
        best_candidate = None
        best_score = float('-inf')

        # Evaluate each candidate
        for _, candidate in available_pokemon.iterrows():
            candidate_dict = candidate.to_dict()

            # Check legendary limit
            if candidate_dict.get("is_legendary", False):
                temp_legendary_count = legendary_count + 1
                if temp_legendary_count > max_legendaries:
                    continue

            # Calculate base score for this candidate
            score = _score_candidate_for_team(candidate_dict, team, playstyle, synergy_scorer)

            # Lookahead evaluation
            if lookahead_depth > 0:
                future_score = _evaluate_future_picks(
                    candidate_dict, team, available_pokemon, playstyle,
                    synergy_scorer, lookahead_depth, max_legendaries
                )
                score += future_score * 0.3  # Weight future considerations

            if score > best_score:
                best_score = score
                best_candidate = candidate_dict

        if best_candidate is None:
            break  # No valid candidates found

        # Add to team
        team.append(best_candidate)
        available_pokemon = available_pokemon[available_pokemon["name"] != best_candidate["name"]]

        # Update legendary count
        if best_candidate.get("is_legendary", False):
            legendary_count += 1

    return pd.DataFrame(team)


def _score_candidate_for_team(candidate: Dict, current_team: List[Dict],
                             playstyle: str, synergy_scorer: Optional[PokemonSynergyScorer]) -> float:
    """
    Score how well a candidate fits the current team.
    Higher scores = better fit.
    """
    if not current_team:
        return 1.0  # Perfect score for empty team

    score = 0.0

    # Synergy-based scoring
    if synergy_scorer:
        synergy_scores = synergy_scorer.score_candidate_for_team(candidate, current_team)
        score += synergy_scores['overall'] * 0.35  # 35% weight on synergy

    # Traditional scoring (weakness coverage, type diversity, etc.)
    traditional_score = _calculate_traditional_score(candidate, current_team, playstyle)
    score += traditional_score * 0.65  # 65% weight on traditional metrics (more diversity)

    return score


def _calculate_traditional_score(candidate: Dict, current_team: List[Dict], playstyle: str) -> float:
    """Calculate traditional team-building score (weakness coverage, diversity, etc.)"""
    team_weaknesses = set()
    team_types = set()
    team_roles = []

    for member in current_team:
        team_weaknesses.update(member.get("weaknesses", []))
        team_roles.append(member.get("role", ""))

        team_types.add(member["type1"])
        if has_type2(member.get("type2")):
            team_types.add(member["type2"])

    # Weakness coverage (higher = better)
    candidate_weaknesses = set(candidate.get("weaknesses", []))
    weakness_overlap = len(team_weaknesses.intersection(candidate_weaknesses))
    weakness_score = 1.0 - (weakness_overlap / max(1, len(team_weaknesses)))

    # Type diversity (lower duplication = higher score)
    type_duplication = 0
    if candidate["type1"] in team_types:
        type_duplication += 1
    if has_type2(candidate.get("type2")) and candidate["type2"] in team_types:
        type_duplication += 1
    type_score = 1.0 - (type_duplication * 0.3)  # Penalty for duplication

    # Role diversity
    candidate_role = candidate.get("role", "")
    role_count = team_roles.count(candidate_role)
    role_score = 1.0 - (role_count * 0.2)  # Penalty for role duplication

    # Stat-based scoring for playstyle
    stat_score = _calculate_stat_score(candidate, playstyle)

    # Combine scores (weighted) - prioritize diversity
    final_score = (
        weakness_score * 0.35 +
        type_score * 0.35 +
        role_score * 0.25 +
        stat_score * 0.05
    )

    return final_score


def _calculate_stat_score(candidate: Dict, playstyle: str) -> float:
    """Calculate stat score based on playstyle preference."""
    atk = candidate.get("attack", 0)
    spa = candidate.get("sp_attack", 0)
    speed = candidate.get("speed", 0)
    hp = candidate.get("hp", 0)
    defense = candidate.get("defense", 0)
    spdef = candidate.get("sp_defense", 0)

    if playstyle == "offense":
        return (speed * 0.5 + max(atk, spa) * 0.5) / 100
    elif playstyle == "stall":
        return (hp * 0.4 + defense * 0.3 + spdef * 0.3) / 100
    elif playstyle == "tank":
        return (max(atk, spa) * 0.5 + hp * 0.3 + defense * 0.2) / 100
    else:  # balanced
        return (atk + spa + speed + hp + defense + spdef) / 600

    return 0.5  # Default


def _evaluate_future_picks(candidate: Dict, current_team: List[Dict],
                          available_pokemon: pd.DataFrame, playstyle: str,
                          synergy_scorer: Optional[PokemonSynergyScorer],
                          depth: int, max_legendaries: int) -> float:
    """
    Evaluate the quality of future picks if this candidate is chosen.
    Returns a score bonus based on future team quality.
    """
    if depth <= 0:
        return 0.0

    # Simulate adding candidate to team
    future_team = current_team + [candidate]
    future_available = available_pokemon[available_pokemon["name"] != candidate["name"]].copy()

    # Apply legendary filter
    legendary_count = sum(1 for member in future_team if member.get("is_legendary", False))
    if legendary_count >= max_legendaries:
        future_available = future_available[~future_available["is_legendary"]]

    if future_available.empty:
        return 0.0

    # Evaluate best future pick
    best_future_score = float('-inf')

    # Sample top candidates to avoid excessive computation
    top_candidates = future_available.head(min(10, len(future_available)))

    for _, future_candidate in top_candidates.iterrows():
        future_dict = future_candidate.to_dict()

        # Check legendary limit
        if future_dict.get("is_legendary", False):
            temp_legendary_count = legendary_count + 1
            if temp_legendary_count > max_legendaries:
                continue

        future_score = _score_candidate_for_team(future_dict, future_team, playstyle, synergy_scorer)

        # Recursive lookahead
        if depth > 1:
            future_score += _evaluate_future_picks(
                future_dict, future_team, future_available, playstyle,
                synergy_scorer, depth - 1, max_legendaries
            ) * 0.5  # Diminishing weight

        best_future_score = max(best_future_score, future_score)

    return best_future_score if best_future_score != float('-inf') else 0.0


@lru_cache(maxsize=10000)
def fetch_json(url):
    # In-process LRU keeps us fast within a single request; the underlying
    # `_http` session also writes responses to the SQLite cache on disk so
    # subsequent app restarts don't hit the network at all.
    response = _http.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1024)
def get_pokemon_moves(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    data = fetch_json(url)

    moves = []

    for move_entry in data["moves"]:
        move_name = move_entry["move"]["name"]
        move_url = move_entry["move"]["url"]

        move_data = fetch_json(move_url)

        if move_data["damage_class"]["name"] == "status":
            continue

        if move_data["power"] is None:
            continue

        moves.append({
            "move": move_name,
            "type": move_data["type"]["name"],
            "power": move_data["power"],
            "accuracy": move_data["accuracy"],
            "damage_class": move_data["damage_class"]["name"]
        })

    return tuple(moves)


def score_move(move):
    accuracy = move["accuracy"] if move["accuracy"] is not None else 100
    return move["power"] * (accuracy / 100)


def recommend_moves(pokemon_name, df, max_moves=4):
    moves = list(get_pokemon_moves(pokemon_name))
    moves_df = pd.DataFrame(moves)

    if moves_df.empty:
        return pd.DataFrame()

    pokemon_row = df[df["name"].str.lower() == pokemon_name.lower()]

    if pokemon_row.empty:
        return pd.DataFrame()

    pokemon_row = pokemon_row.iloc[0]

    pokemon_types = {pokemon_row["type1"]}

    if has_type2(pokemon_row["type2"]):
        pokemon_types.add(pokemon_row["type2"])

    moves_df["score"] = moves_df.apply(score_move, axis=1)
    moves_df["is_stab"] = moves_df["type"].apply(lambda move_type: move_type in pokemon_types)

    selected_moves = []
    used_types = set()

    stab_moves = moves_df[moves_df["is_stab"]].sort_values(by="score", ascending=False)

    if not stab_moves.empty:
        best_stab = stab_moves.iloc[0].to_dict()
        selected_moves.append(best_stab)
        used_types.add(best_stab["type"])

    coverage_moves = moves_df.sort_values(by="score", ascending=False)

    for _, move in coverage_moves.iterrows():
        if len(selected_moves) >= max_moves:
            break

        if move["move"] in [m["move"] for m in selected_moves]:
            continue

        if move["type"] not in used_types:
            selected_moves.append(move.to_dict())
            used_types.add(move["type"])

    return pd.DataFrame(selected_moves)


def add_movesets_to_team(team, df, max_moves=4):
    team = team.copy()
    recommended_moves = []

    for _, row in team.iterrows():
        moves_df = recommend_moves(row["name"], df, max_moves=max_moves)

        if moves_df.empty:
            recommended_moves.append([])
        else:
            recommended_moves.append(moves_df["move"].tolist())

    team["recommended_moves"] = recommended_moves

    return team


def summarize_team(team, playstyle):
    all_weaknesses = []
    all_types = []
    roles = []

    for _, row in team.iterrows():
        all_weaknesses.extend(row["weaknesses"])
        all_types.append(row["type1"])

        if has_type2(row["type2"]):
            all_types.append(row["type2"])

        roles.append(row["role"])

    weakness_counts = pd.Series(all_weaknesses).value_counts().to_dict()
    type_counts = pd.Series(all_types).value_counts().to_dict()
    role_counts = pd.Series(roles).value_counts().to_dict()

    legendary_count = int(team["is_legendary"].sum())
    average_speed = round(team["speed"].mean(), 2)
    average_attack = round(team["attack"].mean(), 2)
    average_sp_attack = round(team["sp_attack"].mean(), 2)
    average_bulk = round((team["hp"].mean() + team["defense"].mean() + team["sp_defense"].mean()) / 3, 2)

    return {
        "playstyle": playstyle,
        "legendary_count": legendary_count,
        "average_speed": average_speed,
        "average_attack": average_attack,
        "average_sp_attack": average_sp_attack,
        "average_bulk": average_bulk,
        "type_distribution": type_counts,
        "common_weaknesses": weakness_counts,
        "role_distribution": role_counts
    }


def format_name(name):
    return str(name).replace("-", " ").title()


def get_pokemon_types(row):
    types = [row["type1"]]

    if has_type2(row["type2"]):
        types.append(row["type2"])

    return types


def explain_choice(row, previous_team, playstyle):
    name = format_name(row["name"])
    role = row["role"]
    pokemon_types = get_pokemon_types(row)
    pokemon_weaknesses = set(row["weaknesses"])

    if not previous_team:
        return (
            f"{name} is the anchor Pokemon for this team. "
            f"The rest of the team is built around supporting its typing, role, and weaknesses."
        )

    previous_types = set()
    previous_weaknesses = set()
    previous_roles = []

    for member in previous_team:
        previous_types.update(get_pokemon_types(member))
        previous_weaknesses.update(member["weaknesses"])
        previous_roles.append(member["role"])

    new_types = [t for t in pokemon_types if t not in previous_types]
    weakness_overlap = len(pokemon_weaknesses.intersection(previous_weaknesses))

    reasons = []

    if new_types:
        reasons.append(f"adds {', '.join(new_types)} typing to the team")

    if weakness_overlap == 0:
        reasons.append("does not repeat the current team's main weaknesses")
    elif weakness_overlap == 1:
        reasons.append("only slightly overlaps with the current team's weaknesses")
    else:
        reasons.append("still fits despite some shared weaknesses because of its stats and role")

    if role not in previous_roles:
        reasons.append(f"adds a new role as a {role}")
    else:
        reasons.append(f"supports the existing {role} role")

    if playstyle == "offense":
        if row["speed"] >= 100:
            reasons.append("fits the hyper offense playstyle because it has high speed")
        elif max(row["attack"], row["sp_attack"]) >= 100:
            reasons.append("fits the hyper offense playstyle because it has strong attacking stats")

    elif playstyle == "stall":
        bulk_score = (row["hp"] + row["defense"] + row["sp_defense"]) / 3

        if bulk_score >= 80:
            reasons.append("fits the stall playstyle because it has strong defensive stats")
        else:
            reasons.append("helps the stall team by adding useful defensive coverage")

    elif playstyle == "tank":
        bulk_score = (row["hp"] + row["defense"] + row["sp_defense"]) / 3
        attack_score = max(row["attack"], row["sp_attack"])

        if attack_score >= 100 and bulk_score >= 70:
            reasons.append("fits the tank playstyle because it can take hits while still dealing damage")
        elif attack_score >= 100:
            reasons.append("fits the tank playstyle because it hits hard")

    else:
        reasons.append("fits the balanced playstyle by contributing useful stats and team coverage")

    return f"{name} is recommended because it " + "; ".join(reasons) + "."


def add_reasons_to_team(team, playstyle):
    team = team.copy()
    reasons = []
    previous_team = []

    for _, row in team.iterrows():
        reason = explain_choice(row, previous_team, playstyle)
        reasons.append(reason)
        previous_team.append(row.to_dict())

    team["reason"] = reasons

    return team
