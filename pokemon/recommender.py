import ast
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from typing import List, Dict, Optional
from synergy_scorer import PokemonSynergyScorer


# ---- PokeAPI disk cache ----
# fetch_json (kept around for legacy callers) layers a disk cache under its
# in-process @lru_cache. The synergy fallback path no longer calls PokeAPI
# at all — see TYPE_TO_MOVES below — so this cache is now mostly insurance.
POKEAPI_CACHE_DIR = Path(os.environ.get("POKEAPI_CACHE_DIR", "data/cache/pokeapi"))


def _pokeapi_cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return POKEAPI_CACHE_DIR / f"{h}.json"


# ---- Offline type → moves table ----
# Picking a Pokemon outside the Smogon chaos dataset (e.g. Pikachu, Eevee) used
# to fall back to a PokeAPI lookup that fired ~100 sequential HTTP calls per
# team member — roughly 600 requests for a 6-mon team. We now derive the
# "recommended moves" purely from the Pokemon's types using this curated table
# of strong attacking moves per type. Zero network calls, instant team build.
TYPE_TO_MOVES: dict[str, list[dict]] = {
    "normal":   [{"move": "return",        "power": 102, "accuracy": 100},
                 {"move": "body-slam",     "power":  85, "accuracy": 100},
                 {"move": "facade",        "power":  70, "accuracy": 100},
                 {"move": "double-edge",   "power": 120, "accuracy": 100}],
    "fire":     [{"move": "flamethrower",  "power":  90, "accuracy": 100},
                 {"move": "fire-blast",    "power": 110, "accuracy":  85},
                 {"move": "flare-blitz",   "power": 120, "accuracy": 100},
                 {"move": "heat-wave",     "power":  95, "accuracy":  90}],
    "water":    [{"move": "surf",          "power":  90, "accuracy": 100},
                 {"move": "hydro-pump",    "power": 110, "accuracy":  80},
                 {"move": "scald",         "power":  80, "accuracy": 100},
                 {"move": "liquidation",   "power":  85, "accuracy": 100}],
    "electric": [{"move": "thunderbolt",   "power":  90, "accuracy": 100},
                 {"move": "thunder",       "power": 110, "accuracy":  70},
                 {"move": "wild-charge",   "power":  90, "accuracy": 100},
                 {"move": "volt-switch",   "power":  70, "accuracy": 100}],
    "grass":    [{"move": "leaf-blade",    "power":  90, "accuracy": 100},
                 {"move": "energy-ball",   "power":  90, "accuracy": 100},
                 {"move": "giga-drain",    "power":  75, "accuracy": 100},
                 {"move": "leaf-storm",    "power": 130, "accuracy":  90}],
    "ice":      [{"move": "ice-beam",      "power":  90, "accuracy": 100},
                 {"move": "blizzard",      "power": 110, "accuracy":  70},
                 {"move": "ice-punch",     "power":  75, "accuracy": 100},
                 {"move": "icicle-crash",  "power":  85, "accuracy":  90}],
    "fighting": [{"move": "close-combat",  "power": 120, "accuracy": 100},
                 {"move": "drain-punch",   "power":  75, "accuracy": 100},
                 {"move": "focus-blast",   "power": 120, "accuracy":  70},
                 {"move": "brick-break",   "power":  75, "accuracy": 100}],
    "poison":   [{"move": "sludge-bomb",   "power":  90, "accuracy": 100},
                 {"move": "gunk-shot",     "power": 120, "accuracy":  80},
                 {"move": "sludge-wave",   "power":  95, "accuracy": 100},
                 {"move": "poison-jab",    "power":  80, "accuracy": 100}],
    "ground":   [{"move": "earthquake",    "power": 100, "accuracy": 100},
                 {"move": "earth-power",   "power":  90, "accuracy": 100},
                 {"move": "high-horsepower","power": 95, "accuracy":  95},
                 {"move": "drill-run",     "power":  80, "accuracy":  95}],
    "flying":   [{"move": "brave-bird",    "power": 120, "accuracy": 100},
                 {"move": "hurricane",     "power": 110, "accuracy":  70},
                 {"move": "air-slash",     "power":  75, "accuracy":  95},
                 {"move": "dual-wingbeat", "power":  40, "accuracy":  90}],
    "psychic":  [{"move": "psychic",       "power":  90, "accuracy": 100},
                 {"move": "psyshock",      "power":  80, "accuracy": 100},
                 {"move": "zen-headbutt",  "power":  80, "accuracy":  90},
                 {"move": "future-sight",  "power": 120, "accuracy": 100}],
    "bug":      [{"move": "u-turn",        "power":  70, "accuracy": 100},
                 {"move": "bug-buzz",      "power":  90, "accuracy": 100},
                 {"move": "megahorn",      "power": 120, "accuracy":  85},
                 {"move": "x-scissor",     "power":  80, "accuracy": 100}],
    "rock":     [{"move": "stone-edge",    "power": 100, "accuracy":  80},
                 {"move": "rock-slide",    "power":  75, "accuracy":  90},
                 {"move": "power-gem",     "power":  80, "accuracy": 100},
                 {"move": "accelerock",    "power":  40, "accuracy": 100}],
    "ghost":    [{"move": "shadow-ball",   "power":  80, "accuracy": 100},
                 {"move": "shadow-claw",   "power":  70, "accuracy": 100},
                 {"move": "poltergeist",   "power": 110, "accuracy":  90},
                 {"move": "hex",           "power":  65, "accuracy": 100}],
    "dragon":   [{"move": "outrage",       "power": 120, "accuracy": 100},
                 {"move": "draco-meteor",  "power": 130, "accuracy":  90},
                 {"move": "dragon-claw",   "power":  80, "accuracy": 100},
                 {"move": "dragon-pulse",  "power":  85, "accuracy": 100}],
    "dark":     [{"move": "knock-off",     "power":  65, "accuracy": 100},
                 {"move": "crunch",        "power":  80, "accuracy": 100},
                 {"move": "dark-pulse",    "power":  80, "accuracy": 100},
                 {"move": "sucker-punch",  "power":  70, "accuracy": 100}],
    "steel":    [{"move": "iron-head",     "power":  80, "accuracy": 100},
                 {"move": "flash-cannon",  "power":  80, "accuracy": 100},
                 {"move": "meteor-mash",   "power":  90, "accuracy":  90},
                 {"move": "heavy-slam",    "power":  80, "accuracy": 100}],
    "fairy":    [{"move": "play-rough",    "power":  90, "accuracy":  90},
                 {"move": "moonblast",     "power":  95, "accuracy": 100},
                 {"move": "dazzling-gleam","power":  80, "accuracy": 100},
                 {"move": "draining-kiss", "power":  50, "accuracy": 100}],
}


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
                       use_synergy: bool = True, lookahead_depth: int = 2) -> pd.DataFrame:
    """
    Recommend Pokemon teammates using synergy-based scoring with lookahead.

    Args:
        selected_pokemon: Name of the starting Pokemon
        df: Pokemon DataFrame
        playstyle: "balanced", "offense", "stall", "tank"
        team_size: Target team size
        max_legendaries: Maximum legendary Pokemon allowed
        use_synergy: Whether to use synergy scoring
        lookahead_depth: How many picks ahead to evaluate (0 = greedy, 1+ = lookahead)

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
    """Fetch a PokeAPI URL, consulting an on-disk JSON cache first.

    Layered cache: @lru_cache (in-process) → disk cache → network. Disk hits
    are what make team-build with cold in-memory cache fast across restarts.
    """
    cache_path = _pokeapi_cache_path(url)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass  # corrupt cache file — fall through and refetch

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    try:
        POKEAPI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Write atomically so a Ctrl-C mid-write doesn't leave a half file.
        tmp = cache_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, cache_path)
    except OSError:
        pass  # disk write failures must not break the request

    return data


@lru_cache(maxsize=1024)
def get_pokemon_moves(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    data = fetch_json(url)

    entries = data["moves"]
    move_urls = [m["move"]["url"] for m in entries]
    move_names = [m["move"]["name"] for m in entries]

    # Parallel fetch — PokeAPI move detail pages are the bottleneck (~100 per
    # Pokemon). Threads are fine here because fetch_json is I/O bound and its
    # disk-write step is short. Warm cache hits skip the network entirely.
    max_workers = min(16, max(1, len(move_urls)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        move_data_list = list(executor.map(fetch_json, move_urls))

    moves = []
    for move_name, move_data in zip(move_names, move_data_list):
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


# Coverage-move priority order: high-BP, broadly useful attacking types listed
# first so they fill remaining slots after STAB if the Pokemon's own types
# don't supply enough moves.
_COVERAGE_TYPE_ORDER = [
    "ground", "ice", "fighting", "rock", "fire", "electric",
    "dark", "ghost", "fairy", "psychic", "water", "grass",
    "dragon", "steel", "flying", "bug", "poison", "normal",
]


def recommend_moves(pokemon_name, df, max_moves=4):
    """Recommend up to ``max_moves`` strong attacking moves for a Pokemon.

    Uses the offline ``TYPE_TO_MOVES`` table so there are zero network calls,
    even for Pokemon that don't appear in any Smogon chaos metagame. Picks
    one STAB move per Pokemon type first, then fills remaining slots with
    high-power coverage moves from broadly useful types.
    """
    pokemon_row = df[df["name"].str.lower() == pokemon_name.lower()]
    if pokemon_row.empty:
        return pd.DataFrame()
    pokemon_row = pokemon_row.iloc[0]

    pokemon_types: list[str] = []
    t1 = pokemon_row.get("type1")
    if isinstance(t1, str) and t1:
        pokemon_types.append(t1.lower())
    if has_type2(pokemon_row.get("type2")):
        t2 = pokemon_row.get("type2")
        if isinstance(t2, str) and t2.lower() not in pokemon_types:
            pokemon_types.append(t2.lower())

    selected: list[dict] = []
    used_moves: set[str] = set()
    used_types: set[str] = set()

    def _take(type_name: str) -> bool:
        candidates = TYPE_TO_MOVES.get(type_name, [])
        for cand in candidates:
            if cand["move"] in used_moves:
                continue
            selected.append({
                "move": cand["move"],
                "type": type_name,
                "power": cand["power"],
                "accuracy": cand["accuracy"],
                "damage_class": "physical",  # placeholder; not displayed
                "score": score_move(cand),
            })
            used_moves.add(cand["move"])
            used_types.add(type_name)
            return True
        return False

    # STAB first — one move per native type.
    for typ in pokemon_types:
        if len(selected) >= max_moves:
            break
        _take(typ)

    # Coverage — fill remaining slots from the priority order, skipping STAB
    # types we've already used.
    for typ in _COVERAGE_TYPE_ORDER:
        if len(selected) >= max_moves:
            break
        if typ in used_types:
            continue
        _take(typ)

    return pd.DataFrame(selected)


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
            f"{name} is the anchor Pokémon for this team. "
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