import ast
from functools import lru_cache

import pandas as pd
import requests


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


def recommend_teammates(selected_pokemon, df, playstyle="balanced", team_size=6, max_legendaries=0):
    selected_row = df[df["name"].str.lower() == selected_pokemon.lower()]

    if selected_row.empty:
        return f"{selected_pokemon} not found in dataset."

    if playstyle == "offensive":
        playstyle = "offense"

    team = [selected_row.iloc[0].to_dict()]
    candidates = df[df["name"].str.lower() != selected_pokemon.lower()].copy()

    while len(team) < team_size:
        team_weaknesses = set()
        team_types = []
        team_roles = []

        for member in team:
            team_weaknesses.update(member["weaknesses"])
            team_roles.append(member["role"])

            team_types.append(member["type1"])

            if has_type2(member["type2"]):
                team_types.append(member["type2"])

        legendary_count = sum(1 for member in team if member["is_legendary"])

        if legendary_count >= max_legendaries:
            candidates = candidates[candidates["is_legendary"] == False]

        if candidates.empty:
            break

        def score_candidate(row):
            weaknesses = set(row["weaknesses"])
            weakness_overlap = len(team_weaknesses.intersection(weaknesses))

            type_duplication = 0

            if row["type1"] in team_types:
                type_duplication += 1

            if has_type2(row["type2"]) and row["type2"] in team_types:
                type_duplication += 1

            role_duplication = team_roles.count(row["role"])

            atk = row["attack"]
            spa = row["sp_attack"]
            speed = row["speed"]
            hp = row["hp"]
            defense = row["defense"]
            spdef = row["sp_defense"]

            if playstyle == "offense":
                stat_score = (speed * 0.5) + (max(atk, spa) * 0.5)

            elif playstyle == "stall":
                stat_score = (hp * 0.4) + (defense * 0.3) + (spdef * 0.3)

            elif playstyle == "tank":
                stat_score = (max(atk, spa) * 0.5) + (hp * 0.3) + (defense * 0.2)

            else:
                stat_score = (atk + spa + speed + hp + defense + spdef) / 6

            score = (
                weakness_overlap * 3
                + type_duplication * 2
                + role_duplication * 2
                - stat_score / 50
            )

            return score

        candidates["team_score"] = candidates.apply(score_candidate, axis=1)
        candidates = candidates.sort_values(by="team_score", ascending=True)

        next_pick = candidates.iloc[0]
        team.append(next_pick.to_dict())

        candidates = candidates[candidates["name"] != next_pick["name"]]

    return pd.DataFrame(team)


@lru_cache(maxsize=10000)
def fetch_json(url):
    response = requests.get(url)
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