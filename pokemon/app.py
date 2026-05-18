from pathlib import Path

from flask import Flask, render_template, request

from recommender import (
    load_data,
    recommend_teammates,
    add_movesets_to_team,
    summarize_team,
    add_reasons_to_team,
)

app = Flask(__name__)

df = load_data()

# Lazily load the chaos recommender (requires data_pipeline outputs + trained model).
_chaos_rec = None
_chaos_load_error = None

CHAOS_ARTIFACTS = [
    "data/processed/pokemon_meta.parquet",
    "data/processed/teammates.parquet",
    "data/processed/movesets.parquet",
    "data/embeddings/embeddings.parquet",
]


def get_chaos_recommender():
    global _chaos_rec, _chaos_load_error
    if _chaos_rec is not None:
        return _chaos_rec
    missing = [p for p in CHAOS_ARTIFACTS if not Path(p).exists()]
    if missing:
        _chaos_load_error = (
            "Chaos dataset not built yet. Run: "
            "python data_pipeline/fetch_chaos.py, then build_dataset.py, "
            "build_embeddings.py, and python -m model.train_viability."
        )
        return None
    try:
        from chaos_recommender import ChaosRecommender
        _chaos_rec = ChaosRecommender.load()
        return _chaos_rec
    except Exception as e:
        _chaos_load_error = "Failed to load chaos recommender: " + str(e)
        return None


def list_available_metagames():
    rec = get_chaos_recommender()
    if rec is None:
        return []
    return sorted(rec.list_metagames())


def _slot_to_template_pokemon(slot, base_df):
    base = base_df[base_df["name"].str.lower() == slot.name.lower()]
    row = base.iloc[0].to_dict() if not base.empty else {}
    return {
        "name": slot.name,
        "type1": (slot.types[0] if slot.types else row.get("type1")),
        "type2": (slot.types[1] if len(slot.types) > 1 else row.get("type2")),
        "hp": row.get("hp"),
        "attack": row.get("attack"),
        "defense": row.get("defense"),
        "sp_attack": row.get("sp_attack"),
        "sp_defense": row.get("sp_defense"),
        "speed": row.get("speed"),
        "role": slot.role or row.get("role"),
        "is_legendary": slot.is_legendary,
        "weaknesses": row.get("weaknesses", []),
        "recommended_moves": slot.moves,
        "item": slot.item,
        "ability": slot.ability,
        "spread": slot.spread,
        "usage": slot.usage,
        "reason": slot.reason,
    }


def _summarize_chaos_team(team_data, gen, tier):
    n = max(1, len(team_data))
    def avg(key):
        return round(sum((p.get(key) or 0) for p in team_data) / n, 2)
    weak_counts = {}
    role_counts = {}
    for p in team_data:
        for w in (p.get("weaknesses") or []):
            weak_counts[w] = weak_counts.get(w, 0) + 1
        r = p.get("role") or ""
        role_counts[r] = role_counts.get(r, 0) + 1
    return {
        "playstyle": "Smogon gen" + str(gen) + " " + str(tier).upper(),
        "legendary_count": sum(1 for p in team_data if p.get("is_legendary")),
        "average_speed": avg("speed"),
        "average_attack": avg("attack"),
        "average_sp_attack": avg("sp_attack"),
        "average_bulk": round((avg("hp") + avg("defense") + avg("sp_defense")) / 3, 2),
        "common_weaknesses": dict(sorted(weak_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "role_distribution": role_counts,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    team = None
    summary = None
    error = None

    metagames = list_available_metagames()
    chaos_available = bool(metagames)
    chaos_message = _chaos_load_error if not chaos_available else None

    if request.method == "POST":
        pokemon_name = request.form.get("pokemon", "").strip().lower()
        mode = request.form.get("mode", "chaos" if chaos_available else "synergy")

        if mode == "chaos" and chaos_available:
            metagame_raw = (request.form.get("metagame") or "").strip()
            gen, tier = None, None
            if "|" in metagame_raw:
                g, t = metagame_raw.split("|", 1)
                if g.strip().isdigit():
                    gen = int(g.strip())
                    tier = t.strip().lower() or None
            max_leg_raw = request.form.get("max_legendaries", "")
            max_leg = int(max_leg_raw) if max_leg_raw.isdigit() else None

            rec = get_chaos_recommender()
            try:
                result = rec.build_team(
                    pokemon_name, gen=gen, tier=tier,
                    team_size=6, max_legendaries=max_leg,
                )
                team_data = [_slot_to_template_pokemon(s, df) for s in result.members]
                summary = _summarize_chaos_team(team_data, result.gen, result.tier)
                summary["viability_score"] = round(result.viability_score, 3)
                team = team_data
            except ValueError as e:
                error = (
                    "No Smogon chaos data found for '" + pokemon_name +
                    "' - falling back to type-synergy recommender. (" + str(e) + ")"
                )
                mode = "synergy"

        if mode == "synergy":
            playstyle = request.form.get("playstyle", "balanced")
            max_legendaries = int(request.form.get("max_legendaries", 0) or 0)
            try:
                result = recommend_teammates(
                    pokemon_name, df,
                    playstyle=playstyle, max_legendaries=max_legendaries,
                )
                result = add_movesets_to_team(result, df)
                summary = summarize_team(result, playstyle)
                result = add_reasons_to_team(result, playstyle)
                display_columns = [
                    "name", "type1", "type2", "hp", "attack", "defense",
                    "sp_attack", "sp_defense", "speed", "role", "weaknesses",
                    "recommended_moves", "is_legendary", "reason",
                ]
                result = result.where(result.notna(), None)
                team = result[display_columns].to_dict(orient="records")
            except Exception as e:
                error = (error + " ") if error else ""
                error += "Synergy recommender failed: " + str(e)

    return render_template(
        "index.html",
        team=team,
        summary=summary,
        error=error,
        metagames=metagames,
        chaos_available=chaos_available,
        chaos_message=chaos_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
