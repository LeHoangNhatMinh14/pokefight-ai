from flask import Flask, render_template, request

from recommender import (
    load_data,
    recommend_teammates,
    add_movesets_to_team
)

app = Flask(__name__)

df = load_data()


@app.route("/", methods=["GET", "POST"])
def index():
    team = None
    error = None

    if request.method == "POST":
        pokemon_name = request.form.get("pokemon", "").strip().lower()
        playstyle = request.form.get("playstyle", "balanced")
        max_legendaries = int(request.form.get("max_legendaries", 0))

        result = recommend_teammates(
            pokemon_name,
            df,
            playstyle=playstyle,
            max_legendaries=max_legendaries
        )

        if isinstance(result, str):
            error = result
        else:
            result = add_movesets_to_team(result, df)

            display_columns = [
                "name",
                "type1",
                "type2",
                "hp",
                "attack",
                "defense",
                "sp_attack",
                "sp_defense",
                "speed",
                "role",
                "weaknesses",
                "recommended_moves",
                "is_legendary",
                "team_score"
            ]

            team = result[display_columns].to_dict(orient="records")

    return render_template("index.html", team=team, error=error)


if __name__ == "__main__":
    app.run(debug=True)