"""
Chaos-aware Pokemon team builder.

Given a "favorite" Pokemon name, this module picks a metagame (gen, tier) where
that Pokemon has the strongest competitive presence, then uses chaos teammate
co-occurrence + a learned viability reranker to build a 6-mon team. For each
member it picks the most-used moves / item / ability / spread from chaos data,
falling back to PokeAPI / type heuristics when chaos data is missing.

Public API:
    rec = ChaosRecommender.load()
    team = rec.build_team("garchomp", gen=4, tier="ou", team_size=6)
    print(team)        # list of TeamSlot dataclasses
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline.build_dataset import normalize_name
from model.features import FEATURE_NAMES, FeatureBundle, team_features


# ---------- public data classes --------------------------------------------------

@dataclass
class TeamSlot:
    name: str
    types: list[str]
    role: Optional[str]
    usage: float
    moves: list[str]
    item: Optional[str]
    ability: Optional[str]
    spread: Optional[str]
    is_legendary: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_STAT_LABELS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]

# Words that look better in Showdown pastes when title-cased nicely.
# Smogon chaos data sometimes uses compact lowercase like "brickbreak", which
# Showdown's importer accepts but looks ugly. We expand the common cases.
_HIDDEN_POWER_TYPES = {
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
}


def _pretty_name(s: str) -> str:
    """Best-effort title-casing for items / abilities / Pokemon names."""
    if not s:
        return s
    return " ".join(part.capitalize() for part in s.replace("-", " ").split())


def _format_move(move: str) -> str:
    """Convert chaos move name into Showdown paste format.

    Handles both the spaced form ("Hidden Power Flying") and the compact lowercase
    form ("hiddenpowerflying") that some chaos exports use. Output is always:
    'Hidden Power [Flying]' for HP moves.
    """
    if not move:
        return move
    raw = move.strip()
    lower_compact = raw.lower().replace(" ", "").replace("-", "")
    if lower_compact.startswith("hiddenpower") and len(lower_compact) > len("hiddenpower"):
        elem = lower_compact[len("hiddenpower"):]
        if elem in _HIDDEN_POWER_TYPES:
            return f"Hidden Power [{elem.capitalize()}]"
    # Already has spaces? Just title-case each word.
    if " " in raw:
        return " ".join(w.capitalize() for w in raw.split())
    # No spaces: try to split common compound moves (rockslide, dragondance, etc.)
    # by checking against a small dictionary of common heads.
    heads = (
        "rock", "dragon", "iron", "flame", "thunder", "ice", "fire", "stone",
        "earth", "shadow", "psycho", "psyshic", "aqua", "leaf", "giga", "mega",
        "hyper", "swords", "bullet", "body", "double", "self", "sleep", "calm",
        "seismic", "wing", "drill", "leech", "mach", "close", "secret", "high",
        "mud", "u", "v", "x",
    )
    low = raw.lower()
    for head in sorted(heads, key=len, reverse=True):
        if low.startswith(head) and len(low) > len(head):
            return head.capitalize() + " " + low[len(head):].capitalize()
    return raw.capitalize()


def _format_spread(spread: str | None) -> tuple[str | None, str | None]:
    """Parse 'Adamant:0/252/0/0/4/252' into ('Adamant', '252 Atk / 4 SpD / 252 Spe').

    Returns (nature, ev_line). Either may be None if the spread is missing or
    malformed (older gens often don't have meaningful EVs).
    """
    if not spread or ":" not in spread:
        return None, None
    nature, evs_raw = spread.split(":", 1)
    parts = evs_raw.split("/")
    if len(parts) != 6:
        return nature.strip() or None, None
    try:
        evs = [int(p) for p in parts]
    except ValueError:
        return nature.strip() or None, None
    pieces = [f"{v} {label}" for v, label in zip(evs, _STAT_LABELS) if v > 0]
    ev_line = " / ".join(pieces) if pieces else None
    return nature.strip() or None, ev_line


@dataclass
class TeamRecommendation:
    gen: int
    tier: str
    favorite: str
    viability_score: float
    members: list[TeamSlot] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gen": self.gen,
            "tier": self.tier,
            "favorite": self.favorite,
            "viability_score": self.viability_score,
            "members": [m.to_dict() for m in self.members],
        }

    def to_showdown_paste(self) -> str:
        """Render the team in Pokemon Showdown's importable team format."""
        blocks: list[str] = []
        for slot in self.members:
            lines: list[str] = []
            # Header: 'Name @ Item' (drop item for gens that have no real items)
            name_display = _pretty_name(slot.name)
            if slot.item and slot.item.lower() not in ("no item", "nothing"):
                lines.append(f"{name_display} @ {_pretty_name(slot.item)}")
            else:
                lines.append(name_display)
            if slot.ability and slot.ability.lower() not in ("no ability",):
                lines.append(f"Ability: {_pretty_name(slot.ability)}")
            nature, ev_line = _format_spread(slot.spread)
            if ev_line:
                lines.append(f"EVs: {ev_line}")
            if nature:
                lines.append(f"{nature} Nature")
            for move in slot.moves[:4]:
                lines.append(f"- {_format_move(move)}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + "\n"


# ---------- recommender ----------------------------------------------------------

class ChaosRecommender:
    def __init__(
        self,
        bundle: FeatureBundle,
        movesets: pd.DataFrame,
        viability_model: Optional[lgb.Booster] = None,
    ) -> None:
        self.bundle = bundle
        self.movesets = movesets
        self.model = viability_model

        # Indexed move/item/ability lookup
        self._moveset_idx: dict[tuple, pd.DataFrame] = {}
        for (gen, tier, name, kind), df in movesets.groupby(["gen", "tier", "name", "kind"]):
            self._moveset_idx[(gen, tier, name, kind)] = df.sort_values(
                "weight", ascending=False
            ).reset_index(drop=True)

    # ----- factory --------------------------------------------------------------

    @classmethod
    def load(
        cls,
        meta_path: str | Path = "data/processed/pokemon_meta.parquet",
        teammates_path: str | Path = "data/processed/teammates.parquet",
        movesets_path: str | Path = "data/processed/movesets.parquet",
        embeddings_path: str | Path = "data/embeddings/embeddings.parquet",
        model_path: str | Path = "model/viability.lgb",
    ) -> "ChaosRecommender":
        bundle = FeatureBundle(
            meta=pd.read_parquet(meta_path),
            teammates=pd.read_parquet(teammates_path),
            embeddings=pd.read_parquet(embeddings_path),
        )
        movesets = pd.read_parquet(movesets_path)
        model_path = Path(model_path)
        model = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None
        return cls(bundle, movesets, model)

    # ----- metagame selection ---------------------------------------------------

    def list_metagames(self) -> list[tuple[int, str]]:
        df = self.bundle.meta[["gen", "tier"]].drop_duplicates()
        return [tuple(r) for r in df.values.tolist()]

    def best_metagame_for(self, name: str) -> Optional[tuple[int, str, float]]:
        """Return (gen, tier, usage) where this Pokemon is most-used."""
        canon = normalize_name(name)
        rows = self.bundle.meta[self.bundle.meta["name"] == canon]
        if rows.empty:
            return None
        top = rows.sort_values("usage", ascending=False).iloc[0]
        return int(top["gen"]), str(top["tier"]), float(top["usage"])

    # ----- core: build a team ---------------------------------------------------

    def build_team(
        self,
        favorite: str,
        gen: Optional[int] = None,
        tier: Optional[str] = None,
        team_size: int = 6,
        *,
        beam_width: int = 8,
        candidate_pool: int = 20,
        max_legendaries: Optional[int] = None,
    ) -> TeamRecommendation:
        favorite = normalize_name(favorite)

        if gen is None or tier is None:
            meta_pick = self.best_metagame_for(favorite)
            if meta_pick is None:
                raise ValueError(
                    f"'{favorite}' has no Smogon chaos data in any loaded metagame."
                )
            gen, tier, _ = meta_pick

        pool = self.bundle.names_in(int(gen), str(tier))
        if favorite not in pool:
            # Try to find any metagame containing the favorite
            meta_pick = self.best_metagame_for(favorite)
            if meta_pick is None:
                raise ValueError(
                    f"'{favorite}' is not present in gen{gen}{tier}."
                )
            gen, tier, _ = meta_pick
            pool = self.bundle.names_in(int(gen), str(tier))

        team_size = min(team_size, len(pool))

        # ---- Beam search ----
        # Each beam = (team_names: list[str], score: float)
        beams: list[tuple[list[str], float]] = [([favorite], 0.0)]
        while any(len(t) < team_size for t, _ in beams):
            new_beams: list[tuple[list[str], float]] = []
            for team, _ in beams:
                if len(team) >= team_size:
                    new_beams.append((team, 0.0))
                    continue
                # Score candidates by sum of teammate weights with current team
                cands = []
                for c in pool:
                    if c in team:
                        continue
                    meta_row = self.bundle.meta_row(int(gen), str(tier), c)
                    if max_legendaries is not None and meta_row and bool(
                        meta_row.get("is_legendary", False)
                    ):
                        lc = sum(
                            1 for m in team
                            if bool((self.bundle.meta_row(int(gen), str(tier), m) or {})
                                    .get("is_legendary", False))
                        )
                        if lc >= max_legendaries:
                            continue
                    w = 0.0
                    for member in team:
                        w += self.bundle.teammate_weight(int(gen), str(tier), member, c)
                        w += self.bundle.teammate_weight(int(gen), str(tier), c, member)
                    # Slight bias toward higher-usage mons as tiebreaker
                    usage = float((meta_row or {}).get("usage", 0.0) or 0.0)
                    cands.append((c, math.log1p(w) + 0.1 * usage))
                cands.sort(key=lambda x: x[1], reverse=True)
                for cand_name, cand_score in cands[:candidate_pool]:
                    new_team = team + [cand_name]
                    new_beams.append((new_team, cand_score))

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

            scored.sort(key=lambda x: x[1], reverse=True)
            # Dedup beams (same set of names)
            seen: set[frozenset[str]] = set()
            unique = []
            for team, score in scored:
                key = frozenset(team)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((team, score))
                if len(unique) >= beam_width:
                    break
            beams = unique
            # If no progress (all beams already full), stop
            if all(len(t) >= team_size for t, _ in beams):
                break

        # Pick the top beam
        best_team, best_score = beams[0]

        # ---- Fill in builds ----
        members = [
            self._build_slot(int(gen), str(tier), name, team_context=best_team)
            for name in best_team
        ]
        return TeamRecommendation(
            gen=int(gen),
            tier=str(tier),
            favorite=favorite,
            viability_score=float(best_score),
            members=members,
        )

    # ----- build a single slot --------------------------------------------------

    def _top_values(self, gen: int, tier: str, name: str, kind: str, k: int) -> list[str]:
        df = self._moveset_idx.get((gen, tier, name, kind))
        if df is None or df.empty:
            return []
        return df["value"].head(k).tolist()

    def _build_slot(
        self, gen: int, tier: str, name: str, team_context: list[str],
    ) -> TeamSlot:
        meta = self.bundle.meta_row(gen, tier, name) or {}
        types = []
        if meta.get("type1"):
            types.append(str(meta["type1"]).lower())
        t2 = meta.get("type2")
        if t2 and not (isinstance(t2, float) and math.isnan(t2)):
            types.append(str(t2).lower())

        moves = self._top_values(gen, tier, name, "move", 4)
        items = self._top_values(gen, tier, name, "item", 1)
        abilities = self._top_values(gen, tier, name, "ability", 1)
        spreads = self._top_values(gen, tier, name, "spread", 1)

        reason = self._reason(gen, tier, name, team_context)
        return TeamSlot(
            name=name,
            types=types,
            role=meta.get("role"),
            usage=float(meta.get("usage", 0.0) or 0.0),
            moves=moves,
            item=items[0] if items else None,
            ability=abilities[0] if abilities else None,
            spread=spreads[0] if spreads else None,
            is_legendary=bool(meta.get("is_legendary", False)),
            reason=reason,
        )

    def _reason(self, gen: int, tier: str, name: str, team: list[str]) -> str:
        if name == team[0]:
            usage = float((self.bundle.meta_row(gen, tier, name) or {}).get("usage", 0))
            return (f"Anchor pick. In gen{gen} {tier.upper()} this Pokemon appears "
                    f"in {usage*100:.1f}% of teams; the rest of the team is built "
                    f"around it.")
        # Sum teammate weights with rest of team to explain
        total = 0.0
        partners = []
        for m in team:
            if m == name:
                continue
            w = self.bundle.teammate_weight(gen, tier, name, m)
            w += self.bundle.teammate_weight(gen, tier, m, name)
            if w > 0:
                partners.append((m, w))
                total += w
        partners.sort(key=lambda x: x[1], reverse=True)
        if partners:
            top = ", ".join(p for p, _ in partners[:2])
            return (f"Frequently paired with {top} in gen{gen} {tier.upper()} "
                    f"according to Smogon usage stats.")
        return f"Added for type / role coverage in gen{gen} {tier.upper()}."


# ---------- CLI demo --------------------------------------------------------------

def _demo() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("favorite")
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--tier", type=str, default=None)
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--beam", type=int, default=8)
    args = p.parse_args()

    rec = ChaosRecommender.load()
    team = rec.build_team(
        args.favorite, gen=args.gen, tier=args.tier, team_size=args.size,
        beam_width=args.beam,
    )
    print(f"== gen{team.gen} {team.tier.upper()} team built around "
          f"{team.favorite} (viability score: {team.viability_score:.3f}) ==\n")
    for slot in team.members:
        print(f"- {slot.name}  ({'/'.join(slot.types)}, {slot.role}, "
              f"usage {slot.usage*100:.1f}%)")
        print(f"    item={slot.item}  ability={slot.ability}")
        print(f"    spread={slot.spread}")
        print(f"    moves={slot.moves}")
        print(f"    why: {slot.reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
