"""
Synergy scoring system for Pokemon team building.
Provides data-driven synergy calculations based on type compatibility,
role complementarity, and competitive patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations


class PokemonSynergyScorer:
    """
    Calculates synergy scores between Pokemon for team building.
    Uses type effectiveness, role compatibility, and stat balance.
    """

    def __init__(self, pokemon_df: pd.DataFrame):
        """
        Initialize with Pokemon dataset.

        Args:
            pokemon_df: DataFrame with columns: name, type1, type2, hp, attack,
                       defense, sp_attack, sp_defense, speed, role, weaknesses
        """
        self.pokemon_df = pokemon_df.copy()
        self.type_effectiveness = self._load_type_effectiveness()

        # Cache: (type1, type2 or None) -> frozenset of attack types this
        # defender resists (multiplier < 1). There are at most 18*18 unique
        # combos so this fills quickly and lets us skip a hot inner loop that
        # otherwise runs millions of times per team build.
        self._resist_cache: Dict[Tuple[str, Optional[str]], frozenset] = {}

        # Role compatibility matrix (higher = more complementary)
        self.role_synergy = {
            'fast attacker': {
                'fast attacker': 0.3,
                'physical attacker': 0.8,
                'special attacker': 0.8,
                'bulky defender': 0.9,
                'defensive support': 0.7,
                'balanced': 0.6
            },
            'physical attacker': {
                'fast attacker': 0.8,
                'physical attacker': 0.4,
                'special attacker': 0.9,
                'bulky defender': 0.7,
                'defensive support': 0.6,
                'balanced': 0.7
            },
            'special attacker': {
                'fast attacker': 0.8,
                'physical attacker': 0.9,
                'special attacker': 0.4,
                'bulky defender': 0.7,
                'defensive support': 0.6,
                'balanced': 0.7
            },
            'bulky defender': {
                'fast attacker': 0.9,
                'physical attacker': 0.7,
                'special attacker': 0.7,
                'bulky defender': 0.5,
                'defensive support': 0.8,
                'balanced': 0.6
            },
            'defensive support': {
                'fast attacker': 0.7,
                'physical attacker': 0.6,
                'special attacker': 0.6,
                'bulky defender': 0.8,
                'defensive support': 0.5,
                'balanced': 0.7
            },
            'balanced': {
                'fast attacker': 0.6,
                'physical attacker': 0.7,
                'special attacker': 0.7,
                'bulky defender': 0.6,
                'defensive support': 0.7,
                'balanced': 0.5
            }
        }

    def _load_type_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Load the type effectiveness chart."""
        return {
            "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
            "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
            "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
            "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
            "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
            "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
            "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
            "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
            "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
            "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
            "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
            "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
            "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
            "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
            "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
            "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
            "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
            "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5}
        }

    def _resistances_for(self, type1: str, type2: Optional[str] = None) -> frozenset:
        """Cached set of attack types this defender resists (multiplier < 1).

        The first call for a given (type1, type2) pair iterates over all 18
        attacking types; subsequent calls hit the cache. With at most ~324
        unique type combos in the dataset, this cache stays tiny.
        """
        if type2 is None or (isinstance(type2, float) and type2 != type2) or type2 == "":
            type2 = None
        key = (type1, type2)
        cached = self._resist_cache.get(key)
        if cached is not None:
            return cached
        resistances = set()
        for atk_type in self.type_effectiveness.keys():
            if self.get_type_multiplier(atk_type, type1, type2) < 1:
                resistances.add(atk_type)
        frozen = frozenset(resistances)
        self._resist_cache[key] = frozen
        return frozen

    def calculate_type_synergy(self, pokemon1: Dict, pokemon2: Dict) -> float:
        """
        Calculate type synergy between two Pokemon.
        Higher score = better synergy (covering each other's weaknesses).
        """
        p1_weaknesses = set(pokemon1['weaknesses'])
        p2_weaknesses = set(pokemon2['weaknesses'])

        # Resistances pulled from the cache instead of recomputed each call.
        p1_resistances = self._resistances_for(pokemon1['type1'], pokemon1.get('type2'))
        p2_resistances = self._resistances_for(pokemon2['type1'], pokemon2.get('type2'))

        p1_covered = len(p1_weaknesses & p2_resistances)
        p2_covered = len(p2_weaknesses & p1_resistances)
        shared_resistances = len(p1_resistances & p2_resistances)

        coverage_score = (p1_covered + p2_covered) / max(1, len(p1_weaknesses) + len(p2_weaknesses))
        resistance_bonus = shared_resistances * 0.1

        return min(1.0, coverage_score + resistance_bonus)

    def calculate_role_synergy(self, role1: str, role2: str) -> float:
        """Calculate role synergy between two Pokemon roles."""
        return self.role_synergy.get(role1, {}).get(role2, 0.5)

    def calculate_stat_synergy(self, team_stats: List[Dict]) -> float:
        """
        Calculate how well the team's stats are balanced.
        Penalizes teams with extreme stat imbalances.
        """
        if len(team_stats) < 2:
            return 1.0

        stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        variances = []
        for stat in stat_names:
            values = [member.get(stat, 0) for member in team_stats]
            variances.append(np.var(values))
        avg_variance = sum(variances) / len(variances)
        return max(0, 1 - (avg_variance / 1000))

    def get_type_multiplier(self, attacking_type: str, defender_type1: str,
                           defender_type2: Optional[str] = None) -> float:
        """Calculate type effectiveness multiplier."""
        multiplier = self.type_effectiveness.get(attacking_type, {}).get(defender_type1, 1.0)
        if defender_type2:
            multiplier *= self.type_effectiveness.get(attacking_type, {}).get(defender_type2, 1.0)
        return multiplier

    def score_team_synergy(self, team: List[Dict]) -> Dict[str, float]:
        """Calculate overall synergy scores for a team."""
        if len(team) < 2:
            return {'type_synergy': 1.0, 'role_synergy': 1.0, 'stat_balance': 1.0, 'overall': 1.0}

        type_synergies = []
        role_synergies = []
        for p1, p2 in combinations(team, 2):
            type_synergies.append(self.calculate_type_synergy(p1, p2))
            role_synergies.append(self.calculate_role_synergy(p1['role'], p2['role']))

        avg_type_synergy = sum(type_synergies) / len(type_synergies)
        avg_role_synergy = sum(role_synergies) / len(role_synergies)
        stat_balance = self.calculate_stat_synergy(team)

        overall = (avg_type_synergy * 0.4 + avg_role_synergy * 0.3 + stat_balance * 0.3)
        return {
            'type_synergy': avg_type_synergy,
            'role_synergy': avg_role_synergy,
            'stat_balance': stat_balance,
            'overall': overall
        }

    def score_candidate_for_team(self, candidate: Dict, current_team: List[Dict]) -> Dict[str, float]:
        """Score how well a candidate Pokemon fits with the current team."""
        if not current_team:
            return {'type_synergy': 1.0, 'role_synergy': 1.0, 'weakness_coverage': 1.0, 'overall': 1.0}

        # Compute candidate resistances ONCE outside the per-member loop
        # (previously this was recomputed 5x per call -- pure waste).
        candidate_resistances = self._resistances_for(
            candidate['type1'], candidate.get('type2')
        )

        type_synergies = []
        role_synergies = []
        team_weaknesses = set()
        covered_weaknesses = 0

        for member in current_team:
            type_synergies.append(self.calculate_type_synergy(candidate, member))
            role_synergies.append(self.calculate_role_synergy(candidate['role'], member['role']))

            member_weaknesses = set(member['weaknesses'])
            team_weaknesses.update(member_weaknesses)
            covered_weaknesses += len(member_weaknesses & candidate_resistances)

        avg_type_synergy = sum(type_synergies) / len(type_synergies)
        avg_role_synergy = sum(role_synergies) / len(role_synergies)
        weakness_coverage = covered_weaknesses / max(1, len(team_weaknesses))

        overall = (avg_type_synergy * 0.4 + avg_role_synergy * 0.3 + weakness_coverage * 0.3)
        return {
            'type_synergy': avg_type_synergy,
            'role_synergy': avg_role_synergy,
            'weakness_coverage': weakness_coverage,
            'overall': overall
        }
