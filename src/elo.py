"""International-football Elo rating model."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    scale: float = 400.0
    ordinary_home_advantage: float = 55.0
    mean_reversion_years: float = 8.0


def competition_k(tournament: str) -> float:
    value = str(tournament).lower()
    if "fifa world cup" in value and "qualification" not in value:
        return 55.0
    if "world cup qualification" in value:
        return 38.0
    if any(
        label in value
        for label in (
            "uefa euro",
            "copa américa",
            "copa america",
            "african cup",
            "asian cup",
            "gold cup",
            "nations cup",
        )
    ):
        return 42.0
    if "nations league" in value:
        return 30.0
    if "friendly" in value:
        return 16.0
    return 26.0


def expected_score(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / scale))


def _actual_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _goal_multiplier(goal_difference: int) -> float:
    if goal_difference <= 1:
        return 1.0
    return min(2.0, 1.0 + 0.35 * math.log(goal_difference))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def train_elo(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> dict[str, float]:
    config = config or EloConfig()
    ratings: defaultdict[str, float] = defaultdict(lambda: config.initial_rating)
    last_seen: dict[str, pd.Timestamp] = {}

    for row in matches.sort_values("date").itertuples(index=False):
        home = row.home_team
        away = row.away_team
        match_date = pd.Timestamp(row.date)

        # Gradually pull dormant teams toward the global mean.
        for team in (home, away):
            if team in last_seen:
                years = max(
                    0.0,
                    (match_date - last_seen[team]).days / 365.25,
                )
                decay = math.exp(-years / config.mean_reversion_years)
                ratings[team] = (
                    config.initial_rating
                    + (ratings[team] - config.initial_rating) * decay
                )

        neutral = _as_bool(getattr(row, "neutral", False))
        home_bonus = 0.0 if neutral else config.ordinary_home_advantage
        home_rating_for_match = ratings[home] + home_bonus
        expected_home = expected_score(
            home_rating_for_match,
            ratings[away],
            config.scale,
        )
        actual_home = _actual_score(int(row.home_score), int(row.away_score))
        k = competition_k(getattr(row, "tournament", ""))
        multiplier = _goal_multiplier(abs(int(row.home_score) - int(row.away_score)))
        change = k * multiplier * (actual_home - expected_home)

        ratings[home] += change
        ratings[away] -= change
        last_seen[home] = match_date
        last_seen[away] = match_date

    return dict(ratings)


def elo_three_way_probabilities(
    home_rating: float,
    away_rating: float,
    host_adjustment: float = 0.0,
) -> tuple[float, float, float]:
    """Approximate home/draw/away probabilities from Elo ratings."""
    difference = home_rating + host_adjustment - away_rating
    draw_probability = 0.28 * math.exp(-abs(difference) / 650.0)
    decisive_home = expected_score(
        home_rating + host_adjustment,
        away_rating,
    )
    home_probability = (1.0 - draw_probability) * decisive_home
    away_probability = (1.0 - draw_probability) * (1.0 - decisive_home)
    return home_probability, draw_probability, away_probability
