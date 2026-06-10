"""Recency-weighted attack/defence model and scoreline probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class TeamStrength:
    attack: float
    defence: float
    matches: float


@dataclass
class FormModel:
    global_goals_per_team: float
    strengths: dict[str, TeamStrength]


def build_form_model(
    matches: pd.DataFrame,
    half_life_days: float = 730.0,
    prior_matches: float = 8.0,
    lookback_years: int = 8,
) -> FormModel:
    if matches.empty:
        return FormModel(1.25, {})

    latest_data_date = pd.Timestamp(matches["date"].max())
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    reference_date = max(latest_data_date, today)
    cutoff = reference_date - pd.DateOffset(years=lookback_years)
    recent = matches.loc[matches["date"] >= cutoff].copy()
    if recent.empty:
        recent = matches.tail(1000).copy()

    age_days = (reference_date - recent["date"]).dt.days.clip(lower=0)
    recent["weight"] = np.exp(-math.log(2.0) * age_days / half_life_days)

    total_weight = float(recent["weight"].sum())
    total_goals = float(
        ((recent["home_score"] + recent["away_score"]) * recent["weight"]).sum()
    )
    global_average = total_goals / max(2.0 * total_weight, 1.0)
    global_average = float(np.clip(global_average, 0.8, 2.0))

    accum: dict[str, dict[str, float]] = {}
    for row in recent.itertuples(index=False):
        weight = float(row.weight)
        for team, scored, conceded in (
            (row.home_team, row.home_score, row.away_score),
            (row.away_team, row.away_score, row.home_score),
        ):
            item = accum.setdefault(
                team,
                {"weight": 0.0, "scored": 0.0, "conceded": 0.0},
            )
            item["weight"] += weight
            item["scored"] += float(scored) * weight
            item["conceded"] += float(conceded) * weight

    strengths: dict[str, TeamStrength] = {}
    for team, item in accum.items():
        denominator = item["weight"] + prior_matches
        goals_for = (
            item["scored"] + prior_matches * global_average
        ) / denominator
        goals_against = (
            item["conceded"] + prior_matches * global_average
        ) / denominator

        strengths[team] = TeamStrength(
            attack=float(np.clip(goals_for / global_average, 0.45, 1.85)),
            defence=float(np.clip(goals_against / global_average, 0.45, 1.85)),
            matches=item["weight"],
        )

    return FormModel(global_average, strengths)


def expected_goals(
    home_team: str,
    away_team: str,
    model: FormModel,
    home_rating: float,
    away_rating: float,
    host_team: str | None = None,
) -> tuple[float, float]:
    """Estimate goals using Elo as the anchor and form as a small adjustment.

    Raw international goal rates can flatter teams that mostly face weaker
    regional opposition. Elo therefore determines most of the expected-goal
    ratio, while recent attack/defence form only nudges it.
    """
    home = model.strengths.get(home_team, TeamStrength(1.0, 1.0, 0.0))
    away = model.strengths.get(away_team, TeamStrength(1.0, 1.0, 0.0))

    host_elo_adjustment = 55.0 if host_team == home_team else 0.0
    if host_team == away_team:
        host_elo_adjustment = -55.0

    rating_difference = home_rating + host_elo_adjustment - away_rating
    elo_log_ratio = rating_difference / 430.0

    raw_form_ratio = (
        home.attack * away.defence
    ) / max(away.attack * home.defence, 0.05)
    form_log_ratio = 0.18 * math.log(float(np.clip(raw_form_ratio, 0.25, 4.0)))

    goal_ratio = math.exp(elo_log_ratio + form_log_ratio)
    goal_ratio = float(np.clip(goal_ratio, 0.18, 5.5))

    total_goals = float(np.clip(2.0 * model.global_goals_per_team, 2.20, 3.00))
    lambda_home = total_goals * goal_ratio / (1.0 + goal_ratio)
    lambda_away = total_goals / (1.0 + goal_ratio)

    return (
        float(np.clip(lambda_home, 0.20, 3.80)),
        float(np.clip(lambda_away, 0.20, 3.80)),
    )


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate**k) / math.factorial(k)


def poisson_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 8,
) -> tuple[float, float, float, int, int]:
    home_pmf = np.array(
        [_poisson_pmf(goal, lambda_home) for goal in range(max_goals + 1)]
    )
    away_pmf = np.array(
        [_poisson_pmf(goal, lambda_away) for goal in range(max_goals + 1)]
    )
    matrix = np.outer(home_pmf, away_pmf)
    matrix = matrix / matrix.sum()

    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())

    score_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    return home_win, draw, away_win, int(score_index[0]), int(score_index[1])


def most_likely_score_for_outcome(
    lambda_home: float,
    lambda_away: float,
    outcome: str,
    max_goals: int = 8,
) -> tuple[int, int]:
    """Return the highest-probability scoreline consistent with an outcome."""
    home_pmf = np.array(
        [_poisson_pmf(goal, lambda_home) for goal in range(max_goals + 1)]
    )
    away_pmf = np.array(
        [_poisson_pmf(goal, lambda_away) for goal in range(max_goals + 1)]
    )
    matrix = np.outer(home_pmf, away_pmf)

    if outcome == "home":
        mask = np.fromfunction(lambda i, j: i > j, matrix.shape, dtype=int)
    elif outcome == "away":
        mask = np.fromfunction(lambda i, j: i < j, matrix.shape, dtype=int)
    else:
        mask = np.eye(max_goals + 1, dtype=bool)

    masked = np.where(mask, matrix, -1.0)
    index = np.unravel_index(np.argmax(masked), masked.shape)
    return int(index[0]), int(index[1])
