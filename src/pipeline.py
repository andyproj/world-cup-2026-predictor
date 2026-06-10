"""End-to-end prediction and change-tracking pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_sources import (
    completed_schedule_matches,
    load_historical_results,
    load_world_cup_schedule,
)
from .elo import elo_three_way_probabilities, train_elo
from .market_odds import enrich_predictions_with_market_odds
from .poisson_model import (
    build_form_model,
    expected_goals,
    most_likely_score_for_outcome,
    poisson_probabilities,
)

OUTPUT_DIR = Path("output")
PREDICTIONS_PATH = OUTPUT_DIR / "latest_predictions.csv"
STATUS_PATH = OUTPUT_DIR / "model_status.json"
HOSTS = {"Canada", "Mexico", "USA"}
MODEL_VERSION = "0.4-market-blend"


def _host_for_fixture(home: str, away: str) -> str | None:
    for team in (home, away):
        if team in HOSTS:
            return team
    return None


def _confidence(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.48:
        return "Medium"
    return "Low"


def _most_likely_outcome(
    home: str,
    away: str,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> str:
    values = {
        home: home_probability,
        "Draw": draw_probability,
        away: away_probability,
    }
    return max(values, key=values.get)


def _read_previous_status() -> dict[str, object]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def compare_predictions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    generated_at: str,
) -> pd.DataFrame:
    """Compare two prediction snapshots for the same remaining fixtures."""
    columns = [
        "fixture_key",
        "date",
        "home_team",
        "away_team",
        "old_home_win_probability",
        "new_home_win_probability",
        "home_change_pp",
        "old_draw_probability",
        "new_draw_probability",
        "draw_change_pp",
        "old_away_win_probability",
        "new_away_win_probability",
        "away_change_pp",
        "old_predicted_outcome",
        "new_predicted_outcome",
        "outcome_changed",
        "largest_change_pp",
        "generated_at_utc",
    ]
    if previous.empty or current.empty:
        return pd.DataFrame(columns=columns)

    probability_columns = [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]
    required = {"fixture_key", "date", "home_team", "away_team", "predicted_outcome", *probability_columns}
    if not required.issubset(previous.columns) or not required.issubset(current.columns):
        return pd.DataFrame(columns=columns)

    old = previous[
        ["fixture_key", "predicted_outcome", *probability_columns]
    ].copy()
    new = current[
        ["fixture_key", "date", "home_team", "away_team", "predicted_outcome", *probability_columns]
    ].copy()
    old = old.rename(
        columns={
            "predicted_outcome": "old_predicted_outcome",
            **{name: f"old_{name}" for name in probability_columns},
        }
    )
    new = new.rename(
        columns={
            "predicted_outcome": "new_predicted_outcome",
            **{name: f"new_{name}" for name in probability_columns},
        }
    )
    merged = new.merge(old, on="fixture_key", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    for prefix in ("home_win", "draw", "away_win"):
        merged[f"{prefix.replace('_win', '')}_change_pp" if prefix != "draw" else "draw_change_pp"] = 0.0

    merged["home_change_pp"] = 100 * (
        merged["new_home_win_probability"] - merged["old_home_win_probability"]
    )
    merged["draw_change_pp"] = 100 * (
        merged["new_draw_probability"] - merged["old_draw_probability"]
    )
    merged["away_change_pp"] = 100 * (
        merged["new_away_win_probability"] - merged["old_away_win_probability"]
    )
    merged["outcome_changed"] = (
        merged["new_predicted_outcome"] != merged["old_predicted_outcome"]
    )
    merged["largest_change_pp"] = merged[
        ["home_change_pp", "draw_change_pp", "away_change_pp"]
    ].abs().max(axis=1)
    merged["generated_at_utc"] = generated_at

    # Ignore numerical noise smaller than 0.05 percentage points.
    merged = merged.loc[
        (merged["largest_change_pp"] >= 0.05) | merged["outcome_changed"]
    ].copy()
    merged = merged.sort_values(
        ["outcome_changed", "largest_change_pp"],
        ascending=[False, False],
    )
    for column in ["home_change_pp", "draw_change_pp", "away_change_pp", "largest_change_pp"]:
        merged[column] = merged[column].round(2)
    return merged[columns]


def run_pipeline(force_refresh: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    previous_predictions = (
        pd.read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else pd.DataFrame()
    )
    previous_status = _read_previous_status()

    history = load_historical_results(force_refresh=force_refresh)
    fixtures = load_world_cup_schedule(force_refresh=force_refresh)

    completed = fixtures.loc[
        fixtures["is_finished"] & fixtures["has_real_teams"]
    ].copy()
    completed.to_csv(OUTPUT_DIR / "completed_matches.csv", index=False)

    live_results = completed_schedule_matches(fixtures)
    if not live_results.empty:
        history = pd.concat([history, live_results], ignore_index=True)
        history = history.drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="last",
        ).sort_values("date")

    training_history = history.loc[history["date"] >= "1990-01-01"].copy()
    ratings = train_elo(training_history)
    form_model = build_form_model(training_history)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    candidates = fixtures.loc[
        fixtures["has_real_teams"] & ~fixtures["is_finished"]
    ].copy()

    records: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = float(ratings.get(home, 1500.0))
        away_rating = float(ratings.get(away, 1500.0))
        host = _host_for_fixture(home, away)
        host_elo_adjustment = 55.0 if host == home else (-55.0 if host == away else 0.0)

        elo_home, elo_draw, elo_away = elo_three_way_probabilities(
            home_rating,
            away_rating,
            host_adjustment=host_elo_adjustment,
        )
        lambda_home, lambda_away = expected_goals(
            home,
            away,
            form_model,
            home_rating=home_rating,
            away_rating=away_rating,
            host_team=host,
        )
        poisson_home, poisson_draw, poisson_away, _, _ = poisson_probabilities(
            lambda_home,
            lambda_away,
        )

        home_probability = 0.60 * poisson_home + 0.40 * elo_home
        draw_probability = 0.60 * poisson_draw + 0.40 * elo_draw
        away_probability = 0.60 * poisson_away + 0.40 * elo_away
        total = home_probability + draw_probability + away_probability
        home_probability /= total
        draw_probability /= total
        away_probability /= total

        highest = max(home_probability, draw_probability, away_probability)
        if home_probability == highest:
            score_outcome = "home"
        elif away_probability == highest:
            score_outcome = "away"
        else:
            score_outcome = "draw"
        predicted_home_goals, predicted_away_goals = most_likely_score_for_outcome(
            lambda_home,
            lambda_away,
            score_outcome,
        )

        date_text = row.date.date().isoformat() if pd.notna(row.date) else None
        fixture_key = f"{date_text or ''}|{home}|{away}"
        records.append(
            {
                "fixture_key": fixture_key,
                "match_id": row.match_id,
                "date": date_text,
                "time": row.time,
                "round": row.round,
                "group": row.group,
                "ground": row.ground,
                "home_team": home,
                "away_team": away,
                "home_elo": round(home_rating, 1),
                "away_elo": round(away_rating, 1),
                "expected_home_goals": round(lambda_home, 2),
                "expected_away_goals": round(lambda_away, 2),
                "home_win_probability": round(home_probability, 4),
                "draw_probability": round(draw_probability, 4),
                "away_win_probability": round(away_probability, 4),
                "predicted_outcome": _most_likely_outcome(
                    home,
                    away,
                    home_probability,
                    draw_probability,
                    away_probability,
                ),
                "predicted_score": f"{predicted_home_goals}-{predicted_away_goals}",
                "confidence": _confidence(highest),
                "host_adjustment_applied_to": host,
                "data_source": getattr(row, "source", None),
                "generated_at_utc": generated_at,
                "model_version": MODEL_VERSION,
            }
        )

    predictions = pd.DataFrame(records)
    if not predictions.empty:
        predictions = predictions.sort_values(["date", "match_id"], na_position="last")

    market_status = {}
    if not predictions.empty:
        predictions, market_status = enrich_predictions_with_market_odds(
            predictions,
            output_dir=OUTPUT_DIR,
            force_refresh=force_refresh,
        )

    changes = compare_predictions(previous_predictions, predictions, generated_at)
    changes.to_csv(OUTPUT_DIR / "prediction_changes.csv", index=False)
    if not changes.empty:
        history_path = OUTPUT_DIR / "prediction_change_history.csv"
        prior_changes = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        pd.concat([prior_changes, changes], ignore_index=True).to_csv(history_path, index=False)

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    (OUTPUT_DIR / "latest_predictions.json").write_text(
        predictions.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    rating_frame = pd.DataFrame(
        sorted(ratings.items(), key=lambda item: item[1], reverse=True),
        columns=["team", "elo_rating"],
    )
    rating_frame["elo_rating"] = rating_frame["elo_rating"].round(1)
    rating_frame.to_csv(OUTPUT_DIR / "team_ratings.csv", index=False)

    completed_count = int(len(completed))
    previous_completed_count = int(
        previous_status.get("completed_2026_world_cup_matches_ingested", 0) or 0
    )
    new_completed_count = max(0, completed_count - previous_completed_count)
    sources = sorted(
        str(value) for value in fixtures["source"].dropna().unique().tolist()
    )
    status = {
        "generated_at_utc": generated_at,
        "model_version": MODEL_VERSION,
        "data_sources": sources,
        "historical_matches_used": int(len(training_history)),
        "completed_2026_world_cup_matches_ingested": completed_count,
        "new_completed_matches_detected": new_completed_count,
        "remaining_concrete_fixtures_predicted": int(len(predictions)),
        "predictions_changed": int(len(changes)),
        "predicted_outcomes_changed": int(changes["outcome_changed"].sum()) if not changes.empty else 0,
        "market_odds": market_status,
        "latest_training_match_date": (
            training_history["date"].max().date().isoformat()
            if not training_history.empty
            else None
        ),
        "limitations": [
            "Knockout placeholders are predicted after participants become known.",
            "No automated player injury, confirmed-lineup or xG feed is included yet.",
        ],
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    # Keep a full audit snapshot only when data meaningfully changed or on first run.
    history_path = OUTPUT_DIR / "prediction_history.csv"
    if not history_path.exists() or new_completed_count > 0 or not changes.empty:
        previous_history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        pd.concat([previous_history, predictions], ignore_index=True).to_csv(
            history_path,
            index=False,
        )

    return predictions
