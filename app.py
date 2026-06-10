"""Local Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline
from src.tournament import run_tournament_simulation

OUTPUT = Path("output")
PREDICTIONS_PATH = OUTPUT / "latest_predictions.csv"
RATINGS_PATH = OUTPUT / "team_ratings.csv"
COMPLETED_PATH = OUTPUT / "completed_matches.csv"
CHANGES_PATH = OUTPUT / "prediction_changes.csv"
STATUS_PATH = OUTPUT / "model_status.json"
MARKET_STATUS_PATH = OUTPUT / "market_status.json"
TOURNAMENT_PATH = OUTPUT / "tournament_probabilities.csv"
SIMULATION_STATUS_PATH = OUTPUT / "simulation_status.json"

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.apply(lambda value: "—" if pd.isna(value) else f"{100 * value:.1f}%")


st.title("World Cup 2026 Predictor")
st.caption("Elo + recency-weighted scoring model, optional market-odds blend, and live-result monitoring.")

left, right = st.columns([1, 3])
with left:
    if st.button("Refresh data and predictions", type="primary"):
        with st.spinner("Downloading data, rebuilding predictions, and refreshing optional market odds..."):
            run_pipeline(force_refresh=True)
        st.success("Predictions refreshed.")
        st.rerun()

if not PREDICTIONS_PATH.exists():
    st.info("Run `python run.py` in the VS Code terminal first.")
    st.stop()

predictions = pd.read_csv(PREDICTIONS_PATH)
predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
status = _read_json(STATUS_PATH)
market_status = _read_json(MARKET_STATUS_PATH) or status.get("market_odds", {})

teams = sorted(
    set(predictions["home_team"].dropna())
    | set(predictions["away_team"].dropna())
)
groups = sorted(str(value) for value in predictions["group"].dropna().unique())

with st.sidebar:
    st.header("Filters")
    selected_team = st.selectbox("Team", ["All"] + teams)
    selected_group = st.selectbox("Group", ["All"] + groups)
    selected_confidence = st.selectbox("Confidence", ["All", "High", "Medium", "Low"])
    st.divider()
    st.caption("Automated local monitoring")
    st.code("python watch.py --interval 15", language="powershell")
    st.caption("Optional market odds")
    st.code("THE_ODDS_API_KEY=your_key", language="text")

filtered = predictions.copy()
if selected_team != "All":
    filtered = filtered.loc[
        (filtered["home_team"] == selected_team)
        | (filtered["away_team"] == selected_team)
    ]
if selected_group != "All":
    filtered = filtered.loc[filtered["group"] == selected_group]
if selected_confidence != "All":
    filtered = filtered.loc[filtered["confidence"] == selected_confidence]

metric1, metric2, metric3, metric4, metric5 = st.columns(5)
metric1.metric("Remaining fixtures", len(filtered))
metric2.metric("Completed matches", status.get("completed_2026_world_cup_matches_ingested", 0))
metric3.metric("Predictions changed", status.get("predictions_changed", 0))
metric4.metric("Market matches", market_status.get("fixtures_matched_to_predictions", 0))
metric5.metric("Model version", status.get("model_version", "—"))

if status:
    st.caption(
        f"Last model run: {status.get('generated_at_utc', '—')} UTC · "
        f"Data source: {', '.join(status.get('data_sources', [])) or '—'}"
    )

if market_status:
    if market_status.get("available"):
        st.success(
            "Automated market blend is active: "
            f"{market_status.get('fixtures_matched_to_predictions', 0)} fixtures matched. "
            f"Blend = {100 * market_status.get('model_weight', 0.65):.0f}% model / "
            f"{100 * market_status.get('market_weight_when_available', 0.35):.0f}% market."
        )
    else:
        st.info(
            "Market odds automation is not active. "
            f"Reason: {market_status.get('reason', 'No odds data matched yet.')}"
        )

prediction_tab, pool_tab, tournament_tab, completed_tab, change_tab, rating_tab = st.tabs(
    [
        "Predictions",
        "Automated pool picks",
        "Tournament odds",
        "Completed matches",
        "What changed",
        "Elo ratings",
    ]
)

with prediction_tab:
    base_columns = [
        "date",
        "ground",
        "group",
        "home_team",
        "away_team",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "predicted_outcome",
        "predicted_score",
        "confidence",
    ]
    optional_columns = [
        "market_home_win_probability",
        "market_draw_probability",
        "market_away_win_probability",
        "blended_home_win_probability",
        "blended_draw_probability",
        "blended_away_win_probability",
        "automated_pool_pick",
        "automated_pool_pick_probability",
    ]
    display_columns = [c for c in [*base_columns, *optional_columns] if c in filtered.columns]
    display = filtered[display_columns].copy()

    for column in [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "market_home_win_probability",
        "market_draw_probability",
        "market_away_win_probability",
        "blended_home_win_probability",
        "blended_draw_probability",
        "blended_away_win_probability",
        "automated_pool_pick_probability",
    ]:
        if column in display.columns:
            display[column] = _pct(display[column])

    display = display.rename(
        columns={
            "date": "Date",
            "ground": "Venue",
            "group": "Group",
            "home_team": "Team 1",
            "away_team": "Team 2",
            "home_win_probability": "Model: Team 1",
            "draw_probability": "Model: Draw",
            "away_win_probability": "Model: Team 2",
            "market_home_win_probability": "Market: Team 1",
            "market_draw_probability": "Market: Draw",
            "market_away_win_probability": "Market: Team 2",
            "blended_home_win_probability": "Blend: Team 1",
            "blended_draw_probability": "Blend: Draw",
            "blended_away_win_probability": "Blend: Team 2",
            "predicted_outcome": "Pure model pick",
            "automated_pool_pick": "Automated pool pick",
            "automated_pool_pick_probability": "Pool pick probability",
            "predicted_score": "Score",
            "confidence": "Confidence",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

with pool_tab:
    if "automated_pool_pick" not in filtered.columns:
        st.info("Run `python run.py --refresh` after installing the market-blend patch.")
    else:
        pool_cols = [
            "date",
            "ground",
            "group",
            "home_team",
            "away_team",
            "automated_pool_pick",
            "automated_pool_pick_probability",
            "market_match_found",
            "market_bookmaker_count",
            "predicted_score",
        ]
        pool_cols = [c for c in pool_cols if c in filtered.columns]
        pool = filtered[pool_cols].copy()
        if "automated_pool_pick_probability" in pool.columns:
            pool["automated_pool_pick_probability"] = _pct(pool["automated_pool_pick_probability"])
        pool = pool.rename(
            columns={
                "date": "Date",
                "ground": "Venue",
                "group": "Group",
                "home_team": "Team 1",
                "away_team": "Team 2",
                "automated_pool_pick": "Submit this pick",
                "automated_pool_pick_probability": "Pick probability",
                "market_match_found": "Market used",
                "market_bookmaker_count": "Bookmakers",
                "predicted_score": "Model score",
            }
        )
        st.dataframe(pool, use_container_width=True, hide_index=True)

        st.caption(
            "If market odds are unavailable for a fixture, the automated pool pick falls back to the model. "
            "If market odds are available, the pick uses the configured model/market blend."
        )

with tournament_tab:
    sim_left, sim_right = st.columns([1, 3])
    with sim_left:
        simulation_runs = st.selectbox("Simulation runs", [1000, 5000, 10000], index=1)
        if st.button("Run tournament simulation"):
            with st.spinner(f"Running {simulation_runs:,} tournaments..."):
                run_tournament_simulation(simulations=simulation_runs)
            st.success("Tournament probabilities updated.")
            st.rerun()

    if TOURNAMENT_PATH.exists():
        tournament = pd.read_csv(TOURNAMENT_PATH)
        probability_columns = [
            "win_group_probability",
            "round_of_32_probability",
            "round_of_16_probability",
            "quarterfinal_probability",
            "semifinal_probability",
            "final_probability",
            "champion_probability",
        ]
        display_tournament = tournament[["team", *probability_columns]].copy()
        for column in probability_columns:
            display_tournament[column] = _pct(display_tournament[column])
        display_tournament = display_tournament.rename(
            columns={
                "team": "Team",
                "win_group_probability": "Win group",
                "round_of_32_probability": "Reach R32",
                "round_of_16_probability": "Reach R16",
                "quarterfinal_probability": "Reach QF",
                "semifinal_probability": "Reach SF",
                "final_probability": "Reach final",
                "champion_probability": "Champion",
            }
        )
        st.dataframe(display_tournament, use_container_width=True, hide_index=True)

        simulation_status = _read_json(SIMULATION_STATUS_PATH)
        if simulation_status:
            st.caption(
                f"{simulation_status.get('simulations', 0):,} simulations · "
                f"Generated {simulation_status.get('generated_at_utc', '—')} UTC · "
                f"Third-place mapping: {simulation_status.get('third_place_mapping_source', '—')}"
            )
    else:
        st.info("Run `python simulate.py --runs 5000` to create tournament odds.")

with completed_tab:
    if COMPLETED_PATH.exists():
        completed = pd.read_csv(COMPLETED_PATH)
        if completed.empty:
            st.info("No World Cup matches have been completed yet.")
        else:
            columns = [
                name
                for name in [
                    "date",
                    "group",
                    "home_team",
                    "home_score",
                    "away_score",
                    "away_team",
                    "status",
                    "ground",
                    "source",
                ]
                if name in completed.columns
            ]
            st.dataframe(completed[columns], use_container_width=True, hide_index=True)
    else:
        st.info("No completed-match file exists yet.")

with change_tab:
    if CHANGES_PATH.exists():
        changes = pd.read_csv(CHANGES_PATH)
        if changes.empty:
            st.info("No material probability changes were detected in the latest run.")
        else:
            change_columns = [
                "date",
                "home_team",
                "away_team",
                "old_predicted_outcome",
                "new_predicted_outcome",
                "home_change_pp",
                "draw_change_pp",
                "away_change_pp",
                "largest_change_pp",
                "outcome_changed",
            ]
            change_columns = [c for c in change_columns if c in changes.columns]
            st.dataframe(changes[change_columns], use_container_width=True, hide_index=True)
    else:
        st.info("Run the model at least twice to compare prediction snapshots.")

with rating_tab:
    if RATINGS_PATH.exists():
        ratings = pd.read_csv(RATINGS_PATH).head(50)
        st.dataframe(ratings, use_container_width=True, hide_index=True)

st.warning(
    "This is an analytical model, not a guarantee or betting recommendation. "
    "Version 0.4 automatically blends market odds when THE_ODDS_API_KEY is configured. "
    "It still does not include automated injuries, confirmed lineups or xG."
)
