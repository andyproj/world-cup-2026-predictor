import pandas as pd

from src.data_sources import _football_data_payload_to_frame
from src.pipeline import compare_predictions


def test_football_data_payload_parses_finished_match():
    payload = {
        "matches": [
            {
                "id": 123,
                "utcDate": "2026-06-12T01:00:00Z",
                "status": "FINISHED",
                "stage": "GROUP_STAGE",
                "group": "GROUP_A",
                "homeTeam": {"name": "United States"},
                "awayTeam": {"name": "Korea Republic"},
                "score": {"fullTime": {"home": 2, "away": 1}},
                "venue": "Test Stadium",
                "lastUpdated": "2026-06-12T03:00:00Z",
            }
        ]
    }
    frame = _football_data_payload_to_frame(payload)
    assert frame.iloc[0]["home_team"] == "USA"
    assert frame.iloc[0]["away_team"] == "South Korea"
    assert bool(frame.iloc[0]["is_finished"])
    assert frame.iloc[0]["group"] == "Group A"


def test_compare_predictions_reports_percentage_point_change():
    previous = pd.DataFrame(
        [
            {
                "fixture_key": "2026-06-20|A|B",
                "date": "2026-06-20",
                "home_team": "A",
                "away_team": "B",
                "home_win_probability": 0.50,
                "draw_probability": 0.25,
                "away_win_probability": 0.25,
                "predicted_outcome": "A",
            }
        ]
    )
    current = previous.copy()
    current.loc[0, "home_win_probability"] = 0.46
    current.loc[0, "away_win_probability"] = 0.29
    changes = compare_predictions(previous, current, "2026-06-08T00:00:00+00:00")
    assert len(changes) == 1
    assert changes.iloc[0]["home_change_pp"] == -4.0
    assert changes.iloc[0]["away_change_pp"] == 4.0
