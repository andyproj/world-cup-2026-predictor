"""Run the World Cup prediction pipeline from the command line."""

from __future__ import annotations

import argparse

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh downloads instead of using the local cache.",
    )
    args = parser.parse_args()

    predictions = run_pipeline(force_refresh=args.refresh)
    print()
    print(f"Generated {len(predictions)} predictions.")
    print("Files written to the output folder.")
    print()
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "predicted_outcome",
        "predicted_score",
    ]
    print(predictions[columns].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
