"""Run the full World Cup tournament simulation."""

from __future__ import annotations

import argparse

from src.tournament import run_tournament_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-mapping", action="store_true")
    args = parser.parse_args()

    results = run_tournament_simulation(
        simulations=args.runs,
        seed=args.seed,
        force_refresh=args.refresh,
        refresh_mapping=args.refresh_mapping,
    )
    print()
    print(f"Completed {args.runs:,} tournament simulations.")
    print("Top championship probabilities:")
    print(
        results[["team", "champion_probability", "final_probability"]]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
