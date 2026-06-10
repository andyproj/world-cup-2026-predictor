"""Monitor the World Cup result feed and rebuild predictions after new results."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.data_sources import load_world_cup_schedule
from src.pipeline import run_pipeline
from src.tournament import run_tournament_simulation

STATE_PATH = Path("output/monitor_state.json")


def _completed_results() -> dict[str, str]:
    fixtures = load_world_cup_schedule(force_refresh=True)
    finished = fixtures.loc[
        fixtures["is_finished"] & fixtures["has_real_teams"]
    ]
    return {
        str(row.fixture_key): f"{int(row.home_score)}-{int(row.away_score)}"
        for row in finished.itertuples(index=False)
    }


def _load_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload.get("completed_results", {}).items()}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(results: dict[str, str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "checked_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                "completed_results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def check_once() -> bool:
    previous = _load_state()
    current = _completed_results()
    changed = {
        key: score
        for key, score in current.items()
        if key not in previous or previous[key] != score
    }

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    if not STATE_PATH.exists():
        print(f"[{timestamp}] Initializing monitor and building current predictions.")
        run_pipeline(force_refresh=False)
        run_tournament_simulation(simulations=5000)
        _save_state(current)
        print(f"[{timestamp}] Monitor initialized with {len(current)} completed matches.")
        return True

    if changed:
        print(f"[{timestamp}] Detected {len(changed)} new or corrected result(s):")
        for fixture, score in changed.items():
            print(f"  {fixture} -> {score}")
        run_pipeline(force_refresh=False)
        run_tournament_simulation(simulations=5000)
        _save_state(current)
        print(f"[{timestamp}] Predictions and tournament odds rebuilt.")
        return True

    _save_state(current)
    print(f"[{timestamp}] No new completed matches. Completed total: {len(current)}.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Minutes between checks. Default: 15.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit.",
    )
    args = parser.parse_args()

    if args.interval < 5 and not args.once:
        raise SystemExit("Use an interval of at least 5 minutes to avoid wasteful polling.")

    if args.once:
        check_once()
        return

    print(
        f"Monitoring World Cup results every {args.interval:g} minutes. "
        "Press Ctrl+C to stop."
    )
    try:
        while True:
            try:
                check_once()
            except Exception as exc:  # Keep a long-running local monitor alive.
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                print(f"[{timestamp}] Check failed: {exc}")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
