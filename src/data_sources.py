"""Free World Cup and historical-data clients with local caching."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .aliases import canonical_team

HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
WORLD_CUP_FIXTURES_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/"
    "master/2026/worldcup.json"
)
WORLD_CUP_TEXT_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup/"
    "master/2026--usa/cup.txt"
)
FOOTBALL_DATA_URL = (
    "https://api.football-data.org/v4/competitions/WC/matches?season=2026"
)

DATA_DIR = Path("data")
HISTORY_PATH = DATA_DIR / "historical_results.csv"
FIXTURES_PATH = DATA_DIR / "worldcup_2026.json"
FIXTURES_TEXT_PATH = DATA_DIR / "cup.txt"
FOOTBALL_DATA_CACHE_PATH = DATA_DIR / "football_data_worldcup_2026.json"
ENV_PATH = Path(".env")


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from .env without another dependency."""
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _download(url: str, destination: Path, timeout: int = 45) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "world-cup-2026-predictor/0.2"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def _cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
    return age_hours <= max_age_hours


def load_historical_results(force_refresh: bool = False) -> pd.DataFrame:
    """Load the public international-results dataset."""
    if force_refresh or not _cache_is_fresh(HISTORY_PATH, max_age_hours=24 * 7):
        try:
            _download(HISTORICAL_RESULTS_URL, HISTORY_PATH)
        except requests.RequestException:
            if not HISTORY_PATH.exists():
                raise

    frame = pd.read_csv(HISTORY_PATH)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical dataset is missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_team"] = frame["home_team"].map(canonical_team)
    frame["away_team"] = frame["away_team"].map(canonical_team)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def load_world_cup_schedule(force_refresh: bool = False) -> pd.DataFrame:
    """Load the best available World Cup schedule and result feed.

    If FOOTBALL_DATA_API_KEY exists in .env or the process environment, the
    football-data.org World Cup endpoint is used. Otherwise the project falls
    back to OpenFootball's public schedule repository.
    """
    _load_local_env()
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()

    if api_key:
        try:
            return _load_football_data_schedule(api_key, force_refresh)
        except (requests.RequestException, ValueError, OSError, json.JSONDecodeError):
            # Keep the dashboard usable if the optional API is unavailable.
            pass

    return _load_openfootball_schedule(force_refresh)


def _load_football_data_schedule(
    api_key: str,
    force_refresh: bool,
) -> pd.DataFrame:
    if force_refresh or not _cache_is_fresh(
        FOOTBALL_DATA_CACHE_PATH,
        max_age_hours=0.20,  # approximately 12 minutes
    ):
        response = requests.get(
            FOOTBALL_DATA_URL,
            headers={
                "X-Auth-Token": api_key,
                "User-Agent": "world-cup-2026-predictor/0.2",
            },
            timeout=45,
        )
        response.raise_for_status()
        FOOTBALL_DATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FOOTBALL_DATA_CACHE_PATH.write_text(response.text, encoding="utf-8")

    payload = json.loads(FOOTBALL_DATA_CACHE_PATH.read_text(encoding="utf-8"))
    frame = _football_data_payload_to_frame(payload)
    if len(frame) < 72:
        raise ValueError(
            "football-data.org returned fewer than 72 World Cup matches; "
            "falling back to OpenFootball."
        )
    return frame


def _football_data_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    matches = payload.get("matches", [])
    records: list[dict[str, Any]] = []

    for item in matches:
        score = item.get("score") or {}
        full_time = score.get("fullTime") or {}
        home_data = item.get("homeTeam") or {}
        away_data = item.get("awayTeam") or {}
        utc_date = pd.to_datetime(item.get("utcDate"), errors="coerce", utc=True)
        local_date = utc_date.tz_convert("America/Edmonton") if pd.notna(utc_date) else utc_date
        stage = str(item.get("stage") or "").replace("_", " ").title()
        group = _normalize_group(item.get("group"))

        records.append(
            {
                "match_id": item.get("id"),
                "round": stage,
                "date": local_date.date().isoformat() if pd.notna(local_date) else None,
                "time": local_date.strftime("%H:%M %Z") if pd.notna(local_date) else None,
                "home_team": canonical_team(home_data.get("name", "")),
                "away_team": canonical_team(away_data.get("name", "")),
                "group": group,
                "ground": item.get("venue"),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
                "status": item.get("status"),
                "source": "football-data.org",
                "last_updated": item.get("lastUpdated"),
            }
        )

    return _finalize_schedule_frame(pd.DataFrame(records))


def _normalize_group(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"([A-L])$", text)
    if match:
        return f"Group {match.group(1)}"
    return text.replace("_", " ").title()


def _load_openfootball_schedule(force_refresh: bool) -> pd.DataFrame:
    if force_refresh or not _cache_is_fresh(FIXTURES_PATH, max_age_hours=2):
        try:
            _download(WORLD_CUP_FIXTURES_URL, FIXTURES_PATH)
        except requests.RequestException:
            # Some corporate networks block raw GitHub JSON. Try the source text.
            if force_refresh or not _cache_is_fresh(
                FIXTURES_TEXT_PATH,
                max_age_hours=2,
            ):
                try:
                    _download(WORLD_CUP_TEXT_URL, FIXTURES_TEXT_PATH)
                except requests.RequestException:
                    pass

    if FIXTURES_PATH.exists():
        try:
            payload: dict[str, Any] = json.loads(
                FIXTURES_PATH.read_text(encoding="utf-8")
            )
            matches = payload.get("matches", [])
            if matches:
                return _schedule_records_to_frame(matches)
        except (json.JSONDecodeError, OSError):
            pass

    if FIXTURES_TEXT_PATH.exists():
        return _parse_football_txt(FIXTURES_TEXT_PATH)

    raise FileNotFoundError(
        "Could not download the World Cup schedule. Save cup.txt in the data "
        "folder from the openfootball/worldcup repository and run again."
    )


def _schedule_records_to_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_counter = 0
    for item in matches:
        round_name = str(item.get("round", ""))
        is_group = "Matchday" in round_name
        if is_group:
            group_counter += 1
            match_id = group_counter
        else:
            match_id = item.get("num")

        score = item.get("score") or {}
        full_time = score.get("ft") if isinstance(score, dict) else None
        score1 = item.get("score1")
        score2 = item.get("score2")
        if full_time and len(full_time) >= 2:
            score1, score2 = full_time[0], full_time[1]

        records.append(
            {
                "match_id": match_id,
                "round": round_name,
                "date": item.get("date"),
                "time": item.get("time"),
                "home_team": canonical_team(item.get("team1", "")),
                "away_team": canonical_team(item.get("team2", "")),
                "group": item.get("group"),
                "ground": item.get("ground"),
                "home_score": score1,
                "away_score": score2,
                "status": "FINISHED" if score1 is not None and score2 is not None else "SCHEDULED",
                "source": "OpenFootball",
                "last_updated": None,
            }
        )
    return _finalize_schedule_frame(pd.DataFrame(records))


def _parse_football_txt(path: Path) -> pd.DataFrame:
    """Parse group-stage fixtures from OpenFootball's simple text format."""
    group: str | None = None
    date_value: str | None = None
    records: list[dict[str, Any]] = []
    match_id = 0

    group_pattern = re.compile(r"^▪\s+Group\s+([A-L])$")
    date_pattern = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]+)\s+(\d{1,2})$"
    )
    match_pattern = re.compile(
        r"^\s*(\d{1,2}:\d{2}\s+UTC[+-]\d+)\s+(.+?)\s+v\s+(.+?)\s+@\s+(.+?)\s*$"
    )

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.rstrip()
        group_match = group_pattern.match(line.strip())
        if group_match:
            group = f"Group {group_match.group(1)}"
            continue

        date_match = date_pattern.match(line.strip())
        if date_match and group:
            month, day = date_match.groups()
            date_value = pd.Timestamp(f"2026 {month} {day}").date().isoformat()
            continue

        fixture_match = match_pattern.match(line)
        if fixture_match and group and date_value:
            match_id += 1
            time_value, home, away, ground = fixture_match.groups()
            records.append(
                {
                    "match_id": match_id,
                    "round": "Group Stage",
                    "date": date_value,
                    "time": time_value,
                    "home_team": canonical_team(home),
                    "away_team": canonical_team(away),
                    "group": group,
                    "ground": ground,
                    "home_score": None,
                    "away_score": None,
                    "status": "SCHEDULED",
                    "source": "OpenFootball text",
                    "last_updated": None,
                }
            )

    if len(records) < 72:
        raise ValueError(
            f"Expected at least 72 group fixtures in {path}, found {len(records)}."
        )
    return _finalize_schedule_frame(pd.DataFrame(records))


def _finalize_schedule_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {
        "match_id": None,
        "round": None,
        "date": None,
        "time": None,
        "home_team": "",
        "away_team": "",
        "group": None,
        "ground": None,
        "home_score": None,
        "away_score": None,
        "status": None,
        "source": None,
        "last_updated": None,
    }
    for column, default in expected_columns.items():
        if column not in frame.columns:
            frame[column] = default

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame["home_team"] = frame["home_team"].map(canonical_team)
    frame["away_team"] = frame["away_team"].map(canonical_team)
    frame["has_real_teams"] = ~(
        frame["home_team"].map(_is_placeholder)
        | frame["away_team"].map(_is_placeholder)
        | frame["home_team"].eq("")
        | frame["away_team"].eq("")
    )
    score_finished = frame["home_score"].notna() & frame["away_score"].notna()
    status_finished = frame["status"].astype(str).str.upper().eq("FINISHED")
    frame["is_finished"] = score_finished | status_finished
    frame["fixture_key"] = (
        frame["date"].dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + frame["home_team"]
        + "|"
        + frame["away_team"]
    )
    return frame.sort_values(["date", "match_id"], na_position="last").reset_index(drop=True)


def _is_placeholder(team: str) -> bool:
    """Identify bracket references such as 1A, W73 or 3A/B/C/D/F."""
    patterns = (
        r"^[12][A-L]$",
        r"^[WL]\d+$",
        r"^3[A-L](?:/[A-L])+$",
    )
    return any(re.match(pattern, team) for pattern in patterns)


def completed_schedule_matches(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Convert completed World Cup fixtures into the historical match schema."""
    completed = fixtures.loc[
        fixtures["is_finished"] & fixtures["has_real_teams"]
    ].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "neutral",
            ]
        )

    completed["tournament"] = "FIFA World Cup"
    completed["neutral"] = True
    return completed[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ]
    ]
