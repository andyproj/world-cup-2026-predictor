"""Monte Carlo simulation for the 48-team FIFA World Cup 2026 format."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .data_sources import (
    completed_schedule_matches,
    load_historical_results,
    load_world_cup_schedule,
)
from .elo import expected_score, train_elo
from .poisson_model import build_form_model, expected_goals

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
MAPPING_CACHE = DATA_DIR / "third_place_mapping.csv"
WIKIPEDIA_TABLE_URL = (
    "https://en.wikipedia.org/wiki/"
    "Template:2026_FIFA_World_Cup_third-place_table"
)
MODEL_VERSION = "0.3-tournament-simulation"
GROUPS = tuple("ABCDEFGHIJKL")
WINNER_SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
HOSTS = {"Canada", "Mexico", "USA"}

# Official round-of-32 structure from the FIFA World Cup 2026 regulations.
R32_FIXED = {
    73: ("2A", "2B"),
    74: ("1E", "third:1E"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "third:1I"),
    78: ("2E", "2I"),
    79: ("1A", "third:1A"),
    80: ("1L", "third:1L"),
    81: ("1D", "third:1D"),
    82: ("1G", "third:1G"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "third:1B"),
    86: ("1J", "2H"),
    87: ("1K", "third:1K"),
    88: ("2D", "2G"),
}
R16 = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}
QUARTERFINALS = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}
SEMIFINALS = {101: (97, 98), 102: (99, 100)}

# Used only if the exact Annex C table cannot be downloaded.
FALLBACK_PREFERENCES = {
    "1A": tuple("HECFI"),
    "1B": tuple("JGEIF"),
    "1D": tuple("BIJEF"),
    "1E": tuple("CDFAB"),
    "1G": tuple("AHJIE"),
    "1I": tuple("FGDHC"),
    "1K": tuple("LIEDJ"),
    "1L": tuple("KIEHJ"),
}
ALLOWED_THIRD_GROUPS = {
    "1A": set("CEFHI"),
    "1B": set("EFGIJ"),
    "1D": set("BEFIJ"),
    "1E": set("ABCDF"),
    "1G": set("AEHIJ"),
    "1I": set("CDFGH"),
    "1K": set("DEIJL"),
    "1L": set("EHIJK"),
}


@dataclass(frozen=True)
class SimulatedMatch:
    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass
class TeamTableStats:
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _clean_html_cell(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def parse_third_place_mapping_html(page_html: str) -> pd.DataFrame:
    """Parse the 495 Annex C combinations from the Wikipedia transcription."""
    rows: list[dict[str, str | int]] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", page_html, flags=re.I | re.S):
        cells = re.findall(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
            row_html,
            flags=re.I | re.S,
        )
        if not cells:
            continue
        cleaned = [_clean_html_cell(cell) for cell in cells]
        if not re.fullmatch(r"\d{1,3}", cleaned[0]):
            continue

        tokens: list[str] = []
        for cell in cleaned[1:]:
            tokens.extend(re.findall(r"(?<![A-Z0-9])(?:3[A-L]|[A-L])(?![A-Z0-9])", cell))
        if len(tokens) < 16:
            continue

        qualifiers = [token for token in tokens if re.fullmatch(r"[A-L]", token)][:8]
        assignments = [token for token in tokens if re.fullmatch(r"3[A-L]", token)][:8]
        if len(qualifiers) != 8 or len(assignments) != 8:
            continue

        record: dict[str, str | int] = {
            "option": int(cleaned[0]),
            "qualifying_groups": "".join(sorted(qualifiers)),
        }
        record.update(
            {
                slot: assignment[-1]
                for slot, assignment in zip(WINNER_SLOTS, assignments, strict=True)
            }
        )
        rows.append(record)

    frame = pd.DataFrame(rows)
    if len(frame) != 495 or frame["qualifying_groups"].nunique() != 495:
        raise ValueError(
            "Could not parse all 495 official third-place combinations "
            f"(parsed {len(frame)} rows)."
        )
    return frame.sort_values("option").reset_index(drop=True)


def load_third_place_mapping(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the exact FIFA Annex C mapping, cached locally after first use."""
    if MAPPING_CACHE.exists() and not force_refresh:
        frame = pd.read_csv(MAPPING_CACHE, dtype=str)
        if len(frame) == 495:
            return frame, "cached Annex C transcription"

    response = requests.get(
        WIKIPEDIA_TABLE_URL,
        timeout=45,
        headers={"User-Agent": "world-cup-2026-predictor/0.3"},
    )
    response.raise_for_status()
    frame = parse_third_place_mapping_html(response.text)
    MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MAPPING_CACHE, index=False)
    return frame.astype(str), "FIFA Annex C transcription via Wikipedia"


def _fallback_third_assignment(qualifying_groups: Iterable[str]) -> dict[str, str]:
    remaining = set(qualifying_groups)

    def search(index: int, assigned: dict[str, str]) -> dict[str, str] | None:
        if index == len(WINNER_SLOTS):
            return assigned.copy()
        slot = WINNER_SLOTS[index]
        preferred = [
            group
            for group in FALLBACK_PREFERENCES[slot]
            if group in remaining and group in ALLOWED_THIRD_GROUPS[slot]
        ]
        for group in preferred:
            remaining.remove(group)
            assigned[slot] = group
            result = search(index + 1, assigned)
            if result is not None:
                return result
            assigned.pop(slot, None)
            remaining.add(group)
        return None

    result = search(0, {})
    if result is None:
        raise ValueError("Could not construct a valid third-place bracket assignment.")
    return result


def _mapping_for_groups(
    qualifying_groups: Iterable[str],
    mapping: pd.DataFrame | None,
) -> dict[str, str]:
    key = "".join(sorted(qualifying_groups))
    if mapping is not None:
        matched = mapping.loc[mapping["qualifying_groups"] == key]
        if len(matched) == 1:
            row = matched.iloc[0]
            return {slot: str(row[slot]) for slot in WINNER_SLOTS}
    return _fallback_third_assignment(key)


def _group_letter(value: object) -> str:
    match = re.search(r"([A-L])$", str(value))
    if not match:
        raise ValueError(f"Invalid group value: {value!r}")
    return match.group(1)


def _host_for_pair(home: str, away: str) -> str | None:
    for team in (home, away):
        if team in HOSTS:
            return team
    return None


def _calculate_stats(teams: Iterable[str], matches: list[SimulatedMatch]) -> dict[str, TeamTableStats]:
    stats = {team: TeamTableStats() for team in teams}
    for match in matches:
        home = stats[match.home]
        away = stats[match.away]
        home.gf += match.home_goals
        home.ga += match.away_goals
        away.gf += match.away_goals
        away.ga += match.home_goals
        if match.home_goals > match.away_goals:
            home.points += 3
        elif match.home_goals < match.away_goals:
            away.points += 3
        else:
            home.points += 1
            away.points += 1
    return stats


def rank_group(
    teams: list[str],
    matches: list[SimulatedMatch],
    ratings: dict[str, float],
) -> tuple[list[str], dict[str, TeamTableStats]]:
    """Rank a group using FIFA's head-to-head-first structure.

    Fair-play and historic FIFA ranking are unavailable in the free feed. Elo is
    used only as the final, very rare tie-break proxy.
    """
    overall = _calculate_stats(teams, matches)
    by_points: dict[int, list[str]] = defaultdict(list)
    for team in teams:
        by_points[overall[team].points].append(team)

    ranked: list[str] = []
    for points in sorted(by_points, reverse=True):
        tied = by_points[points]
        if len(tied) == 1:
            ranked.extend(tied)
            continue
        mini_matches = [
            match
            for match in matches
            if match.home in tied and match.away in tied
        ]
        mini = _calculate_stats(tied, mini_matches)
        ranked.extend(
            sorted(
                tied,
                key=lambda team: (
                    mini[team].points,
                    mini[team].gd,
                    mini[team].gf,
                    overall[team].gd,
                    overall[team].gf,
                    ratings.get(team, 1500.0),
                ),
                reverse=True,
            )
        )
    return ranked, overall


def _sample_group_match(
    row: object,
    rng: np.random.Generator,
    rates: dict[tuple[str, str], tuple[float, float]],
) -> SimulatedMatch:
    if bool(row.is_finished):
        return SimulatedMatch(
            row.home_team,
            row.away_team,
            int(row.home_score),
            int(row.away_score),
        )
    lambda_home, lambda_away = rates[(row.home_team, row.away_team)]
    return SimulatedMatch(
        row.home_team,
        row.away_team,
        int(rng.poisson(lambda_home)),
        int(rng.poisson(lambda_away)),
    )


def _simulate_knockout_winner(
    home: str,
    away: str,
    rng: np.random.Generator,
    rates: dict[tuple[str, str], tuple[float, float]],
    ratings: dict[str, float],
) -> str:
    lambda_home, lambda_away = rates[(home, away)]
    home_goals = int(rng.poisson(lambda_home))
    away_goals = int(rng.poisson(lambda_away))
    if home_goals > away_goals:
        return home
    if away_goals > home_goals:
        return away

    # Approximate 30 minutes of extra time using one third of the 90-minute rates.
    home_extra = int(rng.poisson(lambda_home / 3.0))
    away_extra = int(rng.poisson(lambda_away / 3.0))
    if home_extra > away_extra:
        return home
    if away_extra > home_extra:
        return away

    host = _host_for_pair(home, away)
    adjustment = 55.0 if host == home else (-55.0 if host == away else 0.0)
    home_advance_probability = expected_score(
        ratings.get(home, 1500.0) + adjustment,
        ratings.get(away, 1500.0),
    )
    return home if rng.random() < home_advance_probability else away


def run_tournament_simulation(
    simulations: int = 5000,
    seed: int = 2026,
    force_refresh: bool = False,
    refresh_mapping: bool = False,
) -> pd.DataFrame:
    if simulations < 100:
        raise ValueError("Use at least 100 simulations.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_historical_results(force_refresh=force_refresh)
    fixtures = load_world_cup_schedule(force_refresh=force_refresh)
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

    group_fixtures = fixtures.loc[fixtures["group"].notna() & fixtures["has_real_teams"]].copy()
    if len(group_fixtures) != 72:
        raise ValueError(f"Expected 72 group-stage fixtures, found {len(group_fixtures)}.")

    teams = sorted(
        set(group_fixtures["home_team"]) | set(group_fixtures["away_team"])
    )
    if len(teams) != 48:
        raise ValueError(f"Expected 48 tournament teams, found {len(teams)}.")

    try:
        mapping, mapping_source = load_third_place_mapping(force_refresh=refresh_mapping)
    except Exception:
        mapping = None
        mapping_source = "valid fallback mapping (Annex C download unavailable)"

    rates: dict[tuple[str, str], tuple[float, float]] = {}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            rates[(home, away)] = expected_goals(
                home,
                away,
                form_model,
                home_rating=float(ratings.get(home, 1500.0)),
                away_rating=float(ratings.get(away, 1500.0)),
                host_team=_host_for_pair(home, away),
            )

    grouped_rows = {
        letter: list(
            group_fixtures.loc[
                group_fixtures["group"].map(_group_letter) == letter
            ].itertuples(index=False)
        )
        for letter in GROUPS
    }
    group_teams = {
        letter: sorted(
            {
                team
                for row in grouped_rows[letter]
                for team in (row.home_team, row.away_team)
            }
        )
        for letter in GROUPS
    }

    count_names = (
        "win_group",
        "finish_second",
        "qualify_as_third",
        "round_of_32",
        "round_of_16",
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
    )
    counts = {team: {name: 0 for name in count_names} for team in teams}
    rng = np.random.default_rng(seed)

    for _ in range(simulations):
        positions: dict[str, list[str]] = {}
        third_records: list[tuple[str, str, TeamTableStats]] = []

        for group in GROUPS:
            simulated_matches = [
                _sample_group_match(row, rng, rates)
                for row in grouped_rows[group]
            ]
            ranked, stats = rank_group(group_teams[group], simulated_matches, ratings)
            positions[group] = ranked
            counts[ranked[0]]["win_group"] += 1
            counts[ranked[1]]["finish_second"] += 1
            third_records.append((group, ranked[2], stats[ranked[2]]))

        third_records.sort(
            key=lambda item: (
                item[2].points,
                item[2].gd,
                item[2].gf,
                ratings.get(item[1], 1500.0),
            ),
            reverse=True,
        )
        qualified_thirds = third_records[:8]
        third_by_group = {group: team for group, team, _ in qualified_thirds}
        third_assignment = _mapping_for_groups(third_by_group, mapping)

        slot_team: dict[str, str] = {}
        for group in GROUPS:
            slot_team[f"1{group}"] = positions[group][0]
            slot_team[f"2{group}"] = positions[group][1]
        for group, team in third_by_group.items():
            counts[team]["qualify_as_third"] += 1

        qualified = set(slot_team.values()) | set(third_by_group.values())
        for team in qualified:
            counts[team]["round_of_32"] += 1

        winners: dict[int, str] = {}
        for match_id, (left, right) in R32_FIXED.items():
            home = slot_team[left]
            if right.startswith("third:"):
                winner_slot = right.split(":", 1)[1]
                away = third_by_group[third_assignment[winner_slot]]
            else:
                away = slot_team[right]
            winners[match_id] = _simulate_knockout_winner(
                home, away, rng, rates, ratings
            )
        for team in winners.values():
            counts[team]["round_of_16"] += 1

        for match_id, (left_id, right_id) in R16.items():
            winners[match_id] = _simulate_knockout_winner(
                winners[left_id], winners[right_id], rng, rates, ratings
            )
        for match_id in R16:
            counts[winners[match_id]]["quarterfinal"] += 1

        for match_id, (left_id, right_id) in QUARTERFINALS.items():
            winners[match_id] = _simulate_knockout_winner(
                winners[left_id], winners[right_id], rng, rates, ratings
            )
        for match_id in QUARTERFINALS:
            counts[winners[match_id]]["semifinal"] += 1

        for match_id, (left_id, right_id) in SEMIFINALS.items():
            winners[match_id] = _simulate_knockout_winner(
                winners[left_id], winners[right_id], rng, rates, ratings
            )
        for match_id in SEMIFINALS:
            counts[winners[match_id]]["final"] += 1

        champion = _simulate_knockout_winner(
            winners[101], winners[102], rng, rates, ratings
        )
        counts[champion]["champion"] += 1

    records: list[dict[str, object]] = []
    for team in teams:
        row: dict[str, object] = {
            "team": team,
            "elo_rating": round(float(ratings.get(team, 1500.0)), 1),
        }
        for name in count_names:
            row[f"{name}_probability"] = round(counts[team][name] / simulations, 4)
        records.append(row)

    probabilities = pd.DataFrame(records).sort_values(
        ["champion_probability", "final_probability", "semifinal_probability"],
        ascending=False,
    )
    probabilities.to_csv(OUTPUT_DIR / "tournament_probabilities.csv", index=False)
    (OUTPUT_DIR / "tournament_probabilities.json").write_text(
        probabilities.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    status = {
        "generated_at_utc": generated_at,
        "model_version": MODEL_VERSION,
        "simulations": simulations,
        "seed": seed,
        "third_place_mapping_source": mapping_source,
        "completed_world_cup_matches_fixed_in_simulation": int(
            group_fixtures["is_finished"].sum()
        ),
        "limitations": [
            "Fair-play card data is unavailable, so Elo is the final group tie-break proxy.",
            "Extra time is modelled at one third of regulation scoring rates.",
            "Penalty shoot-outs use Elo-adjusted advancement probability.",
            "Completed knockout matches will be fixed into the simulation in a later update.",
        ],
    }
    (OUTPUT_DIR / "simulation_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    return probabilities
