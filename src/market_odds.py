"""Optional automated market-odds integration.

Uses The Odds API when THE_ODDS_API_KEY is present in .env or the process
environment. The dashboard remains fully usable without an odds key.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .aliases import canonical_team

API_ROOT = "https://api.the-odds-api.com/v4"
ENV_PATH = Path(".env")
DATA_DIR = Path("data")
SPORTS_CACHE_PATH = DATA_DIR / "the_odds_api_sports.json"
ODDS_CACHE_PATH = DATA_DIR / "the_odds_api_worldcup_odds.json"
MARKET_STATUS_PATH = Path("output") / "market_status.json"
MARKET_MATCH_PATH = Path("output") / "market_odds_matches.csv"


@dataclass
class OddsConfig:
    model_weight: float = 0.65
    market_weight: float = 0.35
    regions: str = "us,uk,eu,au"
    markets: str = "h2h"
    odds_format: str = "decimal"
    cache_minutes: int = 45


def _load_local_env() -> None:
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


def _cache_is_fresh(path: Path, max_age_minutes: float) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - modified).total_seconds() / 60
    return age_minutes <= max_age_minutes


def _clean_team(name: object) -> str:
    value = canonical_team(name).lower()
    value = re.sub(r"\b(men|w|women|national team|fc|sc)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _clean_team(left), _clean_team(right)).ratio()


def _http_get_json(url: str, params: dict[str, Any], destination: Path | None = None) -> dict[str, Any] | list[Any]:
    response = requests.get(
        url,
        params=params,
        timeout=45,
        headers={"User-Agent": "world-cup-2026-predictor/0.4"},
    )
    response.raise_for_status()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(response.text, encoding="utf-8")
    return response.json()


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_world_cup_sport_keys(api_key: str, force_refresh: bool) -> list[str]:
    if force_refresh or not _cache_is_fresh(SPORTS_CACHE_PATH, max_age_minutes=24 * 60):
        payload = _http_get_json(f"{API_ROOT}/sports", {"apiKey": api_key}, SPORTS_CACHE_PATH)
    else:
        payload = _read_json(SPORTS_CACHE_PATH)

    candidates: list[str] = []
    for item in payload:
        key = str(item.get("key", ""))
        title = str(item.get("title", ""))
        group = str(item.get("group", ""))
        haystack = f"{key} {title} {group}".lower()
        if "soccer" not in haystack:
            continue
        if "world cup" in haystack or "fifa" in haystack or "world_cup" in haystack:
            # Prefer event/match markets, not championship/futures sport keys.
            if "winner" not in haystack and "championship" not in haystack:
                candidates.append(key)

    # Most likely keys first if available.
    preferred_order = [
        "soccer_fifa_world_cup",
        "soccer_fifa_world_cup_2026",
        "soccer_world_cup",
    ]
    ordered = [key for key in preferred_order if key in candidates]
    ordered += [key for key in candidates if key not in ordered]
    return ordered


def _extract_event_probabilities(event: dict[str, Any]) -> dict[str, Any] | None:
    home = canonical_team(event.get("home_team", ""))
    away = canonical_team(event.get("away_team", ""))
    if not home or not away:
        return None

    per_book: list[dict[str, float]] = []
    for bookmaker in event.get("bookmakers", []) or []:
        market = next(
            (m for m in bookmaker.get("markets", []) or [] if m.get("key") == "h2h"),
            None,
        )
        if not market:
            continue
        raw: dict[str, float] = {}
        for outcome in market.get("outcomes", []) or []:
            name = canonical_team(outcome.get("name", ""))
            price = outcome.get("price")
            if not price or float(price) <= 1.0:
                continue
            if _clean_team(name) == "draw":
                raw["draw"] = 1.0 / float(price)
            elif _similarity(name, home) >= 0.78:
                raw["home"] = 1.0 / float(price)
            elif _similarity(name, away) >= 0.78:
                raw["away"] = 1.0 / float(price)
        if {"home", "draw", "away"}.issubset(raw):
            total = raw["home"] + raw["draw"] + raw["away"]
            if total > 0:
                per_book.append({key: raw[key] / total for key in ("home", "draw", "away")})

    if not per_book:
        return None

    return {
        "market_event_id": event.get("id"),
        "market_sport_key": event.get("sport_key"),
        "market_commence_time": event.get("commence_time"),
        "market_home_team": home,
        "market_away_team": away,
        "market_home_win_probability": sum(item["home"] for item in per_book) / len(per_book),
        "market_draw_probability": sum(item["draw"] for item in per_book) / len(per_book),
        "market_away_win_probability": sum(item["away"] for item in per_book) / len(per_book),
        "market_bookmaker_count": len(per_book),
    }


def fetch_world_cup_market_odds(force_refresh: bool = False, config: OddsConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch World Cup 1X2 market odds and return vig-free probabilities."""
    config = config or OddsConfig()
    _load_local_env()
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if not api_key:
        return pd.DataFrame(), {
            "enabled": False,
            "available": False,
            "source": "The Odds API",
            "reason": "THE_ODDS_API_KEY not found in .env or environment.",
            "generated_at_utc": generated_at,
        }

    try:
        sport_keys = _find_world_cup_sport_keys(api_key, force_refresh=force_refresh)
        if not sport_keys:
            return pd.DataFrame(), {
                "enabled": True,
                "available": False,
                "source": "The Odds API",
                "reason": "No active FIFA/World Cup soccer sport key found.",
                "generated_at_utc": generated_at,
            }

        payloads: list[dict[str, Any]] = []
        for sport_key in sport_keys[:3]:
            cache_path = DATA_DIR / f"the_odds_api_{sport_key}.json"
            if force_refresh or not _cache_is_fresh(cache_path, config.cache_minutes):
                events = _http_get_json(
                    f"{API_ROOT}/sports/{sport_key}/odds",
                    {
                        "apiKey": api_key,
                        "regions": config.regions,
                        "markets": config.markets,
                        "oddsFormat": config.odds_format,
                    },
                    cache_path,
                )
            else:
                events = _read_json(cache_path)
            for event in events:
                event["sport_key"] = sport_key
                payloads.append(event)

        records = [record for event in payloads if (record := _extract_event_probabilities(event))]
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=["market_home_team", "market_away_team", "market_commence_time"],
                keep="first",
            )
            frame.to_csv(MARKET_MATCH_PATH, index=False)

        return frame, {
            "enabled": True,
            "available": not frame.empty,
            "source": "The Odds API",
            "sport_keys_used": sport_keys[:3],
            "events_returned": len(payloads),
            "matches_with_1x2_probabilities": int(len(frame)),
            "generated_at_utc": generated_at,
        }
    except Exception as exc:  # keep the model usable if optional odds fail
        return pd.DataFrame(), {
            "enabled": True,
            "available": False,
            "source": "The Odds API",
            "reason": f"Odds refresh failed: {type(exc).__name__}: {exc}",
            "generated_at_utc": generated_at,
        }


def _match_market_row(prediction_row: pd.Series, market: pd.DataFrame) -> pd.Series | None:
    if market.empty:
        return None
    home = prediction_row["home_team"]
    away = prediction_row["away_team"]
    candidates = market.copy()
    candidates["same_order_score"] = candidates.apply(
        lambda r: (_similarity(home, r["market_home_team"]) + _similarity(away, r["market_away_team"])) / 2,
        axis=1,
    )
    candidates["reverse_order_score"] = candidates.apply(
        lambda r: (_similarity(home, r["market_away_team"]) + _similarity(away, r["market_home_team"])) / 2,
        axis=1,
    )
    best = candidates.sort_values(
        ["same_order_score", "reverse_order_score", "market_bookmaker_count"],
        ascending=[False, False, False],
    ).iloc[0]
    if max(best["same_order_score"], best["reverse_order_score"]) < 0.82:
        return None
    return best


def enrich_predictions_with_market_odds(
    predictions: pd.DataFrame,
    output_dir: Path | str = "output",
    force_refresh: bool = False,
    config: OddsConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add market and blended probabilities to the prediction table."""
    config = config or OddsConfig()
    output_dir = Path(output_dir)
    market, status = fetch_world_cup_market_odds(force_refresh=force_refresh, config=config)
    enriched = predictions.copy()

    # Always create these columns so the app has a stable schema.
    columns_defaults: dict[str, object] = {
        "market_home_win_probability": pd.NA,
        "market_draw_probability": pd.NA,
        "market_away_win_probability": pd.NA,
        "market_bookmaker_count": 0,
        "market_match_found": False,
        "blended_home_win_probability": pd.NA,
        "blended_draw_probability": pd.NA,
        "blended_away_win_probability": pd.NA,
        "automated_pool_pick": pd.NA,
        "automated_pool_pick_probability": pd.NA,
        "market_blend_weight": 0.0,
    }
    for column, default in columns_defaults.items():
        enriched[column] = default

    matched = 0
    for idx, row in enriched.iterrows():
        market_row = _match_market_row(row, market)
        if market_row is None:
            # No market match: blended equals the pure model.
            home = row["home_win_probability"]
            draw = row["draw_probability"]
            away = row["away_win_probability"]
            blend_weight = 0.0
        else:
            matched += 1
            same_order = market_row["same_order_score"] >= market_row["reverse_order_score"]
            if same_order:
                m_home = float(market_row["market_home_win_probability"])
                m_draw = float(market_row["market_draw_probability"])
                m_away = float(market_row["market_away_win_probability"])
            else:
                m_home = float(market_row["market_away_win_probability"])
                m_draw = float(market_row["market_draw_probability"])
                m_away = float(market_row["market_home_win_probability"])

            enriched.at[idx, "market_home_win_probability"] = m_home
            enriched.at[idx, "market_draw_probability"] = m_draw
            enriched.at[idx, "market_away_win_probability"] = m_away
            enriched.at[idx, "market_bookmaker_count"] = int(market_row["market_bookmaker_count"])
            enriched.at[idx, "market_match_found"] = True

            home = config.model_weight * float(row["home_win_probability"]) + config.market_weight * m_home
            draw = config.model_weight * float(row["draw_probability"]) + config.market_weight * m_draw
            away = config.model_weight * float(row["away_win_probability"]) + config.market_weight * m_away
            blend_weight = config.market_weight

        total = float(home + draw + away)
        if total <= 0:
            continue
        home, draw, away = home / total, draw / total, away / total
        enriched.at[idx, "blended_home_win_probability"] = round(home, 4)
        enriched.at[idx, "blended_draw_probability"] = round(draw, 4)
        enriched.at[idx, "blended_away_win_probability"] = round(away, 4)
        options = {
            row["home_team"]: home,
            "Draw": draw,
            row["away_team"]: away,
        }
        pick = max(options, key=options.get)
        enriched.at[idx, "automated_pool_pick"] = pick
        enriched.at[idx, "automated_pool_pick_probability"] = round(options[pick], 4)
        enriched.at[idx, "market_blend_weight"] = blend_weight

    status["fixtures_matched_to_predictions"] = matched
    status["model_weight"] = config.model_weight
    status["market_weight_when_available"] = config.market_weight
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "market_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return enriched, status
