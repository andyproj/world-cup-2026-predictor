# World Cup 2026 Predictor

A comprehensive prediction system for the 2026 FIFA World Cup, featuring Elo-based team ratings, Poisson model forecasting, Monte Carlo tournament simulation, and optional market odds integration.

## Overview

This project:

1. Downloads a free historical men's international-match dataset
2. Downloads the open 2026 World Cup schedule
3. Builds current team Elo ratings and attack/defense strengths
4. Predicts every World Cup fixture with match-level probabilities
5. Simulates complete tournament outcomes with team advancement probabilities
6. Optionally blends model predictions with live market odds
7. Monitors live match results and updates forecasts in real-time
8. Provides a local Streamlit dashboard with multiple analytical views

## Prerequisites

Install:

- Python 3.11 or 3.12
- Visual Studio Code
- The VS Code Python extension

Open this folder in VS Code.

## Setup

### Create the Python environment

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, try:

```powershell
python -m venv .venv
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), choose
**Python: Select Interpreter**, and select the interpreter inside `.venv`.

## Quick Start

### 1. Generate predictions

```bash
python run.py
```

To force fresh downloads:

```bash
python run.py --refresh
```

**Outputs:**
- `output/latest_predictions.csv` — Match-level predictions
- `output/latest_predictions.json` — Match predictions in JSON format
- `output/team_ratings.csv` — Current Elo ratings for all teams
- `output/model_status.json` — Model metadata
- `output/prediction_history.csv` — Historical prediction records

### 2. Generate tournament odds

```powershell
python simulate.py --runs 5000
```

For a faster test: `python simulate.py --runs 1000`
For more stable forecasts: `python simulate.py --runs 10000`

**Outputs:**
- `output/tournament_probabilities.csv` — Team advancement probabilities
- `output/tournament_probabilities.json` — Tournament odds in JSON
- `output/simulation_status.json` — Simulation metadata

### 3. Open the dashboard

```bash
streamlit run app.py
```

Your browser should open a local page, normally: `http://localhost:8501`

**Dashboard tabs:**
- **Predictions** — Match-level forecasts
- **Completed Matches** — Results and before/after analysis
- **Prediction Changes** — How forecasts shift after new results
- **Tournament Odds** — Team probabilities for each tournament stage
- **Automated Pool Picks** — Market-blended recommendations (if configured)

### 4. Run tests

```bash
pytest
```

## Live Match Monitoring

The monitor automatically detects newly completed matches and recalculates forecasts.

**One-time check:**

```powershell
python watch.py --once
```

**Continuous monitoring:**

```powershell
.\.venv\Scripts\Activate.ps1
python watch.py --interval 15
```

Keep this running in a second terminal. Press `Ctrl+C` to stop.

**Outputs:**
- `output/completed_matches.csv` — Match results and scoring
- `output/prediction_change_history.csv` — Impact of each result on forecasts

### Optional: Use football-data.org feed

The project uses OpenFootball by default. For a more purpose-built result feed:

1. Register for a free token at [football-data.org](https://football-data.org)
2. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Replace `replace_with_your_token` with your token
4. Restart the monitor

The `.env` file is excluded by `.gitignore` and only read locally.

## Market Odds Integration (Optional)

To blend model predictions with live betting odds:

1. Get an API key from [The Odds API](https://theoddsapi.com)
2. Add to `.env`:
   ```text
   THE_ODDS_API_KEY=your_key_here
   ```
3. Run:
   ```powershell
   python run.py --refresh
   python simulate.py --runs 5000
   streamlit run app.py
   ```

**Default blend:** 65% model + 35% market (when available)

The **Automated Pool Picks** tab appears when market data is active.

## Modelling Notes

- **Group ties** use official head-to-head rules; Elo replaces fair-play cards as the final tiebreaker
- **Extra time** uses one-third of regulation expected-goal rates
- **Penalty shootouts** use Elo-adjusted advancement probabilities
- **Third-place bracket** requires separate logic for the 48-team format (caches mapping on first run)

## Project Structure

```
├── app.py                     # Streamlit dashboard
├── run.py                     # Generate match predictions
├── simulate.py                # Tournament simulation
├── watch.py                   # Live result monitoring
├── requirements.txt
│
├── src/
│   ├── aliases.py            # Team name mappings
│   ├── elo.py                # Elo rating engine
│   ├── poisson_model.py       # Match probability model
│   ├── tournament.py          # Bracket logic and simulation
│   ├── market_odds.py         # Odds API integration
│   ├── data_sources.py        # Data fetching
│   └── pipeline.py            # Main orchestration
│
├── data/
│   ├── historical_results.csv
│   ├── worldcup_2026.json     # Schedule
│   └── third_place_mapping.csv
│
└── output/                    # Generated predictions and results
```

## No internet required (except on `--refresh`)

The project includes cached historical data and the 2026 schedule. Internet is only needed when forcing fresh downloads with `--refresh` or when using live monitoring and market odds features.

## How Version 0.1 works

### Elo component

Elo is updated chronologically after each historical result. The update size
depends on competition importance and goal margin. Old, dormant ratings slowly
move toward the global average.

### Poisson component

Recent goals scored and conceded are exponentially weighted. The resulting
attack and defence strengths produce expected goals and a scoreline
distribution.

### Ensemble

Final probabilities are:

```text
60% Poisson probabilities
40% Elo probabilities
```

Host adjustment is applied to Canada, Mexico and the USA.

## Important current limitations

- The historical public dataset may lag the latest international friendlies.
- Injuries and confirmed lineups are not yet included.
- There is no betting-market input.
- Match-result timing depends on the open schedule repository being updated.
- Pre-tournament knockout simulation is not yet implemented.
- The model is for analysis and entertainment, not a betting guarantee.

## Next implementation milestones

1. Add a football-data.org result feed as the primary live source.
2. Add prediction-change reporting after every completed match.
3. Implement the official 48-team group ranking and best-third-place mapping.
4. Run 10,000 tournament simulations.
5. Publish JSON through GitHub Pages.
6. Connect the published JSON to a Custom GPT Action.
