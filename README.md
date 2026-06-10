# World Cup 2026 Predictor — Lean MVP

This starter project:

1. downloads a free historical men's international-match dataset;
2. downloads the open 2026 World Cup schedule;
3. builds current team Elo ratings;
4. builds recency-weighted attack and defence strengths;
5. predicts every currently known team-versus-team World Cup fixture;
6. writes CSV and JSON outputs;
7. provides a local Streamlit dashboard.

The first version covers the 72 group-stage fixtures before the tournament.
Knockout fixtures are included automatically once their participants are known
in the schedule feed. A complete pre-tournament knockout simulation is the next
module, because the 48-team third-place qualification rules require separate
bracket logic.

## 1. Prerequisites

Install:

- Python 3.11 or 3.12
- Visual Studio Code
- The VS Code Python extension

Open this folder in VS Code.

## 2. Create the Python environment

### Windows PowerShell

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

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), choose
**Python: Select Interpreter**, and select the interpreter inside `.venv`.

## 3. Generate predictions

```bash
python run.py
```

To force fresh downloads:

```bash
python run.py --refresh
```

Expected outputs:

```text
output/latest_predictions.csv
output/latest_predictions.json
output/team_ratings.csv
output/model_status.json
output/prediction_history.csv
```

The starter archive already contains a current historical-results snapshot and the 2026 group schedule. Internet access is only required when you force a refresh.

## 4. Open the dashboard

```bash
streamlit run app.py
```

Your browser should open a local page, normally:

```text
http://localhost:8501
```

## 5. Run tests

```bash
pytest
```

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
