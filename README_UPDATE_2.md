# Update 2 - Full Tournament Simulation

This update adds a Monte Carlo simulation of the complete 48-team World Cup.

## Install

Stop Streamlit with `Ctrl+C`, then extract the update ZIP into the project root with `-Force`.
No new Python packages are required.

## Run tests

```powershell
python -m pytest
```

## Generate tournament odds

```powershell
python simulate.py --runs 5000
```

For a faster test:

```powershell
python simulate.py --runs 1000
```

For a more stable final forecast:

```powershell
python simulate.py --runs 10000
```

The first run downloads and caches the exact 495-case third-place mapping table.
If that page is blocked, the simulator uses a valid fallback bracket assignment and
labels that clearly in `output/simulation_status.json`.

## Outputs

```text
output/tournament_probabilities.csv
output/tournament_probabilities.json
output/simulation_status.json
```

## Dashboard

Restart Streamlit:

```powershell
streamlit run app.py
```

Open the new **Tournament odds** tab. It shows each team's probability of:

- winning its group;
- reaching the Round of 32;
- reaching the Round of 16;
- reaching the quarterfinals;
- reaching the semifinals;
- reaching the final;
- winning the World Cup.

## Live updates

The monitor now reruns 5,000 simulations whenever it detects a new or corrected
match result:

```powershell
python watch.py --interval 15
```

## Current modelling limitations

- Group ties use the official head-to-head-first structure, but Elo replaces fair-play
  cards and historic FIFA ranking as the final rare tie-break.
- Extra time uses one third of the regulation expected-goal rates.
- Penalty shoot-outs use Elo-adjusted advancement probabilities.
- Completed knockout matches are not yet fixed into future simulations; that is the
  next tournament-stage enhancement.
