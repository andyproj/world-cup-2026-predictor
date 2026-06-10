# Update 1 — Live result monitoring

This update adds:

- optional football-data.org live World Cup results;
- a local monitor that checks for newly completed matches;
- automatic model recalculation after a new or corrected score;
- completed-match storage;
- before/after probability comparison;
- new dashboard tabs for completed matches and prediction changes.

## Apply the update

Extract the update ZIP directly into the existing project folder and overwrite
files when prompted. Do not delete `.venv`, `data`, or `output`.

With the virtual environment active:

```powershell
python -m pip install -r requirements.txt
pytest
python run.py --refresh
python watch.py --once
```

For continuous local monitoring, open a second VS Code terminal, activate the
same environment, and run:

```powershell
.\.venv\Scripts\Activate.ps1
python watch.py --interval 15
```

Keep that terminal open. Press `Ctrl+C` to stop it.

## Optional: use the free football-data.org feed

The project works without a key using OpenFootball. For a more purpose-built
result feed:

1. Register for a free token at football-data.org.
2. Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

3. Open `.env` and replace `replace_with_your_token` with the token.
4. Restart the monitor.

The token is read locally and `.env` is excluded by `.gitignore`.
