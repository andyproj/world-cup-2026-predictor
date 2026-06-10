# Update 4 — Automated market-odds blend

This update adds an optional market-odds automation layer using The Odds API.

If `THE_ODDS_API_KEY` is present in `.env`, the pipeline:

1. finds the active FIFA / World Cup soccer sport key;
2. downloads 1X2 match odds;
3. converts decimal odds to vig-free implied probabilities;
4. matches market events to World Cup fixtures;
5. blends model probabilities with market probabilities;
6. adds an **Automated pool picks** tab to the dashboard.

Default blend:

```text
65% model
35% market when a market match is found
100% model when no market match is found
```

No API key is required for the app to run. Without a key, the app simply uses the pure model.

## Setup

Add this to `.env`:

```text
THE_ODDS_API_KEY=your_key_here
```

Then run:

```powershell
python run.py --refresh
python simulate.py --runs 5000
streamlit run app.py
```

The dashboard will show whether the market blend is active and how many fixtures were matched.
