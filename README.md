# OKC Thunder Game Predictor

A machine learning model that predicts Oklahoma City Thunder win probability for
upcoming games, built from scratch on live NBA.com data. Includes a Streamlit web
app for interactive predictions and a full backtest/methodology writeup.

**Live demo:** https://tuhunderwins-ecw4tngqkqytiuuugmhpdi.streamlit.app/

## What this is

A Random Forest classifier trained on 7 seasons (2019-20 through 2025-26) of team,
opponent, and player-level statistics, evaluated with a time-based backtest against
a "predict the favorite" baseline. Rather than just reporting a final accuracy
number, this project documents the actual iterative process: what was tried, what
worked, what didn't, and why -- including real mistakes (data leakage, a corrupted
data row, a feature-set change that broke backtest stability) that were caught and
fixed along the way.

## Methodology highlights

- **Time-based train/test splits**, never random shuffling -- random splits would
  leak future games into training, a common mistake with time-series sports data.
- **Shifted rolling averages** for every "recent form" feature, so a game's own
  result never leaks into its own prediction.
- **Recency-weighted training** -- older seasons count less, since the roster
  changes meaningfully year to year.
- **Correlation-based feature trimming**, validated empirically rather than
  assumed. It measurably helped for team-level stats and measurably *hurt* for
  player-level stats -- both results were kept, not just the flattering one.
- **Distribution-shift diagnosis** -- the model badly underperforms in 2024-25, the
  season the Thunder jumped from a 69.6% to an 82.1% win rate. That's a real,
  expected limitation (an event outside anything in the training data), not a bug,
  and is documented as such in the app's Model Validation page.
- **Separate player lists for backtesting vs. production** -- the historical
  backtest uses whoever was actually on the roster each season (e.g. Luguentz
  Dort), while the live model uses the actual 2026-27 roster (Dort was swapped
  for Isaiah Hartenstein and Alex Caruso, who replaced him). Mixing the two
  caused a real accuracy collapse that's documented in the code.
- **Live injury awareness** -- NBA.com's stats API has no injury-report endpoint,
  so the app cross-references ESPN's unofficial injury feed to auto-detect
  confirmed-out players, with a manual override for anything it misses.

## Project structure

| File | Purpose |
|---|---|
| `explore_data.py` | Pulls OKC's raw game log from NBA.com (`LeagueGameFinder`) |
| `features.py` | Builds leakage-safe rolling-average features |
| `opponent_strength.py` | Opponent (and OKC's own) prior-season strength |
| `opponent_defense.py` | What OKC's defense allowed, per game |
| `player_stats.py` | Rolling form + played/injured status for tracked OKC players |
| `opponent_stars.py` | Identifies and tracks each opponent's leading scorer |
| `live_injuries.py` | Live injury-report cross-reference (ESPN, unofficial) |
| `dataset.py` | Assembles the full engineered dataset from all of the above |
| `train_model.py` | Backtests Logistic Regression vs. Random Forest by season |
| `build_validation_results.py` | Precomputes backtest results for the web app |
| `predict_upcoming.py` | Trains the final production model, predicts new games |
| `app.py` / `pages/1_Model_Validation.py` | Streamlit web app |

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 explore_data.py               # pull raw data
python3 predict_upcoming.py           # train the final model
python3 build_validation_results.py   # precompute backtest results for the app
streamlit run app.py
```

## Known limitations

- 2024-25 backtest accuracy is well below baseline -- documented distribution-shift
  finding, see the Model Validation page for detail.
- Live prediction features depend on NBA.com's and ESPN's APIs being reachable and
  unblocked; both are unofficial/undocumented and could change without notice.
- The opponent-star and injury features track the *identified* leading scorer only,
  not full-roster depth.
