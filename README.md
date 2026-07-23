# OKC Thunder Game Predictor

![Tests](https://github.com/JackHildebrand/ThunderWins/actions/workflows/tests.yml/badge.svg)

Machine learning models that predict Oklahoma City Thunder win probability and
point margin for upcoming games, trained on 7 seasons of live NBA.com data. Includes
a Streamlit web app with per-prediction SHAP explanations and a full backtest
writeup.

**Live demo:** https://tuhunderwins-ecw4tngqkqytiuuugmhpdi.streamlit.app/

## What this is

Two models sharing one feature set: a Random Forest **classifier** (win/loss) and a
Random Forest **regressor** (point margin, i.e. the spread), both evaluated with a
time-based backtest against a "predict the favorite" baseline. Rather than just
reporting a final accuracy number, this project documents the actual iterative
process -- what was tried, what worked, what didn't, and why -- including real
mistakes (data leakage, a corrupted data row, a feature-set change that silently
broke backtest stability) that were caught and fixed along the way, each with a
regression test.

### Results at a glance

Backtested by training on all prior seasons and testing on one held-out season at a
time (never randomly shuffled -- see Methodology below):

| Season | Win/loss accuracy | Baseline | Spread MAE (pts) | Baseline MAE |
|---|---|---|---|---|
| 2023-24 | 68.4% | 69.6% | 13.1 | 15.9 |
| 2024-25 | 46.2% | 82.1% | 15.8 | 16.6 |
| 2025-26 | 78.2% | 78.2% | 12.6 | 15.8 |

2024-25 is a known, diagnosed weak point (see below) -- included here rather than
hidden, since the diagnosis is more informative than the number itself. The spread
model beats its baseline in every season tested, including 2024-25.

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
  expected limitation (an event outside anything in the training data), not a bug.
  Adding OKC's own prior-season record and player-level continuity features
  measurably closed part of the gap, documented in the code and the app's Model
  Validation page.
- **Separate player lists for backtesting vs. production** -- the historical
  backtest uses whoever was actually on the roster each season (e.g. Luguentz
  Dort), while the live model uses the actual 2026-27 roster (Dort was swapped
  for Isaiah Hartenstein and Alex Caruso, who replaced him). Mixing the two
  caused a real accuracy collapse, caught and regression-tested.
- **Point-margin (spread) regression** alongside the win/loss classifier --
  evaluated with a separate baseline (predicting the training set's average
  margin) and more robust to the 2024-25 distribution shift than the classifier.
- **Per-prediction SHAP explanations** -- the app shows *why* a specific
  prediction came out the way it did (which factors pushed it up or down), not
  just which features matter in general.
- **Live injury awareness** -- NBA.com's stats API has no injury-report endpoint,
  so the app cross-references ESPN's unofficial injury feed to auto-detect
  confirmed-out players, with a manual override for anything it misses.
- **No live NBA.com dependency at prediction time** -- the deployed app initially
  broke with read-timeouts, because NBA.com occasionally blocks cloud/datacenter
  IPs. Root-caused and fixed: the "live" calls were re-fetching an already-complete
  season's static stats on every prediction. Precomputed and cached instead; only
  the genuinely time-sensitive ESPN injury check stays live.

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
| `weighting.py` | Recency-weighting logic, pulled out as a pure function for testability |
| `train_model.py` | Backtests Logistic Regression vs. Random Forest by season, including the spread regressor |
| `build_validation_results.py` | Precomputes backtest results for the web app |
| `build_live_data_cache.py` | Precomputes 2025-26 season stats/leading scorers, so live predictions don't depend on reaching NBA.com's stats API |
| `predict_upcoming.py` | Trains the final production model (classifier + spread regressor), predicts new games with SHAP explanations |
| `app.py` / `pages/1_Model_Validation.py` | Streamlit web app |
| `tests/` | pytest suite -- see Testing below |

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 explore_data.py               # pull raw data
python3 build_live_data_cache.py      # precompute current-season data for live predictions
python3 predict_upcoming.py           # train the final model
python3 build_validation_results.py   # precompute backtest results for the app
streamlit run app.py
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

29 tests, all pure/offline (no network calls, run in under a second) -- they cover
the anti-leakage logic directly (rolling averages excluding a game's own result,
resets across season boundaries), plus regression tests for real bugs caught during
development: a corrupted data row from a mislabeled API response, an accuracy
collapse from mixing up the backtest and production player lists, and a
pandas-version-dependent dtype mismatch caught by CI but not local testing. CI runs
the suite on every push via GitHub Actions (`.github/workflows/tests.yml`).

## Known limitations

- 2024-25 backtest accuracy is well below baseline -- documented distribution-shift
  finding, see the Model Validation page for detail.
- Live prediction features depend on ESPN's injury feed being reachable; it's
  unofficial/undocumented and could change without notice.
- The opponent-star and injury features track the *identified* leading scorer only,
  not full-roster depth.
