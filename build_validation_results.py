"""Precompute backtest results and feature importance for the Model
Validation page, so the deployed app doesn't need to re-run the full
backtest (and its ~35 live NBA.com API calls) on every cold start.
Re-run this after any change to the model or feature set."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import train_model as tm

rf_results = tm.evaluate(
    lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
    tm.test_seasons
)

model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(tm.df[tm.feature_cols], tm.df['WIN'])
importances = pd.Series(model.feature_importances_, index=tm.feature_cols).sort_values(ascending=False)

joblib.dump({'rf_results': rf_results, 'importances': importances}, 'validation_results.joblib')
print("Saved validation_results.joblib")
for season, acc, baseline in rf_results:
    print(f"  {season}: accuracy={acc:.1%}  baseline={baseline:.1%}")
