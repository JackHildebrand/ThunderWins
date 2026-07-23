from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error

from dataset import build_dataset, BACKTEST_PLAYERS
from weighting import recency_weights

# Experiment: does dropping the oldest, least-relevant seasons help?
# 2019-20/2020-21 were the post-Westbrook rebuild -- SGA is the only player
# from that era still on the roster today. Recency weighting already
# down-weights them heavily, but toggle this to test dropping them outright.
EXCLUDE_SEASONS = []  # tried excluding ['2019-20', '2020-21'] -- made things
                       # worse on average (2023-24 collapsed since its
                       # training set shrank a lot). Recency weighting
                       # already handles "old seasons matter less" better
                       # than dropping them outright.

# Backtesting uses BACKTEST_PLAYERS (whoever was actually, continuously on
# the roster through our whole historical span) rather than the current
# 2026-27 roster -- see dataset.py for why mixing the two caused 2023-24
# accuracy to collapse.
df, seasons_all, stats_table, feature_cols = build_dataset(BACKTEST_PLAYERS, exclude_seasons=EXCLUDE_SEASONS)


def evaluate(model_builder, test_seasons):
    results = []
    for test_season in test_seasons:
        train_mask = df['SEASON'] < test_season
        test_mask = df['SEASON'] == test_season
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'WIN']
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'WIN']
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        weights = recency_weights(df.loc[train_mask, 'SEASON'], test_season, seasons_all)
        model = model_builder()
        model.fit(X_train, y_train, **({'logisticregression__sample_weight': weights}
                                        if hasattr(model, 'named_steps') else {'sample_weight': weights}))
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        baseline = max(y_test.mean(), 1 - y_test.mean())
        results.append((test_season, acc, baseline))
    return results


def evaluate_spread(test_seasons, decay=0.6):
    """Backtest a point-margin (spread) regressor the same way evaluate()
    backtests the win/loss classifier: time-based split, recency-weighted
    training. Baseline here is 'always predict the training set's average
    margin' -- the regression equivalent of 'always predict the favorite'."""
    results = []
    for test_season in test_seasons:
        train_mask = df['SEASON'] < test_season
        test_mask = df['SEASON'] == test_season
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'PLUS_MINUS']
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'PLUS_MINUS']
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        weights = recency_weights(df.loc[train_mask, 'SEASON'], test_season, seasons_all)
        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
        results.append((test_season, mae, baseline_mae))
    return results


test_seasons = [s for s in seasons_all[-3:] if s in df['SEASON'].unique()]

if __name__ == "__main__":
    print("=== Logistic Regression (recency-weighted) ===")
    for season, acc, baseline in evaluate(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        test_seasons
    ):
        print(f"{season}: accuracy={acc:.1%}  baseline={baseline:.1%}")

    print("\n=== Random Forest (recency-weighted) ===")
    for season, acc, baseline in evaluate(
        lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
        test_seasons
    ):
        print(f"{season}: accuracy={acc:.1%}  baseline={baseline:.1%}")

    print("\n=== Spread (point margin) Regressor ===")
    for season, mae, baseline_mae in evaluate_spread(test_seasons):
        print(f"{season}: MAE={mae:.1f} pts  baseline MAE={baseline_mae:.1f} pts")
