import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from nba_api.stats.static import teams as nba_teams

from dataset import build_dataset, PRODUCTION_PLAYERS
from opponent_strength import get_season_team_stats
from opponent_stars import get_season_leading_scorers_with_names
from player_stats import get_player_id
from live_injuries import get_current_injuries, players_out as espn_players_out
from nba_api.stats.endpoints import playergamelog

MODEL_FILE = 'thunder_model.joblib'
PLAYER_LABELS = [label for _, label in PRODUCTION_PLAYERS]
TEAM_ABBR_TO_NICKNAME = {t['abbreviation']: t['nickname'] for t in nba_teams.get_teams()}


def train_final_model(decay=0.6):
    """Train on ALL available history using the CURRENT roster's tracked
    players (PRODUCTION_PLAYERS), weighted so the most recently completed
    season counts most -- this is the model that actually predicts 2026-27
    games, so there's no held-out test season here. Trains both the
    win/loss classifier and a point-margin (spread) regressor."""
    df, seasons_all, stats_table, feature_cols = build_dataset(PRODUCTION_PLAYERS)
    reference_idx = len(seasons_all)  # one step beyond the last known season
    weights = df['SEASON'].map(lambda s: decay ** (reference_idx - seasons_all.index(s)))

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(df[feature_cols], df['WIN'], sample_weight=weights)

    spread_model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    spread_model.fit(df[feature_cols], df['PLUS_MINUS'], sample_weight=weights)

    joblib.dump({
        'model': model,
        'spread_model': spread_model,
        'df': df,
        'seasons_all': seasons_all,
        'feature_cols': feature_cols,
    }, MODEL_FILE)
    print(f"Trained on {len(df)} games across {len(seasons_all)} seasons. Saved to {MODEL_FILE}.")
    return model, spread_model, df, feature_cols


def _recent_scoring_avg(player_id, season='2025-26'):
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if len(log) == 0:
        return 0.0
    recent = log.sort_values('Game_ID')['PTS'].tail(10).tolist()
    return sum(recent) / len(recent) if recent else 0.0


def predict_game(opponent_abbr, is_home, players_out=None, opp_star_out=None):
    """Predict OKC's win probability (and point margin) for an upcoming
    2026-27 game. Returns a dict:
      win_prob          -- probability OKC wins (0-1)
      predicted_margin  -- predicted point margin, positive = OKC wins by that much
      contributions     -- pandas Series of SHAP values per feature, toward
                            the win prediction, sorted by absolute impact

    opponent_abbr: 3-letter team code, e.g. 'LAL'
    is_home: True if OKC is hosting
    players_out: tracked labels (see PLAYER_LABELS) to manually force OUT,
        on top of whatever ESPN's live injury feed auto-detects. Use this
        for anything the feed misses or gets wrong -- it's undocumented
        and best-effort, not guaranteed accurate.
    opp_star_out: True/False to manually override whether the opponent's
        leading scorer is out; leave as None to auto-detect from the feed.
    """
    saved = joblib.load(MODEL_FILE)
    model = saved['model']
    spread_model = saved['spread_model']
    df, feature_cols = saved['df'], saved['feature_cols']
    manual_out = set(players_out or [])

    all_injuries = get_current_injuries()

    # Auto-detect which tracked OKC players are listed as literally "Out".
    okc_nickname = TEAM_ABBR_TO_NICKNAME['OKC']
    tracked_names = {label: full_name for full_name, label in PRODUCTION_PLAYERS}
    auto_out_names = espn_players_out(all_injuries, okc_nickname, tracked_names.values())
    auto_out_labels = {label for label, name in tracked_names.items() if name in auto_out_names}
    final_players_out = auto_out_labels | manual_out
    if auto_out_labels:
        print(f"Auto-detected OUT from ESPN injury report: {sorted(auto_out_labels)}")

    last_row = df.sort_values('GAME_DATE').iloc[-1]
    median_rest = df['REST_DAYS'].median()

    # "Prior season" for a 2026-27 game is 2025-26, which is now complete.
    season_2526 = get_season_team_stats('2025-26')
    opp_stats = season_2526.get(opponent_abbr, {})
    team_stats = season_2526.get('OKC', {})

    features = {
        'HOME': int(is_home),
        'REST_DAYS': median_rest,  # season-opener/early-season rest isn't
                                    # meaningfully comparable to in-season
                                    # rest gaps, so fall back to a typical value
        'ROLL_PLUS_MINUS': last_row['ROLL_PLUS_MINUS'],
        'ROLL_FG3_PCT': last_row['ROLL_FG3_PCT'],
        'ROLL_REB': last_row['ROLL_REB'],
        'ROLL_STL': last_row['ROLL_STL'],
        'ROLL_BLK': last_row['ROLL_BLK'],
        'ROLL_TOV': last_row['ROLL_TOV'],
        'ROLL_OPP_FG_PCT': last_row['ROLL_OPP_FG_PCT'],
        'ROLL_OPP_FG3_PCT': last_row['ROLL_OPP_FG3_PCT'],
        'OPP_PRIOR_POINT_DIFF': opp_stats.get('POINT_DIFF', 0.0),
        'TEAM_PRIOR_WIN_PCT': team_stats.get('WIN_PCT', 0.5),
        'TEAM_PRIOR_POINT_DIFF': team_stats.get('POINT_DIFF', 0.0),
    }

    for full_name, label in PRODUCTION_PLAYERS:
        features[f'{label}_PLAYED'] = 0 if label in final_players_out else 1
        features[f'{label}_ROLL_PTS'] = 0.0 if label in final_players_out else last_row[f'{label}_ROLL_PTS']

    opp_star_id, opp_star_name = get_season_leading_scorers_with_names('2025-26').get(opponent_abbr, (None, None))

    if opp_star_out is None and opp_star_name is not None:
        opp_nickname = TEAM_ABBR_TO_NICKNAME.get(opponent_abbr, opponent_abbr)
        opp_star_out = bool(espn_players_out(all_injuries, opp_nickname, [opp_star_name]))
        if opp_star_out:
            print(f"Auto-detected opponent star OUT: {opp_star_name}")
    opp_star_out = bool(opp_star_out)

    opp_star_roll_pts = 0.0
    if opp_star_id is not None and not opp_star_out:
        opp_star_roll_pts = _recent_scoring_avg(opp_star_id)
    features['OPP_STAR_PLAYED'] = 0 if opp_star_out else 1
    features['OPP_STAR_ROLL_PTS'] = opp_star_roll_pts

    X = pd.DataFrame([features])[feature_cols]
    win_prob = model.predict_proba(X)[0][1]
    predicted_margin = spread_model.predict(X)[0]

    # SHAP explains THIS specific prediction: how much each feature pushed
    # the win probability up or down from the model's average prediction,
    # not just which features matter in general (that's the feature
    # importance chart on the Model Validation page -- this is per-game).
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    contributions = pd.Series(shap_values[0, :, 1], index=feature_cols)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    return {
        'win_prob': win_prob,
        'predicted_margin': predicted_margin,
        'contributions': contributions,
    }


if __name__ == "__main__":
    train_final_model()

    print("\nExample predictions (illustrative -- run once the real 2026-27 "
          "schedule is out):")
    r1 = predict_game('UTA', is_home=True)
    print(f"  Home vs. UTA, full strength:   win={r1['win_prob']:.1%}  margin={r1['predicted_margin']:+.1f}")
    r2 = predict_game('DEN', is_home=False)
    print(f"  Away vs. DEN, full strength:   win={r2['win_prob']:.1%}  margin={r2['predicted_margin']:+.1f}")
    r3 = predict_game('DEN', is_home=False, players_out=['SGA'])
    print(f"  Away vs. DEN, SGA out:         win={r3['win_prob']:.1%}  margin={r3['predicted_margin']:+.1f}")
    print("\nTop factors for the last prediction:")
    print(r3['contributions'].head(5))
