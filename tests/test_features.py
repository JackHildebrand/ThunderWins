import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from features import add_features


def make_games(n, season='2023-24', start_date='2023-10-20', matchups=None, pts=None):
    """Build a minimal synthetic game log with every column add_features()
    needs, so tests don't depend on real pulled data."""
    dates = pd.date_range(start_date, periods=n, freq='2D')
    matchups = matchups or ['OKC vs. DAL'] * n
    pts = pts or [100 + i for i in range(n)]
    return pd.DataFrame({
        'GAME_DATE': dates,
        'SEASON': [season] * n,
        'MATCHUP': matchups,
        'WL': ['W' if i % 2 == 0 else 'L' for i in range(n)],
        'PTS': pts,
        'PLUS_MINUS': [5.0] * n,
        'FG_PCT': [0.45] * n,
        'FG3_PCT': [0.35] * n,
        'REB': [40] * n,
        'AST': [25] * n,
        'STL': [8] * n,
        'BLK': [5] * n,
        'TOV': [12] * n,
        'OPP_FG_PCT': [0.44] * n,
        'OPP_FG3_PCT': [0.34] * n,
    })


def test_win_derived_from_wl():
    df = make_games(4)
    df.loc[0, 'WL'] = 'W'
    df.loc[1, 'WL'] = 'L'
    result = add_features(df)
    assert result.loc[0, 'WIN'] == 1
    assert result.loc[1, 'WIN'] == 0


def test_home_parsed_from_matchup():
    df = make_games(2, matchups=['OKC vs. DAL', 'OKC @ DAL'])
    result = add_features(df)
    assert result.loc[0, 'HOME'] == 1  # "vs." = home
    assert result.loc[1, 'HOME'] == 0  # "@" = away


def test_opponent_parsed_from_matchup():
    df = make_games(2, matchups=['OKC vs. DAL', 'OKC @ SAS'])
    result = add_features(df)
    assert result.loc[0, 'OPPONENT'] == 'DAL'
    assert result.loc[1, 'OPPONENT'] == 'SAS'


def test_rolling_average_excludes_current_game():
    """The core anti-leakage property: a game's own stats must never
    appear in its own rolling-average features."""
    df = make_games(5, pts=[100, 110, 120, 130, 140])
    result = add_features(df)
    # Row 3 (4th game, PTS=130) should roll over games 0-2 only (100,110,120),
    # NOT include its own PTS=130.
    assert result.loc[3, 'ROLL_PTS'] == pd.Series([100, 110, 120]).mean()


def test_rolling_average_resets_per_season():
    """Rolling stats shouldn't carry over across a season boundary --
    last year's roster isn't necessarily this year's roster."""
    season_a = make_games(5, season='2022-23', pts=[100, 100, 100, 100, 100])
    season_b = make_games(5, season='2023-24', start_date='2023-10-20',
                           pts=[200, 200, 200, 200, 200])
    df = pd.concat([season_a, season_b], ignore_index=True)
    result = add_features(df)
    # 4th game of season_b is the first with enough same-season history
    # (min_periods=3) to produce a value at all -- it should be 200
    # (season_b's own PTS), not any blend with season_a's 100s.
    fourth_season_b_row = result[result['SEASON'] == '2023-24'].iloc[3]
    assert fourth_season_b_row['ROLL_PTS'] == 200.0


def test_rolling_average_nan_before_min_periods():
    """Not enough prior games yet (min_periods=3) should be NaN, not a
    misleadingly confident average from 1-2 games."""
    df = make_games(2, pts=[100, 110])
    result = add_features(df)
    assert pd.isna(result.loc[1, 'ROLL_PTS'])


def test_rest_days_computed_from_game_dates():
    df = make_games(3, start_date='2023-10-20')  # 2-day spacing
    result = add_features(df)
    assert result.loc[1, 'REST_DAYS'] == 2
    assert pd.isna(result.loc[0, 'REST_DAYS'])  # no prior game yet


def test_rest_days_resets_per_season():
    season_a = make_games(2, season='2022-23', start_date='2023-04-01')
    season_b = make_games(2, season='2023-24', start_date='2023-10-20')
    df = pd.concat([season_a, season_b], ignore_index=True)
    result = add_features(df)
    first_season_b_row = result[result['SEASON'] == '2023-24'].iloc[0]
    # Should NOT be the ~200-day gap between the two seasons.
    assert pd.isna(first_season_b_row['REST_DAYS'])
