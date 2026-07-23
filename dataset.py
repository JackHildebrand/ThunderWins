import os
import pandas as pd

from features import add_features
from opponent_strength import build_prior_season_stats_table
from player_stats import build_player_game_log, add_player_features
from opponent_defense import build_opponent_defense_table, add_opponent_defense_raw
from opponent_stars import add_opponent_star_features

BASE_CACHE_FILE = 'thunder_base_cache.csv'

TEAM_FEATURE_COLS = ['HOME', 'REST_DAYS', 'ROLL_PLUS_MINUS', 'ROLL_FG3_PCT',
                      'ROLL_REB', 'ROLL_STL', 'ROLL_BLK', 'ROLL_TOV', 'OPP_PRIOR_POINT_DIFF',
                      'ROLL_OPP_FG_PCT', 'ROLL_OPP_FG3_PCT',
                      'TEAM_PRIOR_WIN_PCT', 'TEAM_PRIOR_POINT_DIFF',
                      'OPP_STAR_PLAYED', 'OPP_STAR_ROLL_PTS']

# Players on the roster continuously across our whole historical dataset
# (2019-20 through 2025-26) -- used for BACKTESTING, since that validates
# the modeling approach against real history.
BACKTEST_PLAYERS = [
    ('Shai Gilgeous-Alexander', 'SGA'),
    ('Jalen Williams', 'JDUB'),
    ('Chet Holmgren', 'CHET'),
    ('Luguentz Dort', 'DORT'),
]

# Players actually on the 2026-27 roster (checked via commonteamroster).
# Dort is no longer on the team, so he's swapped for Hartenstein and Caruso
# here -- this is the list the FINAL production model uses, since it's what
# will actually be true when 2026-27 games are played. Backtesting and
# production intentionally use different player lists: mixing them made
# 2023-24 backtest accuracy collapse, since Hartenstein/Caruso hadn't joined
# yet that season either, so those features carried zero information for a
# season where Dort's did.
PRODUCTION_PLAYERS = [
    ('Shai Gilgeous-Alexander', 'SGA'),
    ('Jalen Williams', 'JDUB'),
    ('Chet Holmgren', 'CHET'),
    ('Isaiah Hartenstein', 'HART'),
    ('Alex Caruso', 'CARUSO'),
]


def feature_cols_for(players):
    cols = list(TEAM_FEATURE_COLS)
    for _, label in players:
        cols += [f'{label}_PLAYED', f'{label}_ROLL_PTS']
    return cols


def build_dataset(players, exclude_seasons=None):
    """Build the full engineered dataset for a given list of (full_name,
    label) players to track. The opponent-star loop (~205 API calls,
    several minutes) is cached separately from everything else, since it
    doesn't depend on which OKC players are tracked -- changing the player
    list only re-runs the cheap ~35-call player pull, not the slow part.
    Delete thunder_base_cache.csv to force a full rebuild of the slow part.

    Returns (df, seasons_all, stats_table, feature_cols)."""
    if os.path.exists(BASE_CACHE_FILE):
        print(f"Loading cached base dataset from {BASE_CACHE_FILE}...")
        df = pd.read_csv(BASE_CACHE_FILE, parse_dates=['GAME_DATE'], dtype={'GAME_ID': str})
    else:
        raw = pd.read_csv('thunder_games_raw.csv', parse_dates=['GAME_DATE'], dtype={'GAME_ID': str})
        seasons_all = sorted(raw['SEASON'].unique())

        defense_table = build_opponent_defense_table(seasons_all)
        raw = add_opponent_defense_raw(raw, defense_table)
        df = add_features(raw)

        print("Building opponent star features -- this takes several minutes "
              "(one API call per season/opponent matchup)...")
        df = add_opponent_star_features(df, seasons_all)

        df.to_csv(BASE_CACHE_FILE, index=False)
        print(f"Cached base dataset to {BASE_CACHE_FILE} for fast re-runs.")

    seasons_all = sorted(df['SEASON'].unique())

    stats_table = build_prior_season_stats_table(seasons_all)
    df['OPP_PRIOR_POINT_DIFF'] = df.apply(
        lambda row: stats_table[row['SEASON']].get(row['OPPONENT'], {}).get('POINT_DIFF'), axis=1
    )
    df['TEAM_PRIOR_WIN_PCT'] = df['SEASON'].map(lambda s: stats_table.get(s, {}).get('OKC', {}).get('WIN_PCT'))
    df['TEAM_PRIOR_POINT_DIFF'] = df['SEASON'].map(lambda s: stats_table.get(s, {}).get('OKC', {}).get('POINT_DIFF'))

    for full_name, label in players:
        log = build_player_game_log(full_name, seasons_all)
        df = add_player_features(df, log, label=label)

    if exclude_seasons:
        df = df[~df['SEASON'].isin(exclude_seasons)]
        seasons_all = [s for s in seasons_all if s not in exclude_seasons]

    feature_cols = feature_cols_for(players)
    df = df.dropna(subset=feature_cols)

    return df, seasons_all, stats_table, feature_cols
