import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import feature_cols_for, TEAM_FEATURE_COLS, BACKTEST_PLAYERS, PRODUCTION_PLAYERS


def test_feature_cols_includes_all_team_level_features():
    cols = feature_cols_for([('Shai Gilgeous-Alexander', 'SGA')])
    for col in TEAM_FEATURE_COLS:
        assert col in cols


def test_feature_cols_adds_played_and_roll_pts_per_player():
    cols = feature_cols_for([('Shai Gilgeous-Alexander', 'SGA'), ('Jalen Williams', 'JDUB')])
    assert 'SGA_PLAYED' in cols
    assert 'SGA_ROLL_PTS' in cols
    assert 'JDUB_PLAYED' in cols
    assert 'JDUB_ROLL_PTS' in cols


def test_feature_cols_empty_player_list_is_just_team_features():
    assert feature_cols_for([]) == TEAM_FEATURE_COLS


def test_backtest_and_production_player_lists_differ():
    """Regression test for a real bug: using the same player list for
    backtesting (needs whoever was ACTUALLY on the roster historically)
    and production (needs the CURRENT roster) caused a real accuracy
    collapse when they were accidentally unified."""
    assert BACKTEST_PLAYERS != PRODUCTION_PLAYERS
    # Dort specifically should be in the historical list but not current.
    backtest_names = [name for name, _ in BACKTEST_PLAYERS]
    production_names = [name for name, _ in PRODUCTION_PLAYERS]
    assert 'Luguentz Dort' in backtest_names
    assert 'Luguentz Dort' not in production_names
