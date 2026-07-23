import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from player_stats import add_player_features


def make_team_df(game_ids, dates):
    return pd.DataFrame({
        'GAME_ID': game_ids,
        'GAME_DATE': pd.to_datetime(dates),
    })


def test_played_flag_true_for_games_in_players_log():
    team_df = make_team_df(['1', '2', '3'], ['2023-10-20', '2023-10-22', '2023-10-24'])
    player_log = pd.DataFrame({
        'GAME_ID': ['1', '3'],  # missed game 2
        'GAME_DATE': pd.to_datetime(['2023-10-20', '2023-10-24']),
        'STAR_ROLL_PTS': [20.0, 22.0],
    })
    result = add_player_features(team_df, player_log, label='TEST')
    assert result.set_index('GAME_ID')['TEST_PLAYED'].to_dict() == {'1': 1, '2': 0, '3': 1}


def test_roll_pts_carries_forward_across_a_missed_game():
    """A game the player missed should carry forward their most recent
    known form, not go blank -- an injury shouldn't erase their track
    record for the games right after they return."""
    team_df = make_team_df(['1', '2', '3'], ['2023-10-20', '2023-10-22', '2023-10-24'])
    player_log = pd.DataFrame({
        'GAME_ID': ['1', '3'],
        'GAME_DATE': pd.to_datetime(['2023-10-20', '2023-10-24']),
        'STAR_ROLL_PTS': [20.0, 22.0],
    })
    result = add_player_features(team_df, player_log, label='TEST')
    # Game 2 (missed): carries forward game 1's value, doesn't jump ahead
    # to game 3's value (that would be looking into the future).
    assert result.set_index('GAME_ID').loc['2', 'TEST_ROLL_PTS'] == 20.0


def test_roll_pts_defaults_to_zero_before_first_ever_appearance():
    """A player who joined the team later (e.g. a rookie) has no games
    before their debut -- those should be 0, not NaN or a fabricated
    value, and definitely shouldn't drop the row."""
    team_df = make_team_df(['1', '2'], ['2019-10-20', '2019-10-22'])
    player_log = pd.DataFrame({'GAME_ID': [], 'GAME_DATE': pd.to_datetime([]), 'STAR_ROLL_PTS': []})
    result = add_player_features(team_df, player_log, label='TEST')
    assert (result['TEST_ROLL_PTS'] == 0.0).all()
    assert (result['TEST_PLAYED'] == 0).all()
    assert len(result) == 2  # no rows dropped
