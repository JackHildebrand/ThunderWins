import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from explore_data import filter_valid_okc_rows


def test_drops_rows_where_matchup_is_not_okcs_own():
    """Regression test for a real bug: LeagueGameFinder occasionally
    returned the OPPONENT's box score (wrong team, wrong WL/PTS) for NBA
    Cup knockout games, under rows that were supposed to be OKC's own."""
    df = pd.DataFrame({
        'GAME_DATE': pd.to_datetime(['2024-12-14', '2024-12-15']),
        'MATCHUP': ['HOU @ OKC', 'OKC vs. DAL'],  # first row is the real bug case
        'WL': ['W', 'L'],
        'PTS': [111, 100],
    })
    result = filter_valid_okc_rows(df, verbose=False)
    assert len(result) == 1
    assert result.iloc[0]['MATCHUP'] == 'OKC vs. DAL'


def test_keeps_all_rows_when_none_are_malformed():
    df = pd.DataFrame({
        'GAME_DATE': pd.to_datetime(['2024-12-14', '2024-12-15']),
        'MATCHUP': ['OKC @ HOU', 'OKC vs. DAL'],
        'WL': ['W', 'L'],
        'PTS': [111, 100],
    })
    result = filter_valid_okc_rows(df, verbose=False)
    assert len(result) == 2
