import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_injuries import players_out

SAMPLE_INJURIES = {
    'Oklahoma City Thunder': {
        'Shai Gilgeous-Alexander': 'Out',
        'Jalen Williams': 'Day-To-Day',
        'Chet Holmgren': 'Questionable',
    },
    'Denver Nuggets': {
        'Jamal Murray': 'Out',
    },
}


def test_only_literal_out_status_counts():
    """Day-To-Day and Questionable players usually still suit up -- only
    a literal 'Out' status should force a player out of a prediction."""
    result = players_out(SAMPLE_INJURIES, 'Thunder',
                          ['Shai Gilgeous-Alexander', 'Jalen Williams', 'Chet Holmgren'])
    assert result == {'Shai Gilgeous-Alexander'}


def test_matches_team_by_name_fragment():
    result = players_out(SAMPLE_INJURIES, 'Nuggets', ['Jamal Murray'])
    assert result == {'Jamal Murray'}


def test_no_matching_team_returns_empty_set():
    result = players_out(SAMPLE_INJURIES, 'Lakers', ['LeBron James'])
    assert result == set()


def test_player_not_in_injury_report_is_not_out():
    result = players_out(SAMPLE_INJURIES, 'Thunder', ['Isaiah Hartenstein'])
    assert result == set()


def test_empty_injuries_dict_returns_empty_set():
    result = players_out({}, 'Thunder', ['Shai Gilgeous-Alexander'])
    assert result == set()
