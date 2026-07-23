import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opponent_stars import previous_season


def test_previous_season_normal_case():
    assert previous_season('2023-24') == '2022-23'


def test_previous_season_across_decade():
    assert previous_season('2020-21') == '2019-20'


def test_previous_season_first_season_in_our_dataset():
    assert previous_season('2019-20') == '2018-19'


def test_previous_season_millennium_boundary():
    # '1999-00' is the real NBA notation for the 1999-2000 season --
    # a genuine edge case for the zero-padding logic.
    assert previous_season('2000-01') == '1999-00'
