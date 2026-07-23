import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from weighting import recency_weights

ALL_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']


def test_immediately_preceding_season_gets_single_decay_step():
    train_seasons = pd.Series(['2023-24'])
    weights = recency_weights(train_seasons, '2024-25', ALL_SEASONS, decay=0.6)
    assert weights.iloc[0] == 0.6  # one season back = decay^1, not full weight


def test_older_seasons_decay_exponentially():
    train_seasons = pd.Series(['2021-22', '2022-23', '2023-24'])
    weights = recency_weights(train_seasons, '2024-25', ALL_SEASONS, decay=0.6)
    # 3, 2, 1 seasons back from 2024-25 respectively
    assert weights.tolist() == [0.6 ** 3, 0.6 ** 2, 0.6 ** 1]


def test_weights_strictly_decrease_with_age():
    train_seasons = pd.Series(['2021-22', '2022-23', '2023-24'])
    weights = recency_weights(train_seasons, '2024-25', ALL_SEASONS, decay=0.6)
    assert weights.iloc[0] < weights.iloc[1] < weights.iloc[2]
