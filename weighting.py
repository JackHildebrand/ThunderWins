def recency_weights(train_seasons, test_season, all_seasons, decay=0.6):
    """Older seasons get exponentially less weight relative to test_season.
    Pulled out of train_model.py into its own module with no top-level
    side effects: train_model.py loads real (network-backed) data at import
    time, which would make importing it in a test slow and dependent on
    network/cache state. This function has neither problem."""
    test_idx = all_seasons.index(test_season)
    return train_seasons.map(lambda s: decay ** (test_idx - all_seasons.index(s)))
