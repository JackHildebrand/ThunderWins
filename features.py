import pandas as pd

def add_features(df):
    df = df.copy()

    # Target: what we're trying to predict (1 = win, 0 = loss)
    df['WIN'] = (df['WL'] == 'W').astype(int)

    # Home vs away: "OKC vs. DAL" = home, "OKC @ DAL" = away
    df['HOME'] = (~df['MATCHUP'].str.contains('@')).astype(int)

    # Opponent abbreviation is always the last 3 characters
    df['OPPONENT'] = df['MATCHUP'].str[-3:]

    # Days of rest since the previous game, within the same season
    df['REST_DAYS'] = df.groupby('SEASON')['GAME_DATE'].diff().dt.days

    # Rolling averages of the Thunder's OWN recent performance.
    # Computed per season (roster/coaching changes each offseason),
    # and shift(1) means today's game is never used to predict itself.
    rolling_cols = ['WIN', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                     'OPP_FG_PCT', 'OPP_FG3_PCT']
    window = 10
    for col in rolling_cols:
        df[f'ROLL_{col}'] = (
            df.groupby('SEASON')[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )

    return df

if __name__ == "__main__":
    raw = pd.read_csv('thunder_games_raw.csv', parse_dates=['GAME_DATE'])
    df = add_features(raw)

    feature_cols = ['GAME_DATE', 'OPPONENT', 'HOME', 'REST_DAYS',
                     'ROLL_WIN', 'ROLL_PTS', 'ROLL_FG_PCT', 'WIN']
    print(df[feature_cols].dropna().head(15))
    print("\nTotal rows:", len(df))
    print("Rows with enough history to use:", len(df.dropna(subset=['ROLL_WIN'])))

    df.to_csv('thunder_games_features.csv', index=False)
    print("\nSaved to thunder_games_features.csv")