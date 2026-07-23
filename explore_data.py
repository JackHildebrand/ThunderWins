import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import time

THUNDER_ID = [t for t in teams.get_teams() if t['abbreviation'] == 'OKC'][0]['id']

def get_season_games(season):
    """Pull one season's regular-season game log for the Thunder as a DataFrame.
    Uses LeagueGameFinder instead of TeamGameLog because it includes PLUS_MINUS
    (point differential), which TeamGameLog doesn't provide."""
    gf = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=THUNDER_ID,
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    return gf.get_data_frames()[0]

def get_multi_season_games(seasons):
    """Pull and combine multiple seasons, oldest to newest."""
    all_games = []
    for season in seasons:
        print(f"Pulling {season}...")
        df = get_season_games(season)
        df['SEASON'] = season
        all_games.append(df)
        time.sleep(1)  # be polite to NBA.com's servers
    combined = pd.concat(all_games, ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    combined = combined.sort_values('GAME_DATE').reset_index(drop=True)

    # Sanity check: every row should be OKC's OWN box score, so MATCHUP
    # should always start with "OKC" (e.g. "OKC vs. DAL" or "OKC @ DAL").
    # NBA Cup knockout-round games have occasionally come back from this
    # endpoint as the OPPONENT's box score instead (wrong team, wrong
    # WL/PTS entirely) -- drop those rather than train on corrupted labels.
    bad_rows = combined[~combined['MATCHUP'].str.startswith('OKC')]
    if len(bad_rows) > 0:
        print(f"Dropping {len(bad_rows)} malformed row(s) where the API returned "
              f"the opponent's box score instead of OKC's:")
        print(bad_rows[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']])
        combined = combined[combined['MATCHUP'].str.startswith('OKC')].reset_index(drop=True)

    return combined

if __name__ == "__main__":
    seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    games = get_multi_season_games(seasons)
    print("\nTotal games pulled:", len(games))
    print(games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'PLUS_MINUS']].head(10))
    games.to_csv('thunder_games_raw.csv', index=False)
    print("\nSaved to thunder_games_raw.csv")