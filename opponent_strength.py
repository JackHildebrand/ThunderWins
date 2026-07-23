import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguestandings
import time

TEAM_ID_TO_ABBR = {t['id']: t['abbreviation'] for t in teams.get_teams()}

def previous_season(season):
    """'2023-24' -> '2022-23'"""
    start_year = int(season[:4])
    prev_start = start_year - 1
    return f"{prev_start}-{str(prev_start + 1)[-2:]}"

def get_season_team_stats(season):
    """Return {team_abbreviation: {'WIN_PCT': ..., 'POINT_DIFF': ...}} for one season.
    Point differential is a steadier measure of team strength than win% alone,
    since win% can be skewed by a handful of close-game bounces."""
    standings = leaguestandings.LeagueStandings(season=season)
    df = standings.get_data_frames()[0]
    df['ABBR'] = df['TeamID'].map(TEAM_ID_TO_ABBR)
    return {
        row.ABBR: {'WIN_PCT': row.WinPCT, 'POINT_DIFF': row.DiffPointsPG}
        for row in df.itertuples()
    }

def build_prior_season_stats_table(seasons):
    """For each season, look up team stats from the season BEFORE it (no leakage).
    Returns {season: {team_abbr: {'WIN_PCT': .., 'POINT_DIFF': ..}}}."""
    prior_seasons_needed = sorted(set(previous_season(s) for s in seasons))
    cache = {}
    for prior in prior_seasons_needed:
        print(f"Pulling standings for {prior}...")
        cache[prior] = get_season_team_stats(prior)
        time.sleep(1)
    return {s: cache[previous_season(s)] for s in seasons}
