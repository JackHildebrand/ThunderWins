import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time

def get_league_games(season):
    """Pull every team's own box score for every regular-season game in a season."""
    gf = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable='Regular Season')
    return gf.get_data_frames()[0]

def build_opponent_defense_table(seasons):
    """Every team's box score for every game, league-wide. We'll filter this
    down to just the OPPONENT's row for each of OKC's games -- i.e. what
    OKC's defense allowed that game, not what OKC shot."""
    all_games = []
    for season in seasons:
        print(f"Pulling league-wide box scores for {season}...")
        all_games.append(get_league_games(season))
        time.sleep(1)
    league = pd.concat(all_games, ignore_index=True)
    return league[['GAME_ID', 'TEAM_ABBREVIATION', 'FG_PCT', 'FG3_PCT']]

def add_opponent_defense_raw(team_df, defense_table):
    """Merge in the OPPONENT's own shooting stats for each Thunder game,
    keyed by GAME_ID (globally unique per game, so this correctly pulls
    only the team OKC actually played that day)."""
    opp_rows = defense_table[defense_table['TEAM_ABBREVIATION'] != 'OKC'].copy()
    opp_rows = opp_rows.rename(columns={'FG_PCT': 'OPP_FG_PCT', 'FG3_PCT': 'OPP_FG3_PCT'})
    opp_rows = opp_rows[['GAME_ID', 'OPP_FG_PCT', 'OPP_FG3_PCT']]
    return team_df.merge(opp_rows, on='GAME_ID', how='left')
