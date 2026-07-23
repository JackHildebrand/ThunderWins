import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog
import time

def previous_season(season):
    """'2023-24' -> '2022-23'"""
    start_year = int(season[:4])
    prev_start = start_year - 1
    return f"{prev_start}-{str(prev_start + 1)[-2:]}"

def get_season_leading_scorers_with_names(season):
    """One API call: {team_abbreviation: (player_id, player_name)} for the
    top scorer (by total season points) on every team that season."""
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, season_type_all_star='Regular Season')
    df = stats.get_data_frames()[0]
    idx = df.groupby('TEAM_ABBREVIATION')['PTS'].idxmax()
    top = df.loc[idx]
    return {row.TEAM_ABBREVIATION: (row.PLAYER_ID, row.PLAYER_NAME) for row in top.itertuples()}

def get_season_leading_scorers(season):
    """One API call: {team_abbreviation: player_id} for the top scorer
    (by total season points) on every team that season."""
    return {abbr: pid for abbr, (pid, name) in get_season_leading_scorers_with_names(season).items()}

def add_opponent_star_features(team_df, seasons):
    """For each (season, opponent) matchup OKC actually played, identify
    that opponent's leading scorer from the PRIOR season (no leakage --
    we're not using this season's still-in-progress totals to decide who
    "the star" is), pull that player's CURRENT-season game log, and record
    whether they played in this specific game plus their rolling scoring
    average entering it.

    This makes one API call per distinct (season, opponent) pair actually
    faced -- roughly 200 across our 7 seasons -- so it takes several
    minutes to run. Individual failures are skipped rather than crashing
    the whole run, since NBA.com's stats API occasionally times out."""
    team_df = team_df.copy()
    team_df['OPP_STAR_PLAYED'] = 0
    team_df['OPP_STAR_ROLL_PTS'] = 0.0

    prior_seasons_needed = sorted(set(previous_season(s) for s in seasons))
    scorer_cache = {}
    for prior in prior_seasons_needed:
        print(f"Identifying leading scorers for {prior}...")
        scorer_cache[prior] = get_season_leading_scorers(prior)
        time.sleep(1)

    matchups = list(team_df.groupby(['SEASON', 'OPPONENT']).groups.items())
    for i, ((season, opponent), idx) in enumerate(matchups):
        player_id = scorer_cache.get(previous_season(season), {}).get(opponent)
        if player_id is None:
            continue
        print(f"  [{i+1}/{len(matchups)}] {opponent} star, {season}...")
        try:
            log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        except Exception as e:
            print(f"    skipped ({e})")
            continue
        time.sleep(1)
        if len(log) == 0:
            continue
        log = log.rename(columns={'Game_ID': 'GAME_ID'})
        log['GAME_DATE'] = pd.to_datetime(log['GAME_DATE'], format='%b %d, %Y')
        log = log.sort_values('GAME_DATE')
        log['ROLL_PTS'] = log['PTS'].shift(1).rolling(10, min_periods=3).mean()
        played_ids = set(log['GAME_ID'])

        for row_idx in idx:
            gid = team_df.at[row_idx, 'GAME_ID']
            gdate = team_df.at[row_idx, 'GAME_DATE']
            team_df.at[row_idx, 'OPP_STAR_PLAYED'] = int(gid in played_ids)
            prior_rows = log.loc[log['GAME_DATE'] <= gdate, 'ROLL_PTS'].dropna()
            if len(prior_rows) > 0:
                team_df.at[row_idx, 'OPP_STAR_ROLL_PTS'] = prior_rows.iloc[-1]

    return team_df
