import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import time

def get_player_id(full_name):
    matches = players.find_players_by_full_name(full_name)
    return matches[0]['id']

def build_player_game_log(full_name, seasons):
    """Pull a player's game log across seasons. Returns a DataFrame sorted
    by date with a rolling scoring average of the player's OWN recent
    games (shifted by 1 so a game's own points never leak into its
    own feature)."""
    player_id = get_player_id(full_name)
    all_logs = []
    for season in seasons:
        print(f"Pulling {full_name} - {season}...")
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = log.get_data_frames()[0]
        df['SEASON'] = season
        if len(df) > 0:
            all_logs.append(df)
        time.sleep(1)
    combined = pd.concat(all_logs, ignore_index=True)
    combined = combined.rename(columns={'Game_ID': 'GAME_ID'})
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'], format='%b %d, %Y')
    combined = combined.sort_values('GAME_DATE').reset_index(drop=True)

    combined['STAR_ROLL_PTS'] = combined['PTS'].shift(1).rolling(10, min_periods=3).mean()

    return combined[['GAME_ID', 'GAME_DATE', 'MIN', 'PTS', 'STAR_ROLL_PTS']]

def add_player_features(team_df, player_log, label):
    """Merge in (1) whether this player played this exact game, and (2) their
    rolling scoring average as of their most recent PRIOR appearance -- so
    a game they missed still carries forward their last known form instead
    of going blank.

    Players who joined the team more recently (e.g. a rookie who debuted in
    2023-24) simply have no game log for earlier seasons. Rather than drop
    those historical rows entirely (which would wipe out years of otherwise
    good training data), their rolling average is filled with 0 for any game
    before their first-ever appearance -- meaning 'this factor didn't exist
    yet,' not a fabricated stat line."""
    team_df = team_df.copy().sort_values('GAME_DATE')

    played_game_ids = set(player_log['GAME_ID'])
    team_df[f'{label}_PLAYED'] = team_df['GAME_ID'].isin(played_game_ids).astype(int)

    log_sorted = player_log[['GAME_DATE', 'STAR_ROLL_PTS']].sort_values('GAME_DATE').dropna()
    team_df = pd.merge_asof(team_df, log_sorted, on='GAME_DATE', direction='backward')
    team_df = team_df.rename(columns={'STAR_ROLL_PTS': f'{label}_ROLL_PTS'})
    team_df[f'{label}_ROLL_PTS'] = team_df[f'{label}_ROLL_PTS'].fillna(0)

    return team_df
