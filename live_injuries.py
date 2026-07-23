import requests

# Unofficial, undocumented ESPN endpoint -- NBA.com's own stats API (nba_api)
# has no live injury data, this is the closest free alternative. Since it's
# undocumented, it could change shape or disappear without notice; callers
# should treat a failed fetch as "no auto-detected injuries" rather than
# crash, and keep a manual override available.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def get_current_injuries():
    """Fetch the current league-wide injury report.
    Returns {team_display_name: {player_display_name: status}}, e.g.
    {'Oklahoma City Thunder': {'Jalen Williams': 'Day-To-Day'}, ...}.
    Returns {} on any failure rather than raising, since this is a
    best-effort convenience, not a dependency the rest of the model needs
    to function."""
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Warning: couldn't fetch live injury data ({e}). "
              f"Falling back to no auto-detected injuries -- use players_out manually.")
        return {}

    result = {}
    for team in data.get('injuries', []):
        team_injuries = {}
        for inj in team.get('injuries', []):
            name = inj.get('athlete', {}).get('displayName')
            status = inj.get('status')
            if name:
                team_injuries[name] = status
        result[team.get('displayName', '')] = team_injuries
    return result


def players_out(all_injuries, team_name_fragment, tracked_full_names):
    """Given the dict from get_current_injuries(), a substring to match a
    team's display name (e.g. 'Thunder' or 'Nuggets'), and a list of full
    player names to check, return the subset whose status is literally
    'Out'. Day-To-Day/Questionable are NOT treated as out, since those
    players usually do suit up."""
    matched_team = next((name for name in all_injuries if team_name_fragment in name), None)
    if matched_team is None:
        return set()
    team_injuries = all_injuries[matched_team]
    return {name for name in tracked_full_names if team_injuries.get(name) == 'Out'}
