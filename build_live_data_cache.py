"""Precompute the '2025-26 season' data predict_game() needs.

2025-26 is complete, so team stats, opponent strength, and leading scorers
for that season are static facts, not something that needs a live NBA.com
call on every prediction. Precomputing and caching them here means the
deployed app doesn't depend on reaching stats.nba.com at prediction time --
which matters because NBA.com occasionally blocks cloud/datacenter IPs
(confirmed: the deployed app got read-timeouts hitting stats.nba.com, while
the same code works fine run locally). The only thing that still needs to
be genuinely live is injury status, which comes from ESPN, not NBA.com, and
is unaffected by this.

Re-run this once 2026-27 games start, at which point "prior season" should
shift to mean 2026-27-so-far rather than the completed 2025-26."""
import time
import joblib

from opponent_strength import get_season_team_stats
from opponent_stars import get_season_leading_scorers_with_names
from nba_api.stats.endpoints import playergamelog

LIVE_DATA_CACHE_FILE = 'live_data_cache.joblib'


def recent_scoring_avg(player_id, season='2025-26'):
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if len(log) == 0:
        return 0.0
    recent = log.sort_values('Game_ID')['PTS'].tail(10).tolist()
    return sum(recent) / len(recent) if recent else 0.0


if __name__ == "__main__":
    print("Pulling 2025-26 team stats...")
    season_2526_team_stats = get_season_team_stats('2025-26')

    print("Identifying 2025-26 leading scorers...")
    leading_scorers = get_season_leading_scorers_with_names('2025-26')

    print("Pulling each opponent's leading scorer's recent form...")
    opp_star_recent_form = {}
    for abbr, (player_id, name) in leading_scorers.items():
        if abbr == 'OKC':
            continue
        print(f"  {abbr}: {name}")
        opp_star_recent_form[abbr] = recent_scoring_avg(player_id)
        time.sleep(1)

    joblib.dump({
        'season_2526_team_stats': season_2526_team_stats,
        'leading_scorers': leading_scorers,
        'opp_star_recent_form': opp_star_recent_form,
    }, LIVE_DATA_CACHE_FILE)

    print(f"\nSaved {LIVE_DATA_CACHE_FILE}")
