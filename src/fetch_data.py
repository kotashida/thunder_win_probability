import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog, playbyplayv2

def get_team_id(team_abbreviation="OKC"): 
    """Gets the team ID for a given team abbreviation."""
    nba_teams = teams.get_teams()
    team = [t for t in nba_teams if t['abbreviation'] == team_abbreviation][0]
    return team['id']

def fetch_league_gamelog(season="2022-23"):
    """Fetches all game logs for the entire league in a given season."""
    print(f"Fetching league game log for season {season}...")
    # The `leaguegamelog` endpoint is more reliable than `teamgamelog`
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
    return log.get_data_frames()[0]

def fetch_pbp_data(game_ids):
    """Fetches play-by-play data for a list of game IDs."""
    all_pbp_data = []
    total_games = len(game_ids)
    print(f"Found {total_games} games to fetch.")
    
    for i, game_id in enumerate(game_ids):
        print(f"Fetching PBP for game {i+1}/{total_games} (ID: {game_id})...")
        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_df = pbp.get_data_frames()[0]
            all_pbp_data.append(pbp_df)
            # Be a good API citizen
            time.sleep(0.7) 
        except Exception as e:
            print(f"Could not fetch game {game_id}. Error: {e}")
            
    if not all_pbp_data:
        print("\nError: No play-by-play data was fetched. Cannot proceed.")
        return pd.DataFrame()
        
    return pd.concat(all_pbp_data, ignore_index=True)

def main():
    """Main function to run the data fetching process."""
    SEASON = "2022-23"
    TEAM_ABBREVIATION = "OKC"
    OUTPUT_PATH = f"data/pbp_data_{SEASON}_{TEAM_ABBREVIATION}.csv"

    team_id = get_team_id(TEAM_ABBREVIATION)
    
    league_gamelog_df = fetch_league_gamelog(SEASON)

    if league_gamelog_df.empty:
        print("Fetching league game log failed. Aborting.")
        return

    # Filter the league log for the specific team's games
    team_game_ids = league_gamelog_df[league_gamelog_df['TEAM_ID'] == team_id]['GAME_ID'].unique()

    pbp_df = fetch_pbp_data(team_game_ids)
    
    if not pbp_df.empty:
        print(f"\nSaving all PBP data to {OUTPUT_PATH}...")
        pbp_df.to_csv(OUTPUT_PATH, index=False)
        print("Data fetching complete.")

if __name__ == "__main__":
    main()