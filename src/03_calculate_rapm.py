import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from nba_api.stats.endpoints import boxscoretraditionalv2
import time

def get_player_name_map(df):
    """Creates a mapping from player ID to player name."""
    player_df = df[df['PLAYER1_NAME'].notna()][['PLAYER1_ID', 'PLAYER1_NAME']]
    player_df = player_df.drop_duplicates(subset='PLAYER1_ID')
    return player_df.set_index('PLAYER1_ID')['PLAYER1_NAME'].to_dict()

def get_starters(game_id):
    """Fetches starting lineups for a game from the box score."""
    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        time.sleep(0.7) # Be a good API citizen
        player_stats = boxscore.get_data_frames()[0]
        starters = player_stats[player_stats['START_POSITION'].notna()]
        home_starters = set(starters[starters['TEAM_ID'] == starters['HOME_TEAM_ID'].iloc[0]]['PLAYER_ID'])
        away_starters = set(starters[starters['TEAM_ID'] == starters['VISITOR_TEAM_ID'].iloc[0]]['PLAYER_ID'])
        if len(home_starters) == 5 and len(away_starters) == 5:
            return home_starters, away_starters
    except Exception as e:
        print(f"  - Could not fetch starters for {game_id}: {e}")
    return None, None

def calculate_rapm_final(df):
    """Calculates RAPM by fetching starters and then tracking substitutions."""
    print("Starting final RAPM calculation...")
    player_name_map = get_player_name_map(df)
    all_stints = []
    game_ids = df['GAME_ID'].unique()

    for i, game_id in enumerate(game_ids):
        print(f"Processing game {i+1}/{len(game_ids)} (ID: {game_id})...")
        game_df = df[df['GAME_ID'] == game_id].copy()
        
        home_starters, away_starters = get_starters(game_id)
        if not home_starters or not away_starters:
            print(f"  - Warning: Could not determine starters for game {game_id}. Skipping game.")
            continue

        on_court = {'home': home_starters.copy(), 'away': away_starters.copy()}
        stint_data = []
        last_time = 0
        last_home_score, last_away_score = 0, 0

        for _, event in game_df.iterrows():
            # Simplified time calculation
            current_time = (event['PERIOD'] - 1) * 720 + (720 - (int(event['PCTIMESTRING'].split(':')[0]) * 60 + int(event['PCTIMESTRING'].split(':')[1])))
            
            home_s, away_s = last_home_score, last_away_score
            if event['SCORE'] and isinstance(event['SCORE'], str):
                try:
                    away_s_new, home_s_new = map(int, event['SCORE'].split(' - '))
                    home_s, away_s = home_s_new, away_s_new
                except ValueError:
                    pass

            # Event Type 8 is a substitution
            if event['EVENTMSGTYPE'] == 8:
                duration = current_time - last_time
                if duration > 0:
                    stint_data.append({
                        'duration': duration,
                        'home_players': tuple(sorted(list(on_court['home']))),
                        'away_players': tuple(sorted(list(on_court['away']))),
                        'net_home_points': (home_s - last_home_score) - (away_s - last_away_score)
                    })
                
                p_out, p_in = event['PLAYER1_ID'], event['PLAYER2_ID']
                if p_out in on_court['home']:
                    on_court['home'].discard(p_out)
                    on_court['home'].add(p_in)
                elif p_out in on_court['away']:
                    on_court['away'].discard(p_out)
                    on_court['away'].add(p_in)
                
                last_time = current_time
                last_home_score, last_away_score = home_s, away_s

        all_stints.extend(stint_data)

    if not all_stints:
        print("Could not generate any stints. Aborting RAPM calculation.")
        return None

    print("\nBuilding regression matrix from stints...")
    stints_df = pd.DataFrame(all_stints)
    all_players = sorted(list(set(p for s in stints_df['home_players'] for p in s) | set(p for s in stints_df['away_players'] for p in s)))
    player_map = {player_id: i for i, player_id in enumerate(all_players)}

    X = np.zeros((len(stints_df), len(all_players)))
    y = stints_df['net_home_points'] / stints_df['duration'] * (48 * 60) # Normalize to per 48 mins

    for i, stint in stints_df.iterrows():
        for p_id in stint['home_players']:
            if p_id in player_map: X[i, player_map[p_id]] = 1
        for p_id in stint['away_players']:
            if p_id in player_map: X[i, player_map[p_id]] = -1

    print("Running Ridge Regression...")
    model = RidgeCV(alphas=np.logspace(2, 6, 20), fit_intercept=True)
    model.fit(X, y)

    rapm_results = pd.DataFrame({
        'PLAYER_ID': all_players,
        'RAPM': model.coef_
    })
    rapm_results['PLAYER_NAME'] = rapm_results['PLAYER_ID'].map(player_name_map)
    return rapm_results.sort_values('RAPM', ascending=False)[['PLAYER_NAME', 'PLAYER_ID', 'RAPM']]

def main():
    INPUT_PATH = "data/pbp_data_2022-23_OKC.csv"
    OUTPUT_PATH = "results/player_rapm.csv"

    raw_df = pd.read_csv(INPUT_PATH, low_memory=False)
    rapm_df = calculate_rapm_final(raw_df)

    if rapm_df is not None and not rapm_df.empty:
        print(f"\nSaving RAPM results to {OUTPUT_PATH}...")
        rapm_df.to_csv(OUTPUT_PATH, index=False)
        print("\nRAPM calculation complete.")
        print("--- Top 15 Players by RAPM ---")
        print(rapm_df.head(15))

if __name__ == "__main__":
    main()
