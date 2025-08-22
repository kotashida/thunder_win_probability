import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- Data Cleaning and Feature Engineering Functions ---

def parse_time(period, time_str):
    """Converts period and MM:SS time string to seconds remaining in the game."""
    try:
        minutes, seconds = map(int, time_str.split(':'))
        if period <= 4:
            return (4 - period) * 720 + (minutes * 60 + seconds)
        else:
            return (minutes * 60 + seconds)
    except:
        return np.nan

def parse_score(score_str):
    """Parses 'H - A' score string into a tuple (home_score, away_score)."""
    if not isinstance(score_str, str):
        return np.nan, np.nan
    try:
        away, home = map(int, score_str.split(' - '))
        return home, away
    except (ValueError, TypeError):
        return np.nan, np.nan

def process_pbp_data(df):
    """Main function to clean and engineer features from raw PBP data."""
    print("Processing raw PBP data...")
    df['SCOREMARGIN'] = pd.to_numeric(df['SCOREMARGIN'], errors='coerce')
    df.ffill(inplace=True)
    df['HOME_TEAM_ID'] = df.groupby('GAME_ID')['PLAYER1_TEAM_ID'].transform('max')
    df['AWAY_TEAM_ID'] = df.groupby('GAME_ID')['PLAYER1_TEAM_ID'].transform('min')
    game_winners = df.groupby('GAME_ID').last()
    game_winners['WINNER_TEAM_ID'] = np.where(game_winners['SCOREMARGIN'] > 0, game_winners['HOME_TEAM_ID'], game_winners['AWAY_TEAM_ID'])
    df = df.merge(game_winners[['WINNER_TEAM_ID']], on='GAME_ID', how='left')
    df['SECONDS_REMAINING'] = df.apply(lambda row: parse_time(row['PERIOD'], row['PCTIMESTRING']), axis=1)
    scores = df['SCORE'].apply(parse_score)
    df[['HOME_SCORE', 'AWAY_SCORE']] = pd.DataFrame(scores.tolist(), index=df.index)
    df[['HOME_SCORE', 'AWAY_SCORE']] = df.groupby('GAME_ID')[[ 'HOME_SCORE', 'AWAY_SCORE']].ffill().bfill()
    df['POSS_TEAM_ID'] = df['PLAYER1_TEAM_ID']
    df = df[df['POSS_TEAM_ID'].notna()]
    df['POSS_TEAM_ID'] = df['POSS_TEAM_ID'].astype(int)
    df['SCORE_MARGIN'] = np.where(
        df['POSS_TEAM_ID'] == df['HOME_TEAM_ID'],
        df['HOME_SCORE'] - df['AWAY_SCORE'],
        df['AWAY_SCORE'] - df['HOME_SCORE']
    )
    df['POSS_TEAM_WON'] = (df['POSS_TEAM_ID'] == df['WINNER_TEAM_ID']).astype(int)
    model_df = df[['SECONDS_REMAINING', 'SCORE_MARGIN', 'POSS_TEAM_WON']].copy()
    model_df.dropna(inplace=True)
    print(f"Processing complete. Usable possessions for modeling: {len(model_df)}")
    return model_df

def main():
    """Main function to build and evaluate the WP model."""
    INPUT_PATH = "data/pbp_data_2022-23_OKC.csv"
    MODEL_OUTPUT_PATH = "results/wp_model.joblib"
    PLOT_OUTPUT_PATH = "results/wp_model_calibration_plot.png"

    raw_df = pd.read_csv(INPUT_PATH)
    model_data = process_pbp_data(raw_df)

    if model_data.empty:
        print("No data available for modeling after processing. Aborting.")
        return

    X = model_data[['SECONDS_REMAINING', 'SCORE_MARGIN']]
    y = model_data['POSS_TEAM_WON']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    print(f"\nSaving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print(f"Saving calibration plot to {PLOT_OUTPUT_PATH}...")
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot for Win Probability Model (XGBoost)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_OUTPUT_PATH)
    print("\nWin Probability model training complete.")

if __name__ == "__main__":
    main()