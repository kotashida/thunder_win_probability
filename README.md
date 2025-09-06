# Thunder Win Probability and Player Impact (RAPM) Analysis

This project demonstrates a complete data science pipeline for sports analytics, including data acquisition, feature engineering, predictive modeling, and player evaluation. It showcases the application of advanced statistical techniques to quantify team performance and individual player impact using NBA play-by-play data for the 2022-23 OKC Thunder season.

## Project Status: Success

This project successfully demonstrates a full pipeline for building a sports analytics model. Key outcomes include:

*   **Automated Data Pipeline:** A robust data pipeline to fetch and process over 40,000 play-by-play events for the full 82-game season of a single NBA team.
*   **Predictive Win Probability Model:** A predictive model built with XGBoost to calculate in-game win probability on a possession-by-possession basis. The final model achieved **71.17% accuracy** on a held-out test set.
*   **Player Impact Metric (RAPM):** Calculation of Regularized Adjusted Plus-Minus (RAPM) for all players, providing a statistically robust measure of a player's on-court impact on net scoring margin.
*   All scripts, final model assets, and results are included and documented in this repository.

---

## 1. Project Overview

This project provides two key analytical tools for the OKC Thunder:

1.  **Win Probability (WP) Model:** This model calculates the team's chance of winning at any point in the game based on real-time game state variables (score, time remaining). This provides a dynamic measure of game flow and helps quantify the leverage of specific moments.
2.  **Regularized Adjusted Plus-Minus (RAPM):** This analysis moves beyond traditional box score statistics to measure a player's true impact on the game. By using a regularized regression model, RAPM isolates a player's contribution to the team's point differential while accounting for the quality of both their teammates and opponents on the court.

## 2. How to Run This Project

Follow these steps to replicate the analysis.

### Step 1: Setup

Clone the repository and install the required Python packages. It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Fetch Data

Run the data acquisition script. This will download play-by-play data for the 2022-23 OKC Thunder season and save it to the `data` directory.

```bash
python src/fetch_data.py
```

### Step 3: Train the Win Probability Model

Run the model-building script. This will load the raw data, perform feature engineering, train an XGBoost classifier, and save the final model and an evaluation plot.

```bash
python src/02_build_wp_model.py
```

### Step 4: Calculate Player RAPM

Run the RAPM calculation script. This will process the play-by-play data to create a stint-based dataset and then run a Ridge regression to calculate player RAPM values.

```bash
python src/03_calculate_rapm.py
```

## 3. Methodology

### Win Probability Model

The win probability model is a supervised classification model designed to predict the outcome of a game (win or loss) for the team possessing the ball at any given moment.

*   **Algorithm:** An **XGBoost (Extreme Gradient Boosting)** classifier was chosen. XGBoost is a powerful and efficient implementation of a gradient-boosted decision tree algorithm, well-suited for handling the non-linear relationships between game state variables and the final outcome. Its built-in regularization helps prevent overfitting.
*   **Features:** The model uses two key features engineered from the play-by-play data:
    *   `SECONDS_REMAINING`: The total time left in the game, normalized to seconds.
    *   `SCORE_MARGIN`: The point differential for the team in possession.
*   **Training & Validation:** The dataset of all possessions was split into a training set (80%) and a test set (20%). The model was trained on the training data, and its performance was evaluated on the unseen test data to ensure generalization. The model achieved an accuracy of **71.17%** on the test set. A calibration plot was also generated to confirm that the model's predicted probabilities are reliable.

### Regularized Adjusted Plus-Minus (RAPM)

RAPM is a linear regression-based technique used to estimate a player's individual impact on the team's performance, measured in net points per 100 possessions.

*   **Model:** **Ridge Regression (L2 Regularization)** was used to build the RAPM model. This is crucial for two reasons:
    1.  **Multicollinearity:** In basketball, players frequently play together, creating high multicollinearity in a standard linear model. Ridge regression mitigates this by penalizing large coefficient values, preventing the model from assigning undue credit or blame to any single player in a lineup.
    2.  **Variance Reduction:** The regularization term shrinks the coefficients of less impactful players towards zero, leading to more stable and reliable estimates of player value, especially for those with less playing time.
*   **Data Structure:** The play-by-play data was transformed into a "stint-based" format. Each row in the regression matrix represents a continuous period of play where the 10 players on the court remain the same.
*   **Regression Formulation:** The model is specified as:

    *   **Target Variable (y):** The net point differential per 100 possessions during a stint.
    *   **Predictor Variables (X):** A sparse matrix where each column represents a player. A value of `+1` is assigned if the player was on the home team during the stint, `-1` if on the away team, and `0` if not on the court.

    The resulting regression coefficients represent each player's RAPM, or their estimated impact on net point differential per 100 possessions, holding constant the quality of all other players on the court.

## 4. Key Quantitative Skills Demonstrated

*   **Data Acquisition & ETL:** Automated data collection from a web API (`nba_api`) and transformation of raw, semi-structured data into a clean, analysis-ready format.
*   **Predictive Modeling:**
    *   Implementation of a high-performance **XGBoost classifier** for a binary classification task.
    *   Model evaluation using accuracy, precision, recall, F1-score, and calibration analysis.
    *   Feature engineering to create predictive variables from raw time-series data.
*   **Advanced Regression Analysis:**
    *   Application of **Ridge Regression (L2 Regularization)** to address multicollinearity and produce robust estimates in a high-dimensional feature space.
    *   Formulation of a complex sports analytics problem (player impact) into a linear regression framework.
*   **Python for Data Science:** Proficient use of core data science libraries (`pandas`, `numpy`, `scikit-learn`, `xgboost`) for data manipulation, modeling, and analysis.
*   **Algorithmic Problem Solving:** Development of custom algorithms to parse game states, track player substitutions, and construct a stint-based matrix for RAPM calculation.

## 5. Project Structure

```
.
├── data/                            # Raw data storage
│   └── pbp_data_2022-23_OKC.csv
├── results/                         # Output files from scripts
│   ├── wp_model.joblib              # The trained XGBoost model object
│   ├── wp_model_calibration_plot.png  # Evaluation plot for the WP model
│   └── player_rapm.csv              # Calculated RAPM values for all players
├── src/                             # All Python source code
│   ├── 01_fetch_data.py             # Script to download data from nba_api
│   ├── 02_build_wp_model.py         # Script to train and save the WP model
│   └── 03_calculate_rapm.py         # Script to calculate player RAPM
├── README.md                        # This file
└── requirements.txt                 # Project dependencies
```