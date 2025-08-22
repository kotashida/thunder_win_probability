# Thunder Win Probability Model

---

## Project Status: Success

This project successfully demonstrates a full pipeline for building a sports analytics model. Key outcomes include:

*   A robust data pipeline to fetch NBA play-by-play data for a full season.
*   A predictive model built with XGBoost to calculate in-game win probability.
*   The final model achieved **71.17% accuracy** on a test set of possessions from the 2022-23 OKC Thunder season.
*   All scripts and final model assets are included and documented in this repository.

---

## 1. Project Overview

This project provides an actionable, data-driven tool for the OKC Thunder by modeling win probability on a possession-by-possession basis. 

The model calculates the team's chance of winning at any point in the game based on the score and time remaining. This provides a real-time measure of game state and can be used to quantify the importance of key moments.

## 2. How to Run This Project

Follow these steps to replicate the Win Probability model.

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

Run the data acquisition script. This will download play-by-play data for the 2022-23 OKC Thunder season and save it to the `data` directory. This may take several minutes.

```bash
python src/01_fetch_data.py
```

### Step 3: Train the Win Probability Model

Run the model-building script. This will load the raw data, perform feature engineering, train an XGBoost classifier, and save the final model and an evaluation plot to the `results` directory.

```bash
python src/02_build_wp_model.py
```

## 3. Project Structure

```
.
├── data/                            # Raw data storage
│   └── pbp_data_2022-23_OKC.csv
├── results/                         # Output files from scripts
│   ├── wp_model.joblib              # The trained XGBoost model object
│   └── wp_model_calibration_plot.png  # Evaluation plot for the model
├── src/                             # All Python source code
│   ├── 01_fetch_data.py             # Script to download data from nba_api
│   └── 02_build_wp_model.py         # Script to train and save the WP model
├── README.md                        # This file
└── requirements.txt                 # Project dependencies
```

## 4. Future Work & Improvements

The Win Probability model can be further improved by engineering more features from the play-by-play data, such as:

*   Timeouts remaining for each team.
*   Team foul / bonus status.
*   Ratings for the specific players on the court for a given possession.
