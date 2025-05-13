import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NHL data
nhl_schedule_df = pd.read_csv('path/to/NHL_Schedule.csv')
nhl_player_stats_df = pd.read_csv('path/to/NHL_Player_Stats.csv')

# Convert dates to datetime for easier matching
nhl_schedule_df['DATE'] = pd.to_datetime(nhl_schedule_df['Date'], format='%m/%d/%Y', errors='coerce')

# Function to select relevant stats based on position
def select_stats(position):
    if position == 'Forward':
        return ['Goals', 'Assists', 'Points', 'PlusMinus', 'Shots']
    elif position == 'Defenseman':
        return ['Goals', 'Assists', 'Points', 'PlusMinus', 'Blocks']
    else:  # Goalie
        return ['Saves', 'GoalsAgainst', 'SavePercentage', 'Wins']

# Function to select display names for stats
def select_stats_display(position):
    if position == 'Forward':
        return ['Goals', 'Assists', 'Points', '+/-', 'Shots']
    elif position == 'Defenseman':
        return ['Goals', 'Assists', 'Points', '+/-', 'Blocks']
    else:  # Goalie
        return ['Saves', 'Goals Against', 'Save Percentage', 'Wins']

# Function to filter players based on activity
def filter_active_players(df, position):
    if position == 'Forward':
        return df[df['Goals'] > 0]  # Assuming 'Goals' for forwards
    elif position == 'Defenseman':
        return df[df['Goals'] > 0]  # Assuming 'Goals' for defensemen
    else:  # Goalie
        return df[df['Saves'] > 0]  # For goalies, we check for saves

# Prediction function for NHL stats
def predict_stats(date, schedule, players, stats, encoder, models, selected_player):
    games_today = schedule[schedule['DATE'] == date]

    if games_today.empty:
        st.write(f"No games scheduled for {date}")
        return pd.DataFrame()

    all_predictions = []
    for _, game in games_today.iterrows():
        home_team, away_team = game['HomeTeam'], game['AwayTeam']
        for team, opponent in [(home_team, away_team), (away_team, home_team)]:
            if players[players['Team'] == team]['Player'].str.contains(selected_player, case=False, na=False).any():
                player_features = pd.DataFrame({
                    'Opp': [opponent],
                    'Player': [selected_player]
                })

                player_encoded = encoder.transform(player_features)
                player_df = pd.DataFrame(player_encoded, columns=encoder.get_feature_names_out(['Opp', 'Player']))

                if player_df.shape[1] < list(models.values())[0].model.exog.shape[1]:
                    missing_cols = set(list(models.values())[0].model.exog.columns) - set(player_df.columns)
                    for c in missing_cols:
                        player_df[c] = 0

                prediction = {}
                for stat in stats:
                    if models[stat] is not None:  # Check if the model exists
                        prediction[stat] = round(models[stat].predict(player_df)[0], 4)
                    else:
                        prediction[stat] = np.nan  # or some default value

                all_predictions.append({
                    'Date': date,
                    'Player': selected_player,
                    'Team': team,
                    'Opponent': opponent,
                    **prediction
                })

    return pd.DataFrame(all_predictions)

# Function to get predictions for a specific date
def get_predictions_for_date(date, stats, players, encoder, models, selected_player):
    date = pd.to_datetime(date, format='%m/%d/%Y')
    return predict_stats(date, nhl_schedule_df, players, stats, encoder, models, selected_player)

# Function to fit models with error handling
def fit_models(filtered_df, stats):
    if filtered_df.empty:
        st.write("No data available for model fitting.")
        return {}, None

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    player_features = encoder.fit_transform(filtered_df[['Opp', 'Player']])

    if player_features.shape[0] == 0:
        st.write("No valid data for encoding, cannot proceed with model fitting.")
        return {}, encoder

    X_df = pd.DataFrame(player_features, columns=encoder.get_feature_names_out(['Opp', 'Player']))
    filtered_df = filtered_df.reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    models = {}
    for stat in stats:
        if not filtered_df[stat].isna().all():
            try:
                models[stat] = sm.OLS(filtered_df[stat], X_df).fit()
            except np.linalg.LinAlgError as e:
                st.error(f"Error fitting model for {stat}: {e}. Check data for multicollinearity or insufficient variability.")
                models[stat] = None
            except Exception as e:
                st.error(f"Unexpected error for {stat}: {e}")
                models[stat] = None
        else:
            st.warning(f"No data for {stat}, skipping model fit for this stat.")
            models[stat] = None

    return models, encoder

# Sidebar for view selection and filters
st.sidebar.title("NHL Player Stats Visualization")

# Player Position selection
position = st.sidebar.radio(
    "Select Player Position:",
    ('Forward', 'Defenseman', 'Goalie')
)

# Select the appropriate DataFrame based on position and filter for active players
if position == 'Forward':
    filtered_df = filter_active_players(nhl_player_stats_df, position)
elif position == 'Defenseman':
    filtered_df = filter_active_players(nhl_player_stats_df, position)
else:  # Goalie
    filtered_df = filter_active_players(nhl_player_stats_df, position)

stats = select_stats(position)
stats_display = select_stats_display(position)

logging.info(f"Data for {position} after activity filter: {len(filtered_df)} rows")

# Player selection
player_counts = filtered_df.groupby('Player').size().reset_index(name='GameCount')
valid_players = player_counts[player_counts['GameCount'] > 1]['Player'].tolist()
selected_player = st.sidebar.selectbox('Select a player:', valid_players, format_func=lambda x: x if x else '')

# Filter for the selected player, ensuring more than one game
filtered_df = filtered_df[filtered_df['Player'] == selected_player]
logging.info(f"Filtered data for {selected_player}: {filtered_df.shape}")

if filtered_df.empty:
    selected_player = None  # Set to None if there's no data
else:
    # Date range selection
    min_date_str = filtered_df['Date'].min().strftime('%m/%d/%Y')
    max_date_str = filtered_df['Date'].max().strftime('%m/%d/%Y')
    
    start_date_str = st.sidebar.text_input('Start Date (MM/DD/YYYY):', min_date_str)
    end_date_str = st.sidebar.text_input('End Date (MM/DD/YYYY):', max_date_str)

    try:
        start_date = pd.to_datetime(start_date_str, format='%m/%d/%Y')
        end_date = pd.to_datetime(end_date_str, format='%m/%d/%Y')

        if start_date < filtered_df['Date'].min():
            start_date = filtered_df['Date'].min()
            st.sidebar.warning(f"Start date set to the earliest available date: {start_date.strftime('%m/%d/%Y')}")
        if end_date > filtered_df['Date'].max():
            end_date = filtered_df['Date'].max()
            st.sidebar.warning(f"End date set to the latest available date: {end_date.strftime('%m/%d/%Y')}")

        if start_date > end_date:
            start_date, end_date = end_date, start_date  # Swap dates
            st.sidebar.warning("Start date cannot be after end date. Dates have been swapped.")
    except ValueError:
        st.sidebar.error("Please enter dates in the format MM/DD/YYYY.")
        start_date = filtered_df['Date'].min()
        end_date = filtered_df['Date'].max()

    # Filter data based on date range
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    # Clean data for the selected statistics
    for stat in stats:
        filtered_df[stat] = pd.to_numeric(filtered_df[stat], errors='coerce')
    filtered_df = filtered_df.dropna(subset=stats)

    if filtered_df.empty:
        st.error(f"No data available after filtering for {selected_player}. Adjust your date range or check data.")
    else:
        if any(filtered_df[stat].isna().sum() > 0 for stat in stats):
            st.warning("There are missing values in some of the statistics, which might affect model predictions.")

        models, encoder = fit_models(filtered_df, stats)

        # Show predictions
        if models:
            predictions_df = get_predictions_for_date(start_date, stats, filtered_df, encoder, models, selected_player)
            st.write(predictions_df)
        else:
            st.error("Error fitting models. Please check data or parameters.")
