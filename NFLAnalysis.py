import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import threading
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NFL data
nfl_schedule_df = pd.read_csv('NFL_Schedule.csv')
nfl_fb_df = pd.read_csv('combined_fb_football_game_logs.csv')
nfl_qb_df = pd.read_csv('combined_qb_football_game_logs.csv')
nfl_rb_df = pd.read_csv('combined_rb_football_game_logs.csv')
nfl_te_df = pd.read_csv('combined_te_football_game_logs.csv')
nfl_wr_df = pd.read_csv('combined_wr_football_game_logs.csv')

# Convert dates to datetime for easier matching
for df in [nfl_fb_df, nfl_qb_df, nfl_rb_df, nfl_te_df, nfl_wr_df]:
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

nfl_schedule_df['DATE'] = pd.to_datetime(nfl_schedule_df['Date'], format='%m/%d/%Y', errors='coerce')

# Function to select relevant stats based on position
def select_stats(position):
    if position == 'QB':
        return ['Cmp', 'Attempts', 'Yds', 'TD', 'Int', 'Rate']
    elif position == 'RB':
        return ['Attempts', 'Yds', 'TD']
    elif position in ['WR', 'TE']:
        return ['Tgt', 'Rec', 'Yds']
    else:  # FB
        return ['Solo', 'Ast', 'Comb']

# Function to filter players based on activity with condition for at least one completion/attempt/tackle/etc.
def filter_active_players(df, position):
    if position == 'QB':
        return df[df['Cmp'] > 0]  # Assuming 'Cmp' for Completions
    elif position == 'RB':
        return df[df['Attempts'] > 0]  # For Running Backs, we check for rush attempts
    elif position == 'WR':
        return df[df['Rec'] > 0]  # For Wide Receivers, we check for receptions
    elif position == 'TE':
        return df[df['Rec'] > 0]  # Same for Tight Ends
    else:  # FB, assuming 'Comb' for combined tackles
        return df[df['Comb'] > 0]  # For Fullbacks, we might check for tackles

# Prediction function for NFL stats
def predict_stats(date, schedule, players, stats, encoder, models, selected_player):
    games_today = schedule[schedule['DATE'] == date]

    if games_today.empty:
        st.write(f"No games scheduled for {date}")
        return pd.DataFrame()

    all_predictions = []
    for _, game in games_today.iterrows():
        home_team, away_team = game['HOME'], game['AWAY']
        for team, opponent in [(home_team, away_team), (away_team, home_team)]:
            if players[players['Tm'] == team]['Player'].str.contains(selected_player, case=False, na=False).any():
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

# Sidebar for view selection and filters
st.sidebar.title("NFL Player Stats Visualization")

# Player Position selection
position = st.sidebar.radio(
    "Select Player Position:",
    ('QB', 'RB', 'WR', 'TE', 'FB')
)

# Select the appropriate DataFrame based on position and filter for active players
if position == 'QB':
    filtered_df = filter_active_players(nfl_qb_df, position)
elif position == 'RB':
    filtered_df = filter_active_players(nfl_rb_df, position)
elif position == 'WR':
    filtered_df = filter_active_players(nfl_wr_df, position)
elif position == 'TE':
    filtered_df = filter_active_players(nfl_te_df, position)
else:  # FB
    filtered_df = filter_active_players(nfl_fb_df, position)

stats = select_stats(position)

logging.info(f"Data for {position} after activity filter: {len(filtered_df)} rows")

# Player selection
player_counts = filtered_df.groupby('Player').size().reset_index(name='GameCount')
valid_players = player_counts[player_counts['GameCount'] > 1]['Player'].tolist()
selected_player = st.sidebar.selectbox('Select a player:', valid_players, format_func=lambda x: x if x else '')

# Filter for the selected player, ensuring more than one game
filtered_df = filtered_df[filtered_df['Player'].isin(valid_players)]
filtered_df = filtered_df[filtered_df['Player'] == selected_player]
logging.info(f"Filtered data for {selected_player}: {filtered_df.shape}")

if filtered_df.empty:
    selected_player = None  # Set to None if there's no data
else:
    # Date range selection
    if not filtered_df.empty:
        min_date_str = filtered_df['Date'].min().strftime('%m/%d/%Y')
        max_date_str = filtered_df['Date'].max().strftime('%m/%d/%Y')
    else:
        st.write("No valid date entries for the selected criteria after player selection.")
        min_date_str = "No valid dates"
        max_date_str = "No valid dates"

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

        st.sidebar.write(
            f"Showing data from **{start_date.strftime('%m/%d/%Y')}** to **{end_date.strftime('%m/%d/%Y')}**.")
    except ValueError:
        st.sidebar.error("Please enter dates in the format MM/DD/YYYY.")
        if not filtered_df.empty:
            start_date = filtered_df['Date'].min()
            end_date = filtered_df['Date'].max()
        else:
            start_date = pd.Timestamp('1970-01-01')
            end_date = pd.Timestamp('1970-01-01')

    # Filter data based on date range
    logging.info(f"Before filtering: {len(filtered_df)} rows")
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    logging.info(f"After filtering by date range: {filtered_df.shape}")

    # Clean data for the selected statistics
    for stat in stats:
        filtered_df[stat] = pd.to_numeric(filtered_df[stat], errors='coerce')
    filtered_df = filtered_df.dropna(subset=stats)

    if filtered_df.empty:
        st.error(f"No data available after filtering for {selected_player}. Adjust your date range or check data.")
    else:
        if any(filtered_df[stat].empty for stat in stats):
            st.error(f"Empty data for one or more stats for {selected_player}. Cannot fit models.")
        else:
            models, encoder = fit_models(filtered_df, stats)

# Main content area
st.title(f'NFL {position} Stats Visualization')
st.write("All Data has been collected using [Pro Football Reference](https://www.pro-football-reference.com/) for NFL.")

# Display predictions for selected player and date
prediction_date = st.text_input("Enter Date for Predictions (MM/DD/YYYY):", value=pd.Timestamp.now().strftime('%m/%d/%Y'))

if prediction_date:
    predictions = predict_stats(pd.to_datetime(prediction_date), nfl_schedule_df, filtered_df, stats, encoder, models, selected_player)
    if not predictions.empty:
        st.write(predictions)
    else:
        st.write(f"No predictions available for {selected_player} on {prediction_date}.")
