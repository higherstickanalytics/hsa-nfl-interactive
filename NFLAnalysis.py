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
nfl_schedule_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/NFL_Schedule.csv')
nfl_fb_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/football_data/combined_fb_football_game_logs.csv')
nfl_qb_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/football_data/combined_qb_football_game_logs.csv')
nfl_rb_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/football_data/combined_rb_football_game_logs.csv')
nfl_te_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/football_data/combined_te_football_game_logs.csv')
nfl_wr_df = pd.read_csv(
    'C:/Users/nickk/OneDrive - Bentley University/Fall 2024/CS 230 - Introduction to Programming with Python/pythonProject/Sports Analytics/Football data/football_data/combined_wr_football_game_logs.csv')

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


# Function to select display names for stats
def select_stats_display(position):
    if position == 'QB':
        return ['Completions', 'Attempts', 'Yards', 'Touchdowns', 'Interceptions', 'QB Rating']
    elif position == 'RB':
        return ['Rush Attempts', 'Rush Yards', 'Rush Touchdowns']
    elif position in ['WR', 'TE']:
        return ['Targets', 'Receptions', 'Receiving Yards']
    else:  # FB
        return ['Solo Tackles', 'Assisted Tackles', 'Combined Tackles']


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


# Function to get predictions for a specific date
def get_predictions_for_date(date, stats, players, encoder, models, selected_player):
    date = pd.to_datetime(date, format='%m/%d/%Y')
    return predict_stats(date, nfl_schedule_df, players, stats, encoder, models, selected_player)


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
stats_display = select_stats_display(position)

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

if 'filtered_df' in locals() and not filtered_df.empty and selected_player is not None:
    # Visualization setup
    view = st.sidebar.radio(
        "Select Visualization Type:",
        ('Histograms', 'Pie Chart')
    )

    # Statistic Filter for Visualization
    selected_stat_index = st.sidebar.selectbox('Select a statistic for visualization:',
                                               stats_display,
                                               format_func=lambda x: x if x else '')
    selected_stat = stats[stats_display.index(selected_stat_index)]

    if view == 'Histograms':
        player_data = filtered_df[filtered_df['Player'] == selected_player][selected_stat].dropna()

        if not player_data.empty:
            max_value = player_data.max()
            default_value = player_data.median()
            threshold = st.number_input(f'Set Threshold for {selected_stat_index}:', min_value=0.0,
                                        max_value=float(max_value) if pd.notna(max_value) else 1.0,
                                        value=float(default_value) if pd.notna(default_value) else 0.0,
                                        step=0.5)
        else:
            st.write(f"No valid data for {selected_stat_index}.")
            threshold = 0.0

        # Distribution Histogram
        st.subheader(f'{selected_stat_index} Distribution Histogram')

        fig1, ax1 = plt.subplots()
        lock = threading.Lock()

        green_freq = 0
        red_freq = 0
        grey_freq = 0
        total_freq = len(player_data)

        with lock:
            if not player_data.empty:
                n, bins, patches = ax1.hist(player_data, bins=20, edgecolor='black')

                for value in player_data:
                    if value > threshold:
                        green_freq += 1
                    elif value < threshold:
                        red_freq += 1
                    else:
                        grey_freq += 1

                for patch, left, right in zip(patches, bins[:-1], bins[1:]):
                    mean = (left + right) / 2
                    if mean > threshold:
                        patch.set_facecolor('green')
                    elif mean < threshold:
                        patch.set_facecolor('red')
                    else:
                        patch.set_facecolor('grey')

                ax1.set_title(f'{selected_stat_index} Distribution for {selected_player}')
                ax1.set_xlabel(selected_stat_index)
                ax1.set_ylabel('Frequency')

            else:
                ax1.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)

        st.pyplot(fig1)

        if total_freq > 0:
            over_percentage = (green_freq / total_freq) * 100
            under_percentage = (red_freq / total_freq) * 100
            equal_percentage = (grey_freq / total_freq) * 100
        else:
            st.write(f"No data available for Over/Under Percentage calculation in {selected_stat_index} Distribution.")

        # Time Series Histogram
        st.subheader(f'{selected_stat_index} Time Series Histogram')

        fig2, ax2 = plt.subplots(figsize=(12, 6))

        with lock:
            if not player_data.empty:
                player_data_time = filtered_df[filtered_df['Player'] == selected_player][['Date', selected_stat]].dropna()

                bars = ax2.bar(player_data_time['Date'], player_data_time[selected_stat], color='grey', edgecolor='black')

                for bar in bars:
                    height = bar.get_height()
                    if height > threshold:
                        bar.set_color('green')
                    elif height < threshold:
                        bar.set_color('red')
                    else:
                        bar.set_color('grey')

                ax2.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
                ax2.set_title(f'{selected_stat_index} Over Time for {selected_player}')
                ax2.set_xlabel('Date')
                ax2.set_ylabel(selected_stat_index)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                ax2.legend()

            else:
                ax2.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center',
                         transform=ax2.transAxes)

        st.pyplot(fig2)

        # Calculate percentages for Time Series
        try:
            player_data_time = filtered_df[filtered_df['Player'] == selected_player][['Date', selected_stat]].dropna()
            green_count_time = sum(player_data_time[selected_stat] > threshold)
            red_count_time = sum(player_data_time[selected_stat] < threshold)
            grey_count_time = sum(player_data_time[selected_stat] == threshold)
            total_count_time = green_count_time + red_count_time + grey_count_time

            if total_count_time > 0:
                over_percentage_time = (green_count_time / total_count_time) * 100
                under_percentage_time = (red_count_time / total_count_time) * 100
                equal_percentage_time = (grey_count_time / total_count_time) * 100
                st.write(f"Over/Under Percentage for {selected_stat_index}:")
                st.write(f"- Over Percentage: {over_percentage_time:.2f}%")
                st.write(f"- Under Percentage: {under_percentage_time:.2f}%")
                st.write(f"- Equal Percentage: {equal_percentage_time:.2f}%")
            else:
                st.write("No Data Available")
        except:
            st.write("No Data Available")

    elif view == 'Pie Chart':
        st.subheader(f'{selected_stat_index} Pie Chart of Player Performance')

        player_data_pie = filtered_df[filtered_df['Player'] == selected_player][selected_stat].dropna()

        if not filtered_df[selected_stat].empty:
            max_value = filtered_df[selected_stat].max()
            default_value = filtered_df[selected_stat].median()
            threshold = st.number_input(f'Set Threshold for {selected_stat_index} Pie Chart:', min_value=0.0,
                                        max_value=float(max_value) if pd.notna(max_value) else 1.0,
                                        value=float(default_value) if pd.notna(default_value) else 0.0,
                                        step=0.5)
        else:
            st.write(f"No valid data for {selected_stat_index}.")
            threshold = 0.0

        above = sum(player_data_pie > threshold)
        below = sum(player_data_pie < threshold)
        equal = sum(player_data_pie == threshold)

        sizes = [above, below, equal]
        labels = ['Above Threshold', 'Below Threshold', 'Equal to Threshold']
        colors = ['green', 'red', 'grey']
        explode = (0.1, 0.1, 0.1)

        fig3, ax3 = plt.subplots()
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')

        plt.title(f'Performance Distribution for {selected_stat_index} by {selected_player}')
        st.pyplot(fig3)

    prediction_date = st.text_input("Enter Date for Predictions (MM/DD/YYYY):",
                                    value=pd.Timestamp.now().strftime('%m/%d/%Y'))

    # Display stats horizontally
    stats_text = f"Stats for {selected_player} from {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}:"
    st.write(stats_text, unsafe_allow_html=True)
    if not filtered_df[filtered_df['Player'] == selected_player].empty:
        stats_summary = filtered_df[filtered_df['Player'] == selected_player][[selected_stat]].describe().drop('count',
                                                                                                               axis=0).T
        stats_summary.index = [selected_stat_index]
        st.write(stats_summary.to_html(), unsafe_allow_html=True)
    else:
        st.write("No stats available for the selected criteria.")

    # For predictions
    st.subheader(f"Predictions for {selected_player} on {pd.to_datetime(prediction_date).strftime('%m/%d/%Y')}")

    if st.button("Get Predictions") and 'models' in locals():
        predictions = get_predictions_for_date(prediction_date, stats, filtered_df, encoder, models, selected_player)

        if not predictions.empty:
            # Rename columns to match display names
            predictions.columns = ['Date', 'Player', 'Team', 'Opponent'] + stats_display

            # Handle NaT values before formatting
            predictions['Date'] = pd.to_datetime(predictions['Date'], errors='coerce').fillna(
                pd.Timestamp('1970-01-01')).dt.strftime('%m/%d/%Y')
            predictions.set_index('Date', inplace=True)
            st.dataframe(predictions)
        else:
            st.write(
                f"No predictions could be made for {selected_player} on {pd.to_datetime(prediction_date).strftime('%m/%d/%Y')}.")
    else:
        st.write("No models available for predictions.")
else:
    st.error("No data available to display or analyze. Please select a different player or check your date range.")
