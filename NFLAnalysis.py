import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Paths to the files
schedule_path = 'data/NFL_Schedule.csv'
qb_path = 'data/football_data/combined_qb_football_game_logs.csv'
rb_path = 'data/football_data/combined_rb_football_game_logs.csv'
wr_path = 'data/football_data/combined_wr_football_game_logs.csv'
te_path = 'data/football_data/combined_te_football_game_logs.csv'
fb_path = 'data/football_data/combined_fb_football_game_logs.csv'

# Load the data
schedule_df = pd.read_csv(schedule_path, parse_dates=['Date'], dayfirst=False)
qb_df = pd.read_csv(qb_path)
rb_df = pd.read_csv(rb_path)
wr_df = pd.read_csv(wr_path)
te_df = pd.read_csv(te_path)
fb_df = pd.read_csv(fb_path)

# App Title
st.title("Football Data Viewer with Pie and Time-Series Charts")
st.write("Data from [NFL](https://www.nfl.com/)")

# Sidebar: select position
position = st.sidebar.radio("Select Player Position", ['QB', 'RB', 'WR', 'TE', 'FB'])

# Select appropriate dataframe and stats based on position
if position == 'QB':
    df = qb_df
    stats = ['Yds', 'TD', 'Int', 'Cmp', 'Attempts']  # Adjusted QB stat names
    stat_names = ['Passing Yards', 'Touchdowns', 'Interceptions', 'Completions', 'Attempts']
elif position == 'RB':
    df = rb_df
    stats = ['Yds', 'TD', 'Att']
    stat_names = ['Rushing Yards', 'Touchdowns', 'Carries']
elif position == 'WR':
    df = wr_df
    stats = ['Yds', 'TD', 'Rec', 'Tgt']
    stat_names = ['Receiving Yards', 'Touchdowns', 'Receptions', 'Targets']
elif position == 'TE':
    df = te_df
    stats = ['Yds', 'TD', 'Rec', 'Tgt']
    stat_names = ['Receiving Yards', 'Touchdowns', 'Receptions', 'Targets']
else:
    df = fb_df
    stats = ['Yds', 'TD', 'Att']
    stat_names = ['Rushing Yards', 'Touchdowns', 'Carries']

# Sidebar: player and stat selection
player_list = df['Player'].dropna().unique().tolist()
selected_player = st.sidebar.selectbox("Select a player:", sorted(player_list))
selected_stat_display = st.sidebar.selectbox("Select a statistic:", stat_names)
selected_stat = stats[stat_names.index(selected_stat_display)]

# Sidebar: date filter
min_date = df['Date'].min()
max_date = df['Date'].max()

start_date = pd.to_datetime(st.sidebar.date_input("Start Date", min_value=min_date, value=min_date))
end_date = pd.to_datetime(st.sidebar.date_input("End Date", max_value=max_date, value=max_date))

# Filter data
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
player_df = df[df['Player'] == selected_player]
player_df[selected_stat] = pd.to_numeric(player_df[selected_stat], errors='coerce').dropna()

# Histogram threshold
max_val = player_df[selected_stat].max()
default_thresh = player_df[selected_stat].median()
threshold = st.sidebar.number_input("Set Threshold", min_value=0.0, max_value=float(max_val), value=float(default_thresh), step=0.5)

# Pie Chart: Stat Distribution with threshold colors
st.subheader(f"{selected_stat_display} Distribution for {selected_player}")

stat_counts = player_df[selected_stat].value_counts().sort_index()
labels = [f"{int(val)}" if val == int(val) else f"{val:.1f}" for val in stat_counts.index]
sizes = stat_counts.values

# Assign red for below threshold, green for above, gray for equal
colors = []
color_categories = {'green': 0, 'red': 0, 'gray': 0}
for val, count in zip(stat_counts.index, stat_counts.values):
    if val > threshold:
        colors.append('green')
        color_categories['green'] += count
    elif val < threshold:
        colors.append('red')
        color_categories['red'] += count
    else:
        colors.append('gray')
        color_categories['gray'] += count

fig1, ax1 = plt.subplots()
wedges, texts, autotexts = ax1.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 10}
)
ax1.axis('equal')
ax1.set_title(f"{selected_stat_display} Value Distribution")
st.pyplot(fig1)

# Display color category percentages in a cleaner table format
total_entries = sum(color_categories.values())
if total_entries > 0:
    st.markdown("**Pie Chart Color Breakdown:**")
    breakdown_data = {
        'Color': ['🟩 Green', '🟥 Red', '⬜ Gray'],
        'Category': [
            f"Above {threshold} {selected_stat_display}",
            f"Below {threshold} {selected_stat_display}",
            f"At {threshold} {selected_stat_display}"
        ],
        'Count': [color_categories['green'], color_categories['red'], color_categories['gray']],
        'Percentage': [
            f"{color_categories['green'] / total_entries:.2%}",
            f"{color_categories['red'] / total_entries:.2%}",
            f"{color_categories['gray'] / total_entries:.2%}"
        ]
    }
    breakdown_df = pd.DataFrame(breakdown_data)
    st.table(breakdown_df)
else:
    st.write("No data available to display pie chart.")

# Time-Series Histogram
st.subheader(f"{selected_stat_display} Over Time for {selected_player}")
fig2, ax2 = plt.subplots(figsize=(12, 6))
data = player_df[['Date', selected_stat]].dropna()
bars = ax2.bar(data['Date'], data[selected_stat], color='gray', edgecolor='black')

count_above = 0
for bar, val in zip(bars, data[selected_stat]):
    if val > threshold:
        bar.set_color('green')
        count_above += 1
    elif val < threshold:
        bar.set_color('red')
    else:
        bar.set_color('gray')
        count_above += 1

ax2.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
ax2.set_xlabel("Date")
ax2.set_ylabel(selected_stat_display)
ax2.set_title(f"{selected_stat_display} Over Time")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
ax2.legend()
st.pyplot(fig2)

# Proportion summary
total_games = len(data)
if total_games > 0:
    st.write(f"Games at or above threshold: {count_above}/{total_games} ({count_above / total_games:.2%})")
else:
    st.write("No data available in selected date range.")
