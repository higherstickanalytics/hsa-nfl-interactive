import streamlit as st
import pandas as pd

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

# Title of the app
st.title("Football Data Viewer")

# Display the first 5 rows of each dataset
st.subheader("First 5 Rows of NFL Schedule Data")
st.dataframe(schedule_df.head())

st.subheader("First 5 Rows of QB Data")
st.dataframe(qb_df.head())

st.subheader("First 5 Rows of RB Data")
st.dataframe(rb_df.head())

st.subheader("First 5 Rows of WR Data")
st.dataframe(wr_df.head())

st.subheader("First 5 Rows of TE Data")
st.dataframe(te_df.head())

st.subheader("First 5 Rows of FB Data")
st.dataframe(fb_df.head())
