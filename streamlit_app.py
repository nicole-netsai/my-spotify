# prompt: write streamlit code for the recommender system above

import streamlit as st
import pandas as pd
import numpy as np
#from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Load your data (replace with your actual data loading)
data = pd.read_csv("/content/data.csv (1).zip") # Assuming this is in your Streamlit environment
# ... (other code for data preprocessing and Spotify API setup if needed)


# Create the Streamlit app
st.title("Music Recommender System")

# Input for song list
song_list = []
num_songs = st.number_input("Number of songs to enter", min_value=1, max_value=10, value=5)
for i in range(num_songs):
    col1, col2 = st.columns(2)
    song_name = col1.text_input(f"Song Name {i+1}", key=f"song_name_{i}")
    song_year = col2.number_input(f"Year {i+1}", min_value=1900, max_value=2024, key=f"song_year_{i}")
    song_list.append({'name': song_name, 'year': song_year})


if st.button("Recommend Songs"):
    if all(song['name'] and song['year'] for song in song_list):
      # ... Your recommend_songs function here
      recommendations = recommend_songs(song_list, data)
      st.write("Recommended Songs:")
      for song in recommendations:
          st.write(f"- {song['name']} ({song['year']}) by {', '.join(song['artists'])}")
    else:
        st.write("Please fill in all song details.")
