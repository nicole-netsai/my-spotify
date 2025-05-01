import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#from sklearn.metrics import euclidean_distances
#from scipy.spatial.distance import cdist
#from yellowbrick.target import FeatureCorrelation
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Spotify Music Recommendation System", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    genre_data = pd.read_csv('data_by_genres.csv')
    year_data = pd.read_csv('data_by_year.csv')
    return data, genre_data, year_data

data, genre_data, year_data = load_data()

# Add decade column
def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Home", "Data Exploration", "Feature Correlation", 
                           "Music Over Time", "Recommendation System"])

if options == "Home":
    st.title("Spotify Music Recommendation System")
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=300)
    st.write("""
    This app provides insights into Spotify music data and offers song recommendations based on audio features.
    """)
    
    st.subheader("About the Dataset")
    st.write("The dataset contains audio features for tracks, genres, and years from Spotify.")
    
    st.subheader("Features Included")
    st.write("""
    - Valence
    - Year
    - Acousticness
    - Danceability
    - Duration (ms)
    - Energy
    - Explicit
    - Instrumentalness
    - Key
    - Liveness
    - Loudness
    - Mode
    - Popularity
    - Speechiness
    - Tempo
    """)

elif options == "Data Exploration":
    st.title("Data Exploration")
    
    st.subheader("Track Data Overview")
    st.write(data.head())
    
    st.subheader("Genre Data Overview")
    st.write(genre_data.head())
    
    st.subheader("Year Data Overview")
    st.write(year_data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())

elif options == "Feature Correlation":
    st.title("Feature Correlation with Popularity")
    
    feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                    'duration_ms', 'explicit', 'key', 'mode', 'year']
    
    X, y = data[feature_names], data['popularity']
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20))
    visualizer = FeatureCorrelation(labels=feature_names)
    visualizer.fit(X, y)
    visualizer.show()
    
    st.pyplot(fig)

elif options == "Music Over Time":
    st.title("Music Trends Over Decades")
    
    # Countplot of songs per decade
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.countplot(data=data, y='decade', ax=ax, order=sorted(data['decade'].unique()))
    ax.set_title("Number of Songs per Decade")
    ax.set_xlabel("Count")
    ax.set_ylabel("Decade")
    st.pyplot(fig)
    
    # Audio features over time
    st.subheader("Audio Feature Trends Over Years")
    
    feature = st.selectbox("Select feature to analyze:", 
                         ['danceability', 'energy', 'acousticness', 
                          'instrumentalness', 'liveness', 'valence'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=year_data, x='year', y=feature, ax=ax)
    ax.set_title(f"{feature.capitalize()} Trend Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel(feature.capitalize())
    st.pyplot(fig)

elif options == "Recommendation System":
    st.title("Music Recommendation System")
    
    st.write("""
    This system recommends songs based on their audio features using clustering.
    """)
    
    # Select features for clustering
    st.subheader("Select Features for Recommendation")
    selected_features = st.multiselect(
        "Choose audio features to consider:",
        ['acousticness', 'danceability', 'energy', 'instrumentalness',
         'liveness', 'loudness', 'speechiness', 'tempo', 'valence'],
        default=['danceability', 'energy', 'valence']
    )
    
    # Number of clusters
    n_clusters = st.slider("Number of clusters:", min_value=5, max_value=20, value=10)
    
    # Filter data
    song_data = data[['artists', 'name', 'popularity'] + selected_features].dropna()
    
    if st.button("Generate Recommendations"):
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(song_data[selected_features])
        
        # Cluster the songs
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        song_data['cluster'] = clusters
        
        # Show cluster distribution
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=song_data, x='cluster', ax=ax)
        st.pyplot(fig)
        
        # Select a cluster to explore
        cluster_num = st.selectbox("Select a cluster to explore:", range(n_clusters))
        
        # Show songs from selected cluster
        st.subheader(f"Songs in Cluster {cluster_num}")
        cluster_songs = song_data[song_data['cluster'] == cluster_num]
        st.write(cluster_songs[['artists', 'name', 'popularity']].sort_values('popularity', ascending=False).head(20))
        
        # Visualize cluster characteristics
        st.subheader("Cluster Characteristics")
        cluster_means = song_data.groupby('cluster')[selected_features].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_means.T, cmap='YlGnBu', ax=ax)
        ax.set_title("Average Feature Values per Cluster")
        st.pyplot(fig)
        
        # Dimensionality reduction visualization
        st.subheader("Cluster Visualization (t-SNE)")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)
        
        song_data['tsne-2d-one'] = tsne_results[:,0]
        song_data['tsne-2d-two'] = tsne_results[:,1]
        
        fig, ax = plt.subplots(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="cluster",
            palette=sns.color_palette("hls", n_clusters),
            data=song_data,
            legend="full",
            alpha=0.7,
            ax=ax
        )
        ax.set_title("t-SNE Visualization of Song Clusters")
        st.pyplot(fig)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
