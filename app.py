import streamlit as st
import pandas as pd
import numpy as np
from recommendation_engine import get_recommendations
from utils import load_data, preprocess_data

# Page configuration
st.set_page_config(
    page_title="Bollywood Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# Main title
st.title("🎵 Hindi Bollywood Song Recommender")
st.markdown("Find songs similar to your favorites based on song features!")

# Load and preprocess data
@st.cache_data
def get_processed_data():
    df = load_data()
    return preprocess_data(df)

# Get data
try:
    df, tfidf_matrix, tfidf_feature_names = get_processed_data()
    
    # Sidebar for filtering options
    st.sidebar.header("Filter Options")
    
    # Filter by artist
    all_artists = ['All Artists'] + sorted(df['artist'].unique().tolist())
    selected_artist = st.sidebar.selectbox("Select Artist", all_artists)
    
    # Filter by year range
    years = df['year'].dropna().astype(int).unique()
    min_year, max_year = int(min(years)), int(max(years))
    year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Filter by mood
    all_moods = ['All Moods'] + sorted(df['mood'].unique().tolist())
    selected_mood = st.sidebar.selectbox("Select Mood", all_moods)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_artist != 'All Artists':
        filtered_df = filtered_df[filtered_df['artist'] == selected_artist]
    
    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    
    if selected_mood != 'All Moods':
        filtered_df = filtered_df[filtered_df['mood'] == selected_mood]
    
    if len(filtered_df) == 0:
        st.warning("No songs match your filters. Please adjust your criteria.")
    else:
        # Search box
        search_query = st.text_input("Search for a song by title:", "")
        
        # Filter by search query
        if search_query:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False)]
        
        # Show number of results
        st.write(f"Showing {len(filtered_df)} songs")
        
        # Display filtered songs
        if len(filtered_df) > 0:
            # Display songs in a scrollable container
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Select a Song")
                    song_list = filtered_df[['title', 'artist', 'year']].apply(
                        lambda row: f"{row['title']} - {row['artist']} ({int(row['year'])})", axis=1
                    ).tolist()
                    
                    selected_song_idx = st.selectbox(
                        "Choose a song to get recommendations:",
                        range(len(song_list)),
                        format_func=lambda i: song_list[i]
                    )
                    
                    selected_song = filtered_df.iloc[selected_song_idx]
                    
                    # Display selected song details
                    st.subheader("Selected Song Details")
                    song_details = {
                        "Title": selected_song['title'],
                        "Artist": selected_song['artist'],
                        "Album": selected_song['album'],
                        "Year": int(selected_song['year']),
                        "Mood": selected_song['mood'],
                        "Genre": selected_song['genre']
                    }
                    
                    for key, value in song_details.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    if st.button("Get Recommendations"):
                        st.subheader("Recommended Songs")
                        recommendations = get_recommendations(
                            selected_song.name, df, tfidf_matrix
                        )
                        
                        # Display recommendations
                        for i, (idx, score) in enumerate(recommendations):
                            rec_song = df.loc[idx]
                            
                            with st.expander(f"{i+1}. {rec_song['title']} - {rec_song['artist']} ({int(rec_song['year'])})"):
                                st.write(f"**Album:** {rec_song['album']}")
                                st.write(f"**Mood:** {rec_song['mood']}")
                                st.write(f"**Genre:** {rec_song['genre']}")
                                st.write(f"**Similarity Score:** {score:.2f}")
        else:
            st.info("No songs match your search query. Try another search term.")
    
    # Add information about the recommendation system
    with st.expander("About this Recommender System"):
        st.write("""
        This is a content-based recommendation system for Hindi Bollywood songs. 
        It analyzes song features such as artist, genre, mood, lyrics, and other metadata 
        to find songs that are similar to your selection.
        
        The system uses TF-IDF vectorization and cosine similarity to calculate how similar 
        songs are to each other.
        """)

except Exception as e:
    st.error(f"An error occurred while loading or processing the data: {e}")
    st.info("Please check that the data file exists and is correctly formatted.")
