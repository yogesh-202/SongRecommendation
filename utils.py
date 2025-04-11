import pandas as pd
import numpy as np
import re
import os

def load_data():
    """
    Load the Bollywood songs data from CSV file
    """
    try:
        # Load the dataset
        df = pd.read_csv('data/bollywood_songs.csv')
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Could not find the song dataset file. Please ensure 'data/bollywood_songs.csv' exists.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean_text(text):
    """
    Clean and normalize text data
    """
    if isinstance(text, str):
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    return ''

def preprocess_data(df):
    """
    Preprocess the dataframe for recommendation system
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Fill NaN values
    df_copy['year'] = df_copy['year'].fillna(2000)
    df_copy['mood'] = df_copy['mood'].fillna('unknown')
    df_copy['genre'] = df_copy['genre'].fillna('unknown')
    df_copy['album'] = df_copy['album'].fillna('unknown')
    df_copy['lyrics'] = df_copy['lyrics'].fillna('')
    
    # Clean text columns
    for col in ['title', 'artist', 'album', 'mood', 'genre', 'lyrics']:
        df_copy[col] = df_copy[col].apply(clean_text)
    
    # Create a combined features column for TF-IDF
    df_copy['combined_features'] = (
        df_copy['title'] + ' ' +
        df_copy['artist'] + ' ' +
        df_copy['album'] + ' ' +
        df_copy['mood'] + ' ' +
        df_copy['genre'] + ' ' +
        df_copy['lyrics']
    )
    
    # Import TF-IDF vectorizer here to avoid circular imports
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['combined_features'])
    
    # Get feature names for explanation
    tfidf_feature_names = tfidf.get_feature_names_out()
    
    return df_copy, tfidf_matrix, tfidf_feature_names
