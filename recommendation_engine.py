import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(song_idx, df, tfidf_matrix, num_recommendations=5):
    """
    Get song recommendations based on cosine similarity of TF-IDF vectors
    
    Parameters:
    song_idx: Index of the song to get recommendations for
    df: Dataframe containing song data
    tfidf_matrix: TF-IDF matrix of song features
    num_recommendations: Number of recommendations to return
    
    Returns:
    List of (index, similarity score) tuples for the most similar songs
    """
    # Calculate cosine similarity between the selected song and all other songs
    cosine_sim = cosine_similarity(tfidf_matrix[song_idx:song_idx+1], tfidf_matrix).flatten()
    
    # Get the indices of the most similar songs (excluding the selected song itself)
    similar_song_indices = np.argsort(cosine_sim)[::-1][1:num_recommendations+1]
    
    # Get the similarity scores of the most similar songs
    similarity_scores = [(idx, cosine_sim[idx]) for idx in similar_song_indices]
    
    return similarity_scores

def explain_recommendation(song_idx1, song_idx2, df, tfidf_matrix, tfidf_feature_names, top_n=5):
    """
    Explain why a song is recommended based on common features
    
    Parameters:
    song_idx1: Index of the first song (selected song)
    song_idx2: Index of the second song (recommended song)
    df: Dataframe containing song data
    tfidf_matrix: TF-IDF matrix of song features
    tfidf_feature_names: Feature names from the TF-IDF vectorizer
    top_n: Number of top features to return
    
    Returns:
    Dictionary of top matching features and their importance scores
    """
    # Get the TF-IDF vectors for both songs
    song1_vec = tfidf_matrix[song_idx1].toarray().flatten()
    song2_vec = tfidf_matrix[song_idx2].toarray().flatten()
    
    # Calculate element-wise product to find common features
    common_features = song1_vec * song2_vec
    
    # Get indices of top features
    top_indices = np.argsort(common_features)[::-1][:top_n]
    
    # Create dictionary of feature names and scores
    explanation = {tfidf_feature_names[idx]: common_features[idx] for idx in top_indices if common_features[idx] > 0}
    
    return explanation
