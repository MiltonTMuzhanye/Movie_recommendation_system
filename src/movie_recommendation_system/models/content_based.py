import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, similarity_matrix: np.ndarray, movies_df: pd.DataFrame):
        self.similarity_matrix = similarity_matrix
        self.movies_df = movies_df
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
        self.index_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(movies_df['movieId'])}
    
    def recommend_similar_movies(
        self, 
        movie_id: int, 
        top_n: int = 10,
        min_similarity: float = 0.1
    ) -> pd.DataFrame:
        """Recommend similar movies based on content"""
        if movie_id not in self.movie_id_to_index:
            raise ValueError(f"Movie ID {movie_id} not found in dataset")
        
        movie_idx = self.movie_id_to_index[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the query movie itself)
        top_indices = [i for i, score in sim_scores[1:top_n+1] if score >= min_similarity]
        
        recommendations = []
        for idx in top_indices:
            movie_id_rec = self.index_to_movie_id[idx]
            similarity_score = self.similarity_matrix[movie_idx, idx]
            
            movie_data = self.movies_df[self.movies_df['movieId'] == movie_id_rec].iloc[0]
            recommendations.append({
                'movieId': movie_id_rec,
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'similarity_score': similarity_score
            })
        
        return pd.DataFrame(recommendations)
    
    def find_movie_by_title(self, title_query: str) -> pd.DataFrame:
        """Find movies by title query"""
        mask = self.movies_df['title'].str.contains(title_query, case=False, na=False)
        return self.movies_df[mask]