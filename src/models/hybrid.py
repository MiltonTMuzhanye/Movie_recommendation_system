import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

class HybridRecommender:
    def __init__(
        self,
        content_recommender,
        collab_recommender,
        movies_df: pd.DataFrame,
        weights: Dict[str, float] = None
    ):
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
        self.movies_df = movies_df
        self.weights = weights or {'content': 0.5, 'collab': 0.5}
        
        # Create mappings
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
    
    def recommend(
        self,
        user_id: int,
        reference_movie_id: Optional[int] = None,
        reference_movie_title: Optional[str] = None,
        ratings_df: pd.DataFrame = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Generate hybrid recommendations"""
        # Get content-based scores
        if reference_movie_id is None and reference_movie_title is not None:
            # Find movie by title
            movie_matches = self.movies_df[
                self.movies_df['title'].str.contains(reference_movie_title, case=False, na=False)
            ]
            if len(movie_matches) > 0:
                reference_movie_id = movie_matches.iloc[0]['movieId']
        
        if reference_movie_id is None:
            raise ValueError("Either reference_movie_id or reference_movie_title must be provided")
        
        # Get content similarity scores for all movies
        ref_idx = self.movie_id_to_index[reference_movie_id]
        content_scores = self.content_rec.similarity_matrix[ref_idx]
        
        # Get collaborative scores for all movies
        collab_scores = []
        for movie_id in self.movies_df['movieId']:
            pred = self.collab_rec.model.predict(user_id, movie_id)
            collab_scores.append(pred.est)
        collab_scores = np.array(collab_scores)
        
        # Normalize scores
        content_norm = self._normalize_scores(content_scores)
        collab_norm = self._normalize_scores(collab_scores)
        
        # Combine scores
        hybrid_scores = (
            self.weights['content'] * content_norm +
            self.weights['collab'] * collab_norm
        )
        
        # Get top recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            movie_id = self.movies_df.iloc[idx]['movieId']
            movie_data = self.movies_df.iloc[idx]
            
            recommendations.append({
                'movieId': movie_id,
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'content_score': content_norm[idx],
                'collab_score': collab_norm[idx],
                'hybrid_score': hybrid_scores[idx]
            })
        
        return pd.DataFrame(recommendations)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)