import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger

class PopularityModel:
    """
    Simple popularity-based recommendation model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize popularity model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.popularity_scores = None
        self.movie_ids = None
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Fit the popularity model
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
        """
        logger.info("Fitting popularity model")
        
        # Calculate popularity metrics
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'rating_mean']
        
        # Apply popularity weight
        # Weighted by both count and mean rating
        min_ratings = self.config.get('models', {})
            .get('popularity', {})
            .get('min_ratings', 5)
        
        # Filter movies with minimum ratings
        valid_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        # Calculate popularity score (weighted average)
        # Using Bayesian average
        global_mean = ratings_df['rating'].mean()
        count_weight = 0.5  # Weight for number of ratings
        
        valid_movies['popularity_score'] = (
            (1 - count_weight) * valid_movies['rating_mean'] +
            count_weight * (global_mean + (valid_movies['rating_count'] / valid_movies['rating_count'].max()) * 0.5)
        )
        
        # Normalize score
        if valid_movies['popularity_score'].max() > 0:
            valid_movies['popularity_score'] /= valid_movies['popularity_score'].max()
        
        # Store results
        self.popularity_scores = valid_movies['popularity_score'].sort_values(ascending=False)
        self.movie_ids = valid_movies.index.tolist()
        
        logger.info(f"Popularity model fitted with {len(self.movie_ids)} movies")
        
    def recommend(self, user_id: Optional[int] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get popular movie recommendations
        
        Args:
            user_id: User ID (optional, not used in popularity model)
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.popularity_scores is None:
            raise ValueError("Model not fitted yet")
        
        # Get top movies by popularity
        top_movies = self.popularity_scores.head(top_k)
        
        recommendations = [
            (int(movie_id), float(score))
            for movie_id, score in top_movies.items()
        ]
        
        return recommendations
    
    def get_movie_info(self, movie_id: int, movies_df: pd.DataFrame) -> Dict:
        """
        Get movie information for a given movie ID
        
        Args:
            movie_id: Movie ID
            movies_df: Movies dataframe
            
        Returns:
            Dictionary with movie information
        """
        movie_data = movies_df[movies_df['movieId'] == movie_id]
        if len(movie_data) == 0:
            return {}
        
        movie = movie_data.iloc[0]
        return {
            'movie_id': movie['movieId'],
            'title': movie['title'],
            'genres': movie['genres'],
            'popularity_score': self.popularity_scores.get(movie_id, 0.0)
        }