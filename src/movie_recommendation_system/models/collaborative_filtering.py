import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger

class CollaborativeFilteringModel:
    """
    Collaborative filtering model using implicit ALS
    """
    
    def __init__(self, config: Dict):
        """
        Initialize collaborative filtering model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.user_ids = None
        self.movie_ids = None
        self.user_mapping = {}
        self.movie_mapping = {}
        
    def fit(self, ratings_df: pd.DataFrame):
        """
        Fit the collaborative filtering model
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Fitting collaborative filtering model")
        
        # Create user and movie mappings
        self.user_ids = sorted(ratings_df['userId'].unique())
        self.movie_ids = sorted(ratings_df['movieId'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_ids)}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(self.movie_ids)}
        
        # Create user-item matrix
        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        
        # Convert ratings to implicit feedback
        # High rating = positive feedback
        user_indices = [self.user_mapping[user] for user in ratings_df['userId']]
        movie_indices = [self.movie_mapping[movie] for movie in ratings_df['movieId']]
        
        # Create confidence matrix (rating - 2.5)
        confidence = ratings_df['rating'].values - 2.5
        confidence = np.maximum(0, confidence)  # Only positive confidence
        
        # Create sparse matrix
        matrix = csr_matrix((confidence, (user_indices, movie_indices)), 
                           shape=(n_users, n_movies))
        
        # Initialize and train ALS model
        factors = self.config.get('models', {})
            .get('collaborative_filtering', {})
            .get('als', {})
            .get('factors', 50)
        
        regularization = self.config.get('models', {})
            .get('collaborative_filtering', {})
            .get('als', {})
            .get('regularization', 0.01)
        
        iterations = self.config.get('models', {})
            .get('collaborative_filtering', {})
            .get('als', {})
            .get('iterations', 15)
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
        self.model.fit(matrix)
        
        logger.info(f"CF model fitted with {n_users} users and {n_movies} movies")
        
    def recommend_for_user(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get recommendations
        movie_indices, scores = self.model.recommend(
            user_idx,
            None,  # No filtering
            N=top_k,
            filter_already_liked_items=True
        )
        
        # Convert to movie IDs
        recommendations = []
        for idx, score in zip(movie_indices, scores):
            movie_id = self.movie_ids[idx]
            recommendations.append((movie_id, float(score)))
        
        return recommendations
    
    def get_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar movies based on item factors
        
        Args:
            movie_id: Movie ID
            top_k: Number of similar movies
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if movie_id not in self.movie_mapping:
            logger.warning(f"Movie {movie_id} not found in model")
            return []
        
        movie_idx = self.movie_mapping[movie_id]
        
        # Get similar items
        movie_indices, scores = self.model.similar_items(
            movie_idx,
            N=top_k + 1  # Include self
        )
        
        # Convert to movie IDs (skip self)
        recommendations = []
        for idx, score in zip(movie_indices, scores):
            if idx == movie_idx:
                continue
            
            sim_movie_id = self.movie_ids[idx]
            recommendations.append((sim_movie_id, float(score)))
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations