import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger

class MatrixFactorizationModel:
    """
    Matrix factorization recommendation model using SVD
    """
    
    def __init__(self, config: Dict):
        """
        Initialize matrix factorization model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.user_ids = None
        self.movie_ids = None
        self.user_mapping = {}
        self.movie_mapping = {}
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = 0
        
    def fit(self, ratings_df: pd.DataFrame):
        """
        Fit the matrix factorization model
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Fitting matrix factorization model")
        
        # Create user and movie mappings
        self.user_ids = sorted(ratings_df['userId'].unique())
        self.movie_ids = sorted(ratings_df['movieId'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_ids)}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(self.movie_ids)}
        
        # Create user-item matrix
        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        
        # Create sparse matrix
        from scipy.sparse import csr_matrix
        
        user_indices = [self.user_mapping[user] for user in ratings_df['userId']]
        movie_indices = [self.movie_mapping[movie] for movie in ratings_df['movieId']]
        ratings = ratings_df['rating'].values
        
        matrix = csr_matrix((ratings, (user_indices, movie_indices)), 
                           shape=(n_users, n_movies))
        
        # Get global mean
        self.global_mean = ratings.mean()
        
        # Normalize ratings
        matrix = matrix - self.global_mean
        
        # Apply SVD
        n_factors = self.config.get('models', {})
            .get('matrix_factorization', {})
            .get('svd', {})
            .get('n_factors', 50)
        
        self.model = TruncatedSVD(
            n_components=n_factors,
            random_state=42,
            n_iter=20
        )
        
        # Fit SVD
        self.user_factors = self.model.fit_transform(matrix)
        self.movie_factors = self.model.components_.T
        
        logger.info(f"MF model fitted with {n_users} users, {n_movies} movies, "
                   f"and {n_factors} factors")
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_mapping or movie_id not in self.movie_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        movie_idx = self.movie_mapping[movie_id]
        
        # Calculate prediction
        prediction = self.global_mean + np.dot(
            self.user_factors[user_idx],
            self.movie_factors[movie_idx]
        )
        
        # Clip to valid range
        prediction = np.clip(prediction, 0.5, 5.0)
        
        return float(prediction)
    
    def recommend_for_user(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get predictions for all movies
        predictions = []
        for movie_idx, movie_id in enumerate(self.movie_ids):
            if user_idx < len(self.user_factors) and movie_idx < len(self.movie_factors):
                pred = self.global_mean + np.dot(
                    self.user_factors[user_idx],
                    self.movie_factors[movie_idx]
                )
                pred = np.clip(pred, 0.5, 5.0)
                predictions.append((movie_id, float(pred)))
        
        # Sort by prediction and return top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]