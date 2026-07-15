import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.features.feature_engineering import FeatureEngineer
from src.movie_recommendation.features.embeddings import EmbeddingGenerator

class ContentBasedModel:
    """
    Content-based recommendation model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize content-based model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        self.similarity_matrix = None
        self.movie_ids = None
        self.movie_features = None
        
    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Fit the content-based model
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
        """
        logger.info("Fitting content-based model")
        
        # Get movie IDs
        self.movie_ids = movies_df['movieId'].tolist()
        
        # Create features
        self.movie_features = self.feature_engineer.create_movie_features(movies_df, ratings_df)
        
        # Calculate similarity
        # Use cosine similarity on combined features
        feature_columns = [col for col in self.movie_features.columns 
                          if col not in ['year', 'rating_first', 'rating_last']]
        
        feature_matrix = self.movie_features[feature_columns].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        # Calculate similarity
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        logger.info(f"Content-based model fitted with similarity matrix shape {self.similarity_matrix.shape}")
        
    def recommend_for_user(self, user_id: int, ratings_df: pd.DataFrame, 
                           top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            ratings_df: Ratings dataframe
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Get user's ratings
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            logger.warning(f"No ratings found for user {user_id}")
            return []
        
        # Get movies user hasn't rated
        rated_movies = set(user_ratings['movieId'])
        unrated_movies = [m for m in self.movie_ids if m not in rated_movies]
        
        if not unrated_movies:
            logger.warning(f"User {user_id} has rated all movies")
            return []
        
        # Calculate scores for unrated movies
        scores = []
        
        for movie_id in unrated_movies:
            score = self._calculate_user_movie_score(user_id, movie_id, user_ratings)
            scores.append((movie_id, score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def recommend_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar movies for a given movie
        
        Args:
            movie_id: Movie ID
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted yet")
        
        try:
            idx = self.movie_ids.index(movie_id)
        except ValueError:
            logger.error(f"Movie {movie_id} not found")
            return []
        
        # Get similarity scores
        scores = self.similarity_matrix[idx]
        
        # Get top k similar movies (excluding self)
        similar_indices = np.argsort(scores)[::-1][1:top_k+1]
        
        recommendations = []
        for sim_idx in similar_indices:
            sim_movie_id = self.movie_ids[sim_idx]
            sim_score = scores[sim_idx]
            recommendations.append((sim_movie_id, float(sim_score)))
        
        return recommendations
    
    def _calculate_user_movie_score(self, user_id: int, movie_id: int, 
                                   user_ratings: pd.DataFrame) -> float:
        """
        Calculate user-movie score based on content similarity
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            user_ratings: User's ratings
            
        Returns:
            Score for user-movie pair
        """
        try:
            movie_idx = self.movie_ids.index(movie_id)
        except ValueError:
            return 0.0
        
        # Calculate weighted average of similarity to rated movies
        total_weight = 0
        weighted_sum = 0
        
        for _, rating in user_ratings.iterrows():
            rated_movie_id = rating['movieId']
            rating_value = rating['rating']
            
            try:
                rated_idx = self.movie_ids.index(rated_movie_id)
                similarity = self.similarity_matrix[rated_idx][movie_idx]
                
                if similarity > 0:
                    weight = similarity * (rating_value - 2.5) / 2.5
                    weighted_sum += weight * similarity
                    total_weight += similarity
            except ValueError:
                continue
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight