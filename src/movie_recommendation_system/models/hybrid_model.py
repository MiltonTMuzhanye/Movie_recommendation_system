from typing import Dict, List, Tuple, Optional
import numpy as np
from src.movie_recommendation.utils.logger import logger

class HybridModel:
    """
    Hybrid recommendation model combining multiple approaches
    """
    
    def __init__(self, config: Dict):
        """
        Initialize hybrid model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model, weight: float):
        """
        Add a model to the hybrid ensemble
        
        Args:
            name: Model name
            model: Model instance
            weight: Model weight
        """
        self.models[name] = model
        self.weights[name] = weight
        
    def recommend_for_user(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Get recommendations from each model
        all_recommendations = {}
        
        for name, model in self.models.items():
            try:
                recommendations = model.recommend_for_user(user_id, top_k=top_k*2)
                weight = self.weights.get(name, 1.0)
                
                for movie_id, score in recommendations:
                    if movie_id not in all_recommendations:
                        all_recommendations[movie_id] = 0
                    all_recommendations[movie_id] += weight * score
            except Exception as e:
                logger.warning(f"Error getting recommendations from {name}: {str(e)}")
        
        # Sort by combined score
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_recommendations[:top_k]
    
    def recommend_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar movies using hybrid approach
        
        Args:
            movie_id: Movie ID
            top_k: Number of similar movies
            
        Returns:
            List of (movie_id, score) tuples
        """
        all_similar = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'recommend_similar_movies'):
                    similar = model.recommend_similar_movies(movie_id, top_k=top_k*2)
                    weight = self.weights.get(name, 1.0)
                    
                    for sim_movie_id, score in similar:
                        if sim_movie_id not in all_similar:
                            all_similar[sim_movie_id] = 0
                        all_similar[sim_movie_id] += weight * score
            except Exception as e:
                logger.warning(f"Error getting similar movies from {name}: {str(e)}")
        
        # Sort by combined score
        sorted_similar = sorted(
            all_similar.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_similar[:top_k]