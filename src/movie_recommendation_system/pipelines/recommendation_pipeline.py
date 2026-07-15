import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import load_pickle
import redis
import json

class RecommendationPipeline:
    """
    Real-time recommendation pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Initialize recommendation pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.movies_df = None
        self.ratings_df = None
        
        # Redis cache
        self.cache_enabled = config.get('api', {}).get('cache', {}).get('enabled', True)
        if self.cache_enabled:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
    
    def load_models(self, model_path: str = "artifacts/trained_models/"):
        """
        Load trained models
        
        Args:
            model_path: Path to model directory
        """
        model_files = [
            ('popularity', 'popularity_model.pkl'),
            ('content_based', 'content_based_model.pkl'),
            ('collaborative', 'collaborative_model.pkl'),
            ('mf', 'mf_model.pkl'),
            ('neural', 'neural_model.pkl'),
            ('hybrid', 'hybrid_model.pkl')
        ]
        
        for name, file in model_files:
            try:
                self.models[name] = load_pickle(f"{model_path}/{file}")
                logger.info(f"Loaded {name} model")
            except FileNotFoundError:
                logger.warning(f"Model {name} not found")
    
    def load_data(self, data_path: str = "data/processed/"):
        """
        Load processed data
        
        Args:
            data_path: Path to data directory
        """
        try:
            self.movies_df = pd.read_csv(f"{data_path}/movies_processed.csv")
            self.ratings_df = pd.read_csv(f"{data_path}/ratings_processed.csv")
            logger.info(f"Loaded movies data: {len(self.movies_df)} movies")
        except FileNotFoundError:
            logger.warning("Processed data not found")
    
    def get_recommendations(self, user_id: int, model_type: str = 'hybrid',
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            model_type: Type of model to use
            top_k: Number of recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        # Check cache first
        cache_key = f"recommendations:{user_id}:{model_type}:{top_k}"
        if self.cache_enabled:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for user {user_id}")
                return json.loads(cached_result)
        
        # Get recommendations from model
        model = self.models.get(model_type)
        if model is None:
            logger.error(f"Model {model_type} not found")
            return []
        
        try:
            recommendations = model.recommend_for_user(user_id, top_k=top_k)
            
            # Format results
            results = []
            for movie_id, score in recommendations:
                movie_info = self._get_movie_info(movie_id)
                if movie_info:
                    results.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'score': score,
                        'year': movie_info.get('year', 'Unknown')
                    })
            
            # Cache results
            if self.cache_enabled:
                self.redis_client.setex(
                    cache_key,
                    3600,  # TTL: 1 hour
                    json.dumps(results)
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def get_similar_movies(self, movie_id: int, model_type: str = 'content_based',
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get similar movies
        
        Args:
            movie_id: Query movie ID
            model_type: Type of model to use
            top_k: Number of similar movies
            
        Returns:
            List of similar movie dictionaries
        """
        model = self.models.get(model_type)
        if model is None:
            logger.error(f"Model {model_type} not found")
            return []
        
        try:
            if hasattr(model, 'recommend_similar_movies'):
                similar = model.recommend_similar_movies(movie_id, top_k=top_k)
            else:
                # Try alternative method
                similar = model.get_similar_movies(movie_id, top_k=top_k)
            
            results = []
            for sim_id, score in similar:
                movie_info = self._get_movie_info(sim_id)
                if movie_info:
                    results.append({
                        'movie_id': sim_id,
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'similarity_score': score,
                        'year': movie_info.get('year', 'Unknown')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar movies: {str(e)}")
            return []
    
    def get_popular_movies(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular movies
        
        Args:
            top_k: Number of movies
            
        Returns:
            List of popular movie dictionaries
        """
        model = self.models.get('popularity')
        if model is None:
            logger.error("Popularity model not found")
            return []
        
        try:
            popular = model.recommend(None, top_k=top_k)
            
            results = []
            for movie_id, score in popular:
                movie_info = self._get_movie_info(movie_id)
                if movie_info:
                    results.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'popularity_score': score,
                        'year': movie_info.get('year', 'Unknown')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting popular movies: {str(e)}")
            return []
    
    def _get_movie_info(self, movie_id: int) -> Dict[str, Any]:
        """
        Get movie information
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Movie information dictionary
        """
        if self.movies_df is None:
            return {}
        
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            return {}
        
        movie_data = movie.iloc[0]
        return {
            'movie_id': movie_data['movieId'],
            'title': movie_data['title'],
            'genres': movie_data['genres'],
            'year': movie_data.get('year', 'Unknown')
        }