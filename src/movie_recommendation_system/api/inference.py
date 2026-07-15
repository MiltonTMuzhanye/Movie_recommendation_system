import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import joblib

class RecommendationInference:
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.models_loaded = False
        
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.content_model = joblib.load(f"{self.model_path}/content_model.pkl")
            self.collab_model = joblib.load(f"{self.model_path}/collaborative_model.pkl")
            self.hybrid_model = joblib.load(f"{self.model_path}/hybrid_model.pkl")
            self.movies = pd.read_parquet(f"{self.model_path}/movies_processed.parquet")
            self.metadata = joblib.load(f"{self.model_path}/metadata.json")
            self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def get_content_recommendations(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """Get content-based recommendations"""
        if not self.models_loaded:
            self.load_models()
        
        recommendations = self.content_model.recommend_similar_movies(movie_id, top_n)
        return recommendations.to_dict('records')
    
    def get_collab_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        if not self.models_loaded:
            self.load_models()
        
        # Load ratings for collaborative filtering
        ratings = pd.read_parquet("data/processed/cleaned_interactions.parquet")
        recommendations = self.collab_model.recommend_for_user(
            user_id, self.movies, ratings, top_n
        )
        return recommendations.to_dict('records')
    
    def get_hybrid_recommendations(
        self, 
        user_id: int, 
        reference_movie_id: Optional[int] = None,
        reference_movie_title: Optional[str] = None,
        top_n: int = 10
    ) -> List[Dict]:
        """Get hybrid recommendations"""
        if not self.models_loaded:
            self.load_models()
        
        ratings = pd.read_parquet("data/processed/cleaned_interactions.parquet")
        
        recommendations = self.hybrid_model.recommend(
            user_id=user_id,
            reference_movie_id=reference_movie_id,
            reference_movie_title=reference_movie_title,
            ratings_df=ratings,
            top_n=top_n
        )
        
        return recommendations.to_dict('records')