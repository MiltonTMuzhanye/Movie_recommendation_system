import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CollaborativeRecommender:
    def __init__(self, model_type: str = 'svd'):
        self.model_type = model_type
        self.model = None
        self.trainset = None
        
    def train(self, ratings_df: pd.DataFrame) -> Dict:
        """Train collaborative filtering model"""
        # Prepare data for Surprise library
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        
        # Split data
        trainset, testset = train_test_split(data, test_size=0.25)
        self.trainset = trainset
        
        # Initialize and train model
        if self.model_type == 'svd':
            self.model = SVD()
        elif self.model_type == 'knn':
            self.model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
        else:
            raise ValueError("model_type must be 'svd' or 'knn'")
        
        self.model.fit(trainset)
        
        # Evaluate on test set
        from surprise import accuracy
        predictions = self.model.test(testset)
        
        return {
            'rmse': accuracy.rmse(predictions),
            'mae': accuracy.mae(predictions)
        }
    
    def recommend_for_user(
        self, 
        user_id: int, 
        movies_df: pd.DataFrame, 
        ratings_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Recommend movies for a specific user"""
        # Get movies the user hasn't rated
        rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
        all_movies = set(ratings_df['movieId'].unique())
        unseen_movies = list(all_movies - rated_movies)
        
        # Predict ratings for unseen movies
        predictions = []
        for movie_id in unseen_movies:
            pred = self.model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred.est
            })
        
        # Sort by predicted rating
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_rating', ascending=False).head(top_n)
        
        # Merge with movie details
        recommendations = pd.merge(
            predictions_df,
            movies_df,
            on='movieId'
        )[['movieId', 'title', 'genres', 'predicted_rating']]
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        import joblib
        joblib.dump(self.model, filepath)