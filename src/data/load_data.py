import pandas as pd
import os
from typing import Dict, Tuple

class DataLoader:
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        
    def load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MovieLens dataset"""
        try:
            movies = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
            ratings = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
            print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
            return movies, ratings
        except FileNotFoundError as e:
            raise Exception(f"Data files not found. Please ensure movies.csv and ratings.csv are in {self.data_path}") from e
    
    def get_data_summary(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict:
        """Get summary statistics of the dataset"""
        return {
            "n_movies": len(movies),
            "n_ratings": len(ratings),
            "n_users": ratings['userId'].nunique(),
            "avg_ratings_per_user": len(ratings) / ratings['userId'].nunique(),
            "avg_ratings_per_movie": len(ratings) / ratings['movieId'].nunique(),
            "sparsity": 1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
        }