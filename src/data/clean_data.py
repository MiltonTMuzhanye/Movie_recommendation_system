import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self):
        pass
    
    def preprocess_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess movies data"""
        movies = movies_df.copy()
        
        # Extract year from title
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        
        # Clean genres - convert to list
        movies['genres'] = movies['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) else []
        )
        
        # Create one-hot encoded genres
        genre_expanded = movies.explode('genres')
        genre_dummies = pd.get_dummies(genre_expanded['genres']).groupby(genre_expanded['movieId']).max()
        movies = pd.merge(movies, genre_dummies, left_on='movieId', right_index=True)
        
        return movies
    
    def preprocess_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess ratings data"""
        ratings = ratings_df.copy()
        
        # Remove any potential duplicates
        ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
        
        # Ensure proper data types
        ratings['rating'] = ratings['rating'].astype(float)
        
        return ratings
    
    def create_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item rating matrix"""
        user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        return user_item_matrix