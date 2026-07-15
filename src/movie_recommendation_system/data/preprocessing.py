import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import save_pickle, ensure_dir

class DataPreprocessor:
    """
    Data preprocessing module for movie recommendation system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.label_encoders = {}
        
    def preprocess_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess movies dataframe
        
        Args:
            df: Movies dataframe
            
        Returns:
            Preprocessed movies dataframe
        """
        logger.info("Preprocessing movies data")
        df = df.copy()
        
        # Extract year from title
        df['year'] = df['title'].apply(self._extract_year)
        
        # Clean title (remove year)
        df['clean_title'] = df['title'].apply(self._clean_title)
        
        # Split genres into list
        df['genres_list'] = df['genres'].str.split('|')
        
        # Create genre matrix
        genre_matrix = self._create_genre_matrix(df)
        df['genre_vector'] = list(genre_matrix.values)
        
        # Basic statistics
        df['num_genres'] = df['genres_list'].apply(len)
        
        logger.info(f"Preprocessed {len(df)} movies")
        return df
    
    def preprocess_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess ratings dataframe
        
        Args:
            df: Ratings dataframe
            
        Returns:
            Preprocessed ratings dataframe
        """
        logger.info("Preprocessing ratings data")
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract time features
        df['rating_year'] = df['timestamp'].dt.year
        df['rating_month'] = df['timestamp'].dt.month
        df['rating_day'] = df['timestamp'].dt.day
        df['rating_hour'] = df['timestamp'].dt.hour
        
        # Normalize ratings (optional)
        if self.config.get('preprocessing', {}).get('ratings', {}).get('normalize', False):
            df['rating_normalized'] = (df['rating'] - df['rating'].mean()) / df['rating'].std()
        
        logger.info(f"Preprocessed {len(df)} ratings")
        return df
    
    def create_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-item matrix from ratings
        
        Args:
            ratings_df: Ratings dataframe
            
        Returns:
            User-item matrix
        """
        logger.info("Creating user-item matrix")
        
        # Pivot table
        matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Ensure we have user and movie IDs encoded
        if 'userId' not in self.label_encoders:
            self.label_encoders['user'] = LabelEncoder()
            self.label_encoders['user'].fit(ratings_df['userId'])
        
        if 'movieId' not in self.label_encoders:
            self.label_encoders['movie'] = LabelEncoder()
            self.label_encoders['movie'].fit(ratings_df['movieId'])
        
        logger.info(f"Created matrix with {matrix.shape[0]} users and {matrix.shape[1]} movies")
        return matrix
    
    def _extract_year(self, title: str) -> Optional[int]:
        """Extract year from movie title"""
        try:
            if '(' in title and ')' in title:
                year_str = title[title.rfind('(')+1:title.rfind(')')]
                if year_str.isdigit() and 1900 <= int(year_str) <= 2024:
                    return int(year_str)
        except:
            pass
        return None
    
    def _clean_title(self, title: str) -> str:
        """Remove year from movie title"""
        if '(' in title and ')' in title:
            return title[:title.rfind('(')].strip()
        return title
    
    def _create_genre_matrix(self, df: pd.DataFrame) -> Dict[int, Dict[str, int]]:
        """Create genre one-hot encoding matrix"""
        # Get all unique genres
        all_genres = set()
        for genres in df['genres_list']:
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))
        
        # Create genre matrix
        genre_matrix = {}
        for idx, row in df.iterrows():
            movie_id = row['movieId']
            genre_vector = {genre: 1 if genre in row['genres_list'] else 0 for genre in all_genres}
            genre_matrix[movie_id] = genre_vector
        
        return genre_matrix
    
    def save_encoders(self, path: str):
        """
        Save label encoders
        
        Args:
            path: Path to save encoders
        """
        ensure_dir(path)
        for name, encoder in self.label_encoders.items():
            save_pickle(encoder, f"{path}/{name}_encoder.pkl")
        logger.info(f"Saved encoders to {path}")
