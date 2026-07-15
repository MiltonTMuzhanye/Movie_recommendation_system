import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import save_pickle, load_pickle

class FeatureEngineer:
    """
    Feature engineering module for movie recommendation system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vectorizers = {}
        self.feature_matrices = {}
        
    def create_genre_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create genre-based features
        
        Args:
            movies_df: Movies dataframe
            
        Returns:
            DataFrame with genre features
        """
        logger.info("Creating genre features")
        
        # Get all genres
        all_genres = set()
        for genres in movies_df['genres_list']:
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))
        
        # Create binary genre features
        genre_features = pd.DataFrame(index=movies_df.index)
        for genre in all_genres:
            genre_features[f'genre_{genre}'] = movies_df['genres_list'].apply(
                lambda x: 1 if genre in x else 0
            )
        
        # Add genre count
        genre_features['genre_count'] = movies_df['genres_list'].apply(len)
        
        # Add genre diversity score
        genre_features['genre_diversity'] = genre_features['genre_count'] / len(all_genres)
        
        logger.info(f"Created {len(genre_features.columns)} genre features")
        return genre_features
    
    def create_text_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Create TF-IDF features from movie titles
        
        Args:
            movies_df: Movies dataframe
            
        Returns:
            TF-IDF feature matrix
        """
        logger.info("Creating text features")
        
        # Use clean title for text features
        texts = movies_df['clean_title'].fillna('').values
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(
            max_features=self.config.get('models', {})
                .get('content_based', {})
                .get('tfidf', {})
                .get('max_features', 5000),
            ngram_range=self.config.get('models', {})
                .get('content_based', {})
                .get('tfidf', {})
                .get('ngram_range', (1, 2))
        )
        
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # Save vectorizer
        self.vectorizers['tfidf'] = tfidf
        
        logger.info(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def create_movie_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive movie features
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
            
        Returns:
            DataFrame with movie features
        """
        logger.info("Creating movie features")
        
        # Start with basic features
        features = pd.DataFrame(index=movies_df['movieId'])
        
        # Genre features
        genre_features = self.create_genre_features(movies_df)
        for col in genre_features.columns:
            features[col] = genre_features[col].values
        
        # Year features
        features['year'] = movies_df['year'].fillna(0)
        features['year_normalized'] = (features['year'] - features['year'].mean()) / features['year'].std()
        
        # Title length features
        features['title_length'] = movies_df['title'].str.len()
        features['title_words'] = movies_df['title'].str.split().str.len()
        
        # Aggregated rating features
        rating_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        rating_stats.columns = ['rating_mean', 'rating_std', 'rating_count', 'rating_first', 'rating_last']
        rating_stats = rating_stats.fillna(0)
        
        # Merge with features
        features = features.join(rating_stats, how='left')
        
        # Fill missing values
        features = features.fillna({
            'rating_mean': 0,
            'rating_std': 0,
            'rating_count': 0
        })
        
        # Calculate popularity score
        features['popularity'] = features['rating_count'] * features['rating_mean']
        
        # Normalize features
        for col in ['rating_mean', 'rating_count', 'popularity']:
            if features[col].max() > 0:
                features[f'{col}_normalized'] = features[col] / features[col].max()
        
        logger.info(f"Created {len(features.columns)} movie features")
        return features
    
    def create_user_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user features from ratings history
        
        Args:
            ratings_df: Ratings dataframe
            
        Returns:
            DataFrame with user features
        """
        logger.info("Creating user features")
        
        # Basic user statistics
        user_stats = ratings_df.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],
            'movieId': 'nunique'
        }).round(2)
        
        user_stats.columns = ['user_rating_mean', 'user_rating_std', 'user_rating_count', 'user_movies_watched']
        
        # Fill missing values
        user_stats = user_stats.fillna({
            'user_rating_mean': 0,
            'user_rating_std': 0,
            'user_rating_count': 0,
            'user_movies_watched': 0
        })
        
        # User activity features
        user_stats['user_activity_score'] = (
            user_stats['user_rating_count'] * 
            user_stats['user_movies_watched']
        )
        
        # Normalize user features
        for col in ['user_rating_count', 'user_movies_watched', 'user_activity_score']:
            if user_stats[col].max() > 0:
                user_stats[f'{col}_normalized'] = user_stats[col] / user_stats[col].max()
        
        logger.info(f"Created {len(user_stats.columns)} user features")
        return user_stats
    
    def save_features(self, features: Dict[str, np.ndarray], path: str):
        """
        Save engineered features
        
        Args:
            features: Dictionary of feature matrices
            path: Path to save features
        """
        for name, matrix in features.items():
            save_pickle(matrix, f"{path}/{name}_features.pkl")
        logger.info(f"Saved features to {path}")