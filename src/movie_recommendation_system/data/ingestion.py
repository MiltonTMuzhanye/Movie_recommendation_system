import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.exceptions import DataIngestionError

class DataIngestion:
    """
    Data ingestion module for loading movie recommendation datasets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data ingestion
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['data']['raw_path'])
        self.processed_dir = Path(config['data']['processed_path'])
        
    def load_movielens(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens dataset
        
        Args:
            file_path: Path to dataset directory
            
        Returns:
            Tuple of (movies_df, ratings_df)
        """
        try:
            if file_path is None:
                file_path = self.data_dir
            
            logger.info(f"Loading MovieLens dataset from {file_path}")
            
            # Load movies
            movies_path = Path(file_path) / "movies.csv"
            movies_df = pd.read_csv(movies_path)
            logger.info(f"Loaded {len(movies_df)} movies")
            
            # Load ratings
            ratings_path = Path(file_path) / "ratings.csv"
            ratings_df = pd.read_csv(ratings_path)
            logger.info(f"Loaded {len(ratings_df)} ratings")
            
            # Basic validation
            self._validate_data(movies_df, ratings_df)
            
            return movies_df, ratings_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise DataIngestionError(f"Failed to load data: {str(e)}")
    
    def _validate_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Validate loaded data
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
        """
        # Check required columns
        required_movie_cols = ['movieId', 'title', 'genres']
        required_rating_cols = ['userId', 'movieId', 'rating', 'timestamp']
        
        for col in required_movie_cols:
            if col not in movies_df.columns:
                raise DataIngestionError(f"Missing required column '{col}' in movies data")
        
        for col in required_rating_cols:
            if col not in ratings_df.columns:
                raise DataIngestionError(f"Missing required column '{col}' in ratings data")
        
        # Check rating range
        if ratings_df['rating'].min() < 0.5 or ratings_df['rating'].max() > 5.0:
            raise DataIngestionError("Ratings should be in range [0.5, 5.0]")
        
        # Check for missing values
        if movies_df.isnull().any().any():
            logger.warning("Movies data contains null values")
        
        if ratings_df.isnull().any().any():
            logger.warning("Ratings data contains null values")
        
        # Check movie IDs
        movie_ids = set(movies_df['movieId'])
        rating_movie_ids = set(ratings_df['movieId'])
        missing_movies = rating_movie_ids - movie_ids
        
        if missing_movies:
            logger.warning(f"Found {len(missing_movies)} movie IDs in ratings not in movies data"