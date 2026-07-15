import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.movie_recommendation.utils.logger import logger

class DataValidator:
    """
    Data validation module for movie recommendation system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data validator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def validate_movies(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate movies dataframe
        
        Args:
            df: Movies dataframe
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['movieId', 'title', 'genres']
        for col in required_cols:
            if col not in df.columns:
                results['valid'] = False
                results['issues'].append(f"Missing column: {col}")
        
        if not results['valid']:
            return results['valid'], results
        
        # Check for duplicates
        duplicate_movie_ids = df['movieId'].duplicated().sum()
        if duplicate_movie_ids > 0:
            results['issues'].append(f"Found {duplicate_movie_ids} duplicate movie IDs")
            results['valid'] = False
        
        # Check for missing values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                results['issues'].append(f"Column '{col}' has {count} null values")
        
        # Statistics
        results['stats'] = {
            'total_movies': len(df),
            'unique_genres': self._count_unique_genres(df),
            'year_range': self._extract_year_range(df)
        }
        
        return results['valid'], results
    
    def validate_ratings(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate ratings dataframe
        
        Args:
            df: Ratings dataframe
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['userId', 'movieId', 'rating', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                results['valid'] = False
                results['issues'].append(f"Missing column: {col}")
        
        if not results['valid']:
            return results['valid'], results
        
        # Check rating range
        invalid_ratings = df[(df['rating'] < 0.5) | (df['rating'] > 5.0)]
        if len(invalid_ratings) > 0:
            results['issues'].append(f"Found {len(invalid_ratings)} invalid ratings")
            results['valid'] = False
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['userId', 'movieId']).sum()
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate ratings")
        
        # Statistics
        results['stats'] = {
            'total_ratings': len(df),
            'unique_users': df['userId'].nunique(),
            'unique_movies': df['movieId'].nunique(),
            'rating_mean': df['rating'].mean(),
            'rating_std': df['rating'].std(),
            'sparsity': self._calculate_sparsity(df)
        }
        
        return results['valid'], results
    
    def _count_unique_genres(self, df: pd.DataFrame) -> int:
        """Count unique genres in movies dataframe"""
        all_genres = set()
        for genres in df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.update(genres)
        return len(all_genres)
    
    def _extract_year_range(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Extract min and max years from movie titles"""
        years = []
        for title in df['title']:
            try:
                if '(' in title and ')' in title:
                    year = title[title.rfind('(')+1:title.rfind(')')]
                    if year.isdigit():
                        years.append(int(year))
            except:
                continue
        
        if years:
            return min(years), max(years)
        return None, None
    
    def _calculate_sparsity(self, df: pd.DataFrame) -> float:
        """Calculate sparsity of ratings matrix"""
        n_users = df['userId'].nunique()
        n_movies = df['movieId'].nunique()
        n_ratings = len(df)
        
        total_possible = n_users * n_movies
        if total_possible > 0:
            return 1 - (n_ratings / total_possible)
        return 1.0