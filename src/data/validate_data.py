import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self):
        pass
    
    def validate_movies(self, movies_df: pd.DataFrame) -> bool:
        """Validate movies dataset"""
        checks = []
        
        # Check for required columns
        required_cols = ['movieId', 'title', 'genres']
        checks.append(all(col in movies_df.columns for col in required_cols))
        
        # Check for nulls in required columns
        checks.append(not movies_df[required_cols].isnull().any().any())
        
        # Check for duplicate movieIds
        checks.append(not movies_df['movieId'].duplicated().any())
        
        return all(checks)
    
    def validate_ratings(self, ratings_df: pd.DataFrame) -> bool:
        """Validate ratings dataset"""
        checks = []
        
        # Check for required columns
        required_cols = ['userId', 'movieId', 'rating']
        checks.append(all(col in ratings_df.columns for col in required_cols))
        
        # Check rating range (0.5-5)
        checks.append(ratings_df['rating'].between(0.5, 5).all())
        
        # Check data types
        checks.append(ratings_df['rating'].dtype in [float, np.float64])
        
        return all(checks)