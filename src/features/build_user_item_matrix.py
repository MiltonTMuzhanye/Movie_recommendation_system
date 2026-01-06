import pandas as pd
import numpy as np
import os

class UserItemMatrixBuilder:
    def __init__(self):
        self.matrix = None
        
    def build_from_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Build user-item matrix from ratings"""
        self.matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        return self.matrix
    
    def get_sparsity(self) -> float:
        """Calculate sparsity of the matrix"""
        if self.matrix is None:
            return 0.0
        return 1 - (np.count_nonzero(self.matrix.values) / self.matrix.size)
    
    def save(self, filepath: str):
        """Save matrix to parquet"""
        if self.matrix is not None:
            self.matrix.to_parquet(filepath)
    
    def load(self, filepath: str):
        """Load matrix from parquet"""
        self.matrix = pd.read_parquet(filepath)
        return self.matrix