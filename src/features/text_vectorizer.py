import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.title_matrix = None
        
    def fit_transform(self, movie_titles: pd.Series):
        """Fit and transform movie titles"""
        self.title_matrix = self.vectorizer.fit_transform(movie_titles)
        return self.title_matrix
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute cosine similarity matrix for all movies"""
        if self.title_matrix is None:
            raise ValueError("Fit the vectorizer first")
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(self.title_matrix)
        return similarity_matrix
    
    def save(self, filepath: str):
        """Save vectorizer to disk"""
        joblib.dump(self.vectorizer, filepath)
    
    def load(self, filepath: str):
        """Load vectorizer from disk"""
        self.vectorizer = joblib.load(filepath)