import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import save_pickle, load_pickle

class SimilarityCalculator:
    """
    Similarity calculation module for movie recommendation system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize similarity calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.similarity_matrices = {}
        
    def calculate_item_similarity(self, item_features: np.ndarray) -> np.ndarray:
        """
        Calculate item-item similarity matrix
        
        Args:
            item_features: Feature matrix (items x features)
            
        Returns:
            Similarity matrix
        """
        logger.info("Calculating item similarity")
        
        # Normalize features
        if np.any(item_features.sum(axis=1) > 0):
            item_features = item_features / item_features.sum(axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(item_features)
        
        # Apply threshold if specified
        threshold = self.config.get('models', {})
            .get('content_based', {})
            .get('cosine_similarity', {})
            .get('threshold', 0.3)
        
        if threshold > 0:
            similarity[similarity < threshold] = 0
        
        logger.info(f"Calculated similarity matrix with shape {similarity.shape}")
        return similarity
    
    def calculate_user_similarity(self, user_item_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate user-user similarity matrix
        
        Args:
            user_item_matrix: User-item matrix (users x items)
            
        Returns:
            User similarity matrix
        """
        logger.info("Calculating user similarity")
        similarity = cosine_similarity(user_item_matrix)
        logger.info(f"Calculated user similarity matrix with shape {similarity.shape}")
        return similarity
    
    def get_top_similar_items(self, item_id: int, similarity_matrix: np.ndarray,
                             item_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top similar items
        
        Args:
            item_id: Query item ID
            similarity_matrix: Precomputed similarity matrix
            item_ids: List of item IDs corresponding to matrix indices
            top_k: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        # Find index of item
        try:
            idx = item_ids.index(item_id)
        except ValueError:
            logger.error(f"Item ID {item_id} not found")
            return []
        
        # Get similarity scores
        scores = similarity_matrix[idx]
        
        # Get top k similar items (excluding self)
        similar_indices = np.argsort(scores)[::-1][1:top_k+1]
        
        results = []
        for sim_idx in similar_indices:
            sim_item_id = item_ids[sim_idx]
            sim_score = scores[sim_idx]
            if sim_score > 0:
                results.append((sim_item_id, float(sim_score)))
        
        return results
    
    def get_top_similar_users(self, user_id: int, similarity_matrix: np.ndarray,
                             user_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top similar users
        
        Args:
            user_id: Query user ID
            similarity_matrix: Precomputed similarity matrix
            user_ids: List of user IDs corresponding to matrix indices
            top_k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        # Find index of user
        try:
            idx = user_ids.index(user_id)
        except ValueError:
            logger.error(f"User ID {user_id} not found")
            return []
        
        # Get similarity scores
        scores = similarity_matrix[idx]
        
        # Get top k similar users (excluding self)
        similar_indices = np.argsort(scores)[::-1][1:top_k+1]
        
        results = []
        for sim_idx in similar_indices:
            sim_user_id = user_ids[sim_idx]
            sim_score = scores[sim_idx]
            results.append((sim_user_id, float(sim_score)))
        
        return results
    
    def save_similarity_matrix(self, similarity: np.ndarray, name: str, path: str):
        """
        Save similarity matrix
        
        Args:
            similarity: Similarity matrix
            name: Name of the matrix
            path: Path to save
        """
        self.similarity_matrices[name] = similarity
        save_pickle(similarity, f"{path}/{name}_similarity.pkl")
        logger.info(f"Saved {name} similarity matrix to {path}")