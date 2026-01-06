import pandas as pd
import numpy as np
from typing import List, Dict

class HybridFeatureBuilder:
    def __init__(self):
        pass
    
    def create_hybrid_features(
        self, 
        content_scores: np.ndarray,
        collab_scores: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """Create hybrid features by combining content and collaborative scores"""
        if weights is None:
            weights = {'content': 0.5, 'collab': 0.5}
        
        # Normalize scores
        content_norm = self._normalize_scores(content_scores)
        collab_norm = self._normalize_scores(collab_scores)
        
        # Combine with weights
        hybrid_scores = (
            weights['content'] * content_norm +
            weights['collab'] * collab_norm
        )
        
        return hybrid_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)