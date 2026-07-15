import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union
from sklearn.decomposition import PCA
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import save_pickle, load_pickle

class EmbeddingGenerator:
    """
    Embedding generation module for movie recommendation system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize embedding generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.embeddings_cache = {}
        
    def initialize_model(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        if model_name is None:
            model_name = self.config.get('models', {})
                .get('content_based', {})
                .get('text_embedding', {})
                .get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def generate_movie_embeddings(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for movie titles
        
        Args:
            movies_df: Movies dataframe
            
        Returns:
            Embeddings matrix
        """
        logger.info("Generating movie embeddings")
        
        # Prepare text for embedding
        texts = []
        for idx, row in movies_df.iterrows():
            # Combine title and genres for richer representation
            text = f"{row['clean_title']} {' '.join(row['genres_list'])}"
            texts.append(text)
        
        # Generate embeddings
        if self.model is None:
            self.initialize_model()
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Reduce dimensionality if needed
        if embeddings.shape[1] > 128:
            pca = PCA(n_components=128)
            embeddings = pca.fit_transform(embeddings)
            self.pca_model = pca
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def generate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Generate cosine similarity matrix from embeddings
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Cosine similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("Generating similarity matrix")
        similarity = cosine_similarity(embeddings)
        
        logger.info(f"Generated similarity matrix with shape {similarity.shape}")
        return similarity
    
    def get_similar_movies(self, movie_id: int, similarity_matrix: np.ndarray, 
                          movie_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar movies using embedding similarity
        
        Args:
            movie_id: Query movie ID
            similarity_matrix: Precomputed similarity matrix
            movie_ids: List of movie IDs corresponding to matrix indices
            top_k: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        # Find index of movie
        try:
            idx = movie_ids.index(movie_id)
        except ValueError:
            logger.error(f"Movie ID {movie_id} not found")
            return []
        
        # Get similarity scores
        scores = similarity_matrix[idx]
        
        # Get top k similar movies (excluding self)
        similar_indices = np.argsort(scores)[::-1][1:top_k+1]
        
        results = []
        for sim_idx in similar_indices:
            sim_movie_id = movie_ids[sim_idx]
            sim_score = scores[sim_idx]
            results.append((sim_movie_id, float(sim_score)))
        
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """
        Save generated embeddings
        
        Args:
            embeddings: Embedding matrix
            path: Path to save embeddings
        """
        save_pickle(embeddings, f"{path}/movie_embeddings.pkl")
        logger.info(f"Saved embeddings to {path}")