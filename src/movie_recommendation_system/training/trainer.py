import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import mlflow
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.helpers import save_pickle, ensure_dir
from src.movie_recommendation.data.ingestion import DataIngestion
from src.movie_recommendation.data.preprocessing import DataPreprocessor
from src.movie_recommendation.models.popularity_model import PopularityModel
from src.movie_recommendation.models.content_based import ContentBasedModel
from src.movie_recommendation.models.collaborative_filtering import CollaborativeFilteringModel
from src.movie_recommendation.models.matrix_factorization import MatrixFactorizationModel
from src.movie_recommendation.models.neural_recommender import NeuralRecommender
from src.movie_recommendation.models.hybrid_model import HybridModel

class ModelTrainer:
    """
    Model training pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_ingestion = DataIngestion(config)
        self.preprocessor = DataPreprocessor(config)
        self.models = {}
        self.results = {}
        
    def train_all_models(self, data_path: Optional[str] = None):
        """
        Train all models in the pipeline
        
        Args:
            data_path: Path to dataset directory
        """
        logger.info("Starting model training pipeline")
        
        # Load data
        movies_df, ratings_df = self.data_ingestion.load_movielens(data_path)
        
        # Preprocess data
        movies_df = self.preprocessor.preprocess_movies(movies_df)
        ratings_df = self.preprocessor.preprocess_ratings(ratings_df)
        
        # Split data for training
        train_ratings, val_ratings = self._split_data(ratings_df)
        
        # Start MLflow tracking
        mlflow.set_experiment("movie_recommendation")
        
        with mlflow.start_run(run_name="model_training"):
            # Train each model
            self._train_popularity_model(train_ratings, movies_df)
            self._train_content_based_model(train_ratings, movies_df)
            self._train_collaborative_model(train_ratings)
            self._train_matrix_factorization_model(train_ratings)
            self._train_neural_model(train_ratings)
            self._train_hybrid_model(train_ratings, movies_df)
            
            # Save results
            self._save_models()
            
            logger.info("Training pipeline completed")
            
            return self.models, self.results
    
    def _split_data(self, ratings_df: pd.DataFrame) -> tuple:
        """
        Split data into train and validation sets
        
        Args:
            ratings_df: Ratings dataframe
            
        Returns:
            Train and validation dataframes
        """
        from sklearn.model_selection import train_test_split
        
        # Stratified split based on user
        users = ratings_df['userId'].unique()
        train_users, val_users = train_test_split(
            users,
            test_size=0.2,
            random_state=42
        )
        
        train_ratings = ratings_df[ratings_df['userId'].isin(train_users)]
        val_ratings = ratings_df[ratings_df['userId'].isin(val_users)]
        
        logger.info(f"Split data: {len(train_ratings)} train, {len(val_ratings)} validation")
        return train_ratings, val_ratings
    
    def _train_popularity_model(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Train popularity model
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
        """
        logger.info("Training popularity model")
        
        model = PopularityModel(self.config)
        model.fit(ratings_df, movies_df)
        
        self.models['popularity'] = model
        self.results['popularity'] = {'status': 'success'}
        
        mlflow.log_param("popularity_model", "trained")
    
    def _train_content_based_model(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Train content-based model
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
        """
        logger.info("Training content-based model")
        
        model = ContentBasedModel(self.config)
        model.fit(movies_df, ratings_df)
        
        self.models['content_based'] = model
        self.results['content_based'] = {'status': 'success'}
        
        mlflow.log_param("content_based_model", "trained")
    
    def _train_collaborative_model(self, ratings_df: pd.DataFrame):
        """
        Train collaborative filtering model
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Training collaborative filtering model")
        
        model = CollaborativeFilteringModel(self.config)
        model.fit(ratings_df)
        
        self.models['collaborative'] = model
        self.results['collaborative'] = {'status': 'success'}
        
        mlflow.log_param("collaborative_model", "trained")
    
    def _train_matrix_factorization_model(self, ratings_df: pd.DataFrame):
        """
        Train matrix factorization model
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Training matrix factorization model")
        
        model = MatrixFactorizationModel(self.config)
        model.fit(ratings_df)
        
        self.models['mf'] = model
        self.results['mf'] = {'status': 'success'}
        
        mlflow.log_param("mf_model", "trained")
    
    def _train_neural_model(self, ratings_df: pd.DataFrame):
        """
        Train neural recommender
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Training neural recommender")
        
        model = NeuralRecommender(self.config)
        model.fit(ratings_df)
        
        self.models['neural'] = model
        self.results['neural'] = {'status': 'success'}
        
        mlflow.log_param("neural_model", "trained")
    
    def _train_hybrid_model(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Train hybrid model combining all models
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
        """
        logger.info("Training hybrid model")
        
        hybrid_model = HybridModel(self.config)
        
        # Add all trained models with weights
        weights = self.config.get('models', {}).get('hybrid', {}).get('weights', {})
        
        for name, model in self.models.items():
            if name != 'hybrid':
                weight = weights.get(name, 1.0 / len(self.models))
                hybrid_model.add_model(name, model, weight)
        
        self.models['hybrid'] = hybrid_model
        self.results['hybrid'] = {'status': 'success'}
        
        mlflow.log_param("hybrid_model", "trained")
    
    def _save_models(self):
        """
        Save all trained models to artifacts directory
        """
        logger.info("Saving trained models")
        
        artifacts_path = Path("artifacts/trained_models")
        ensure_dir(str(artifacts_path))
        
        for name, model in self.models.items():
            save_pickle(model, str(artifacts_path / f"{name}_model.pkl"))
            logger.info(f"Saved {name} model")
        
        # Log models with MLflow
        for name, model in self.models.items():
            mlflow.log_artifact(
                str(artifacts_path / f"{name}_model.pkl"),
                artifact_path="models"
            )