from typing import Dict, Any, Optional
from src.movie_recommendation.data.ingestion import DataIngestion
from src.movie_recommendation.data.preprocessing import DataPreprocessor
from src.movie_recommendation.training.trainer import ModelTrainer
from src.movie_recommendation.evaluation.ranking_metrics import RankingMetrics
from src.movie_recommendation.utils.logger import logger
import mlflow

class TrainingPipeline:
    """
    Complete training pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_ingestion = DataIngestion(config)
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = RankingMetrics()
        
    def run(self, data_path: Optional[str] = None, 
            experiment_name: str = "movie_recommendation"):
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to dataset
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary of results
        """
        logger.info("Starting training pipeline")
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="training_pipeline") as run:
            # 1. Load data
            movies_df, ratings_df = self.data_ingestion.load_movielens(data_path)
            
            # 2. Preprocess data
            movies_df = self.preprocessor.preprocess_movies(movies_df)
            ratings_df = self.preprocessor.preprocess_ratings(ratings_df)
            
            # 3. Train models
            models, results = self.trainer.train_all_models(data_path)
            
            # 4. Evaluate models
            evaluation_results = {}
            for name, model in models.items():
                score = self.evaluator.evaluate_model(model, ratings_df)
                evaluation_results[name] = score
                
                # Log metrics to MLflow
                for metric_name, metric_value in score.items():
                    mlflow.log_metric(f"{name}_{metric_name}", metric_value)
            
            # 5. Log parameters and artifacts
            mlflow.log_params({
                "dataset_size": len(ratings_df),
                "num_movies": len(movies_df),
                "num_users": ratings_df['userId'].nunique()
            })
            
            # Save model to MLflow
            # This would need model serialization
            
            logger.info("Training pipeline completed")
            
            return {
                'models': models,
                'results': results,
                'evaluation': evaluation_results,
                'run_id': run.info.run_id
            }