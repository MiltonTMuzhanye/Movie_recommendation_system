import sys
import os
from pathlib import Path
import mlflow
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.config import config
from src.movie_recommendation.pipelines.training_pipeline import TrainingPipeline
from src.movie_recommendation.training.hyperparameter_tuning import HyperparameterTuner

def main():
    """Main training function"""
    logger.info("Starting model training")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(config)
    
    try:
        # Run training
        results = pipeline.run()
        
        logger.info("Training completed successfully")
        logger.info(f"Models trained: {list(results['models'].keys())}")
        logger.info(f"Run ID: {results['run_id']}")
        
        # Print evaluation results
        for name, metrics in results['evaluation'].items():
            logger.info(f"Model {name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()