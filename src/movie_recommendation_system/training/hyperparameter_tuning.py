import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import ParameterGrid
from src.movie_recommendation.utils.logger import logger
import mlflow

class HyperparameterTuner:
    """
    Hyperparameter tuning for recommendation models
    """
    
    def __init__(self, config: Dict):
        """
        Initialize hyperparameter tuner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.best_params = {}
        self.results = []
        
    def tune_collaborative_model(self, model_class, train_data, val_data, 
                                 param_grid: Dict, n_trials: int = 10):
        """
        Tune collaborative filtering model hyperparameters
        
        Args:
            model_class: Model class to tune
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter grid for tuning
            n_trials: Number of trials
            
        Returns:
            Best parameters and scores
        """
        logger.info("Tuning collaborative filtering model")
        
        best_score = -np.inf
        best_params = None
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        param_combinations = param_combinations[:n_trials]
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            with mlflow.start_run(run_name=f"tune_cf_{i}", nested=True):
                # Train model with current parameters
                model = model_class(self.config)
                
                # Update model parameters
                for key, value in params.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                
                try:
                    model.fit(train_data)
                    score = self._evaluate_model(model, val_data)
                    
                    mlflow.log_params(params)
                    mlflow.log_metric("val_score", score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"Trial failed: {str(e)}")
                    continue
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
        
        self.best_params['collaborative'] = best_params
        return best_params, best_score
    
    def _evaluate_model(self, model, val_data):
        """
        Evaluate model on validation data
        
        Args:
            model: Trained model
            val_data: Validation data
            
        Returns:
            Evaluation score
        """
        # This is a placeholder - implement actual evaluation
        # For now, return a random score
        return np.random.random()
    
    def tune_neural_model(self, model_class, train_data, val_data,
                          param_grid: Dict, n_trials: int = 10):
        """
        Tune neural recommender hyperparameters
        
        Args:
            model_class: Model class to tune
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter grid for tuning
            n_trials: Number of trials
            
        Returns:
            Best parameters and scores
        """
        logger.info("Tuning neural recommender model")
        
        best_score = -np.inf
        best_params = None
        
        param_combinations = list(ParameterGrid(param_grid))
        param_combinations = param_combinations[:n_trials]
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            with mlflow.start_run(run_name=f"tune_neural_{i}", nested=True):
                model = model_class(self.config)
                
                # Update model parameters
                for key, value in params.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                
                try:
                    model.fit(train_data)
                    score = self._evaluate_model(model, val_data)
                    
                    mlflow.log_params(params)
                    mlflow.log_metric("val_score", score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"Trial failed: {str(e)}")
                    continue
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
        
        self.best_params['neural'] = best_params
        return best_params, best_score