import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from surprise.model_selection import cross_validate

class ModelEvaluator:
    def __init__(self):
        pass
    
    def evaluate_collaborative_models(
        self, 
        ratings_df: pd.DataFrame,
        models: Dict[str, any]
    ) -> pd.DataFrame:
        """Evaluate multiple collaborative filtering models"""
        results = []
        
        reader = Dataset.Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        
        for model_name, model in models.items():
            cv_results = cross_validate(
                model, 
                data, 
                measures=['RMSE', 'MAE'], 
                cv=3, 
                verbose=False
            )
            
            results.append({
                'model': model_name,
                'rmse_mean': np.mean(cv_results['test_rmse']),
                'rmse_std': np.std(cv_results['test_rmse']),
                'mae_mean': np.mean(cv_results['test_mae']),
                'mae_std': np.std(cv_results['test_mae']),
                'fit_time_mean': np.mean(cv_results['fit_time']),
                'test_time_mean': np.mean(cv_results['test_time'])
            })
        
        return pd.DataFrame(results)
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """Plot model comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE comparison
        axes[0].bar(results_df['model'], results_df['rmse_mean'], 
                   yerr=results_df['rmse_std'], capsize=5)
        axes[0].set_title('RMSE Comparison (lower is better)')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1].bar(results_df['model'], results_df['mae_mean'],
                   yerr=results_df['mae_std'], capsize=5)
        axes[1].set_title('MAE Comparison (lower is better)')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_precision_at_k(self, recommendations: pd.DataFrame, 
                                actual_ratings: pd.DataFrame, 
                                k: int = 10, 
                                threshold: float = 4.0) -> float:
        """Calculate Precision@K"""
        # Get top K recommended movie IDs
        top_k_movies = recommendations['movieId'].head(k).tolist()
        
        # Get actual high-rated movies for the user
        user_id = recommendations['userId'].iloc[0] if 'userId' in recommendations.columns else None
        if user_id:
            actual_high_rated = set(
                actual_ratings[
                    (actual_ratings['userId'] == user_id) & 
                    (actual_ratings['rating'] >= threshold)
                ]['movieId'].tolist()
            )
            
            # Calculate precision
            hits = len(set(top_k_movies) & actual_high_rated)
            precision = hits / k if k > 0 else 0
            
            return precision
        
        return 0.0