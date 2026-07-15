import sys
import pandas as pd
import numpy as np
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.utils.config import config
from src.movie_recommendation.evaluation.ranking_metrics import RankingMetrics
from src.movie_recommendation.evaluation.explainability import ExplainabilityEngine
from src.movie_recommendation.utils.helpers import load_pickle

def main():
    """Main evaluation function"""
    logger.info("Starting model evaluation")
    
    # Load models
    models = {}
    model_names = ['popularity', 'content_based', 'collaborative', 'mf', 'neural', 'hybrid']
    
    for name in model_names:
        try:
            model_path = f"artifacts/trained_models/{name}_model.pkl"
            models[name] = load_pickle(model_path)
            logger.info(f"Loaded {name} model")
        except Exception as e:
            logger.warning(f"Could not load {name} model: {str(e)}")
    
    # Load test data
    try:
        ratings_df = pd.read_csv("data/processed/ratings_processed.csv")
        movies_df = pd.read_csv("data/processed/movies_processed.csv")
        logger.info(f"Loaded test data: {len(ratings_df)} ratings")
    except Exception as e:
        logger.error(f"Could not load test data: {str(e)}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = RankingMetrics()
    explainer = ExplainabilityEngine(config)
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating {name} model")
        
        try:
            # Evaluate on test data
            score = evaluator.evaluate_model(model, ratings_df)
            results[name] = score
            
            logger.info(f"{name} model results:")
            for metric, value in score.items():
                logger.info(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Error evaluating {name} model: {str(e)}")
    
    # Generate explanations for sample recommendations
    logger.info("Generating sample explanations")
    
    sample_user = ratings_df['userId'].iloc[0]
    sample_movie = movies_df['movieId'].iloc[0]
    
    for name, model in models.items():
        try:
            if hasattr(model, 'recommend_for_user'):
                recommendations = model.recommend_for_user(sample_user, top_k=5)
                
                if recommendations:
                    movie_id = recommendations[0][0]
                    explanation = explainer.explain_recommendation(
                        sample_user, movie_id,
                        movies_df, model
                    )
                    
                    logger.info(f"Explanation for {name} model:")
                    logger.info(f"  {explanation['summary']}")
                    
        except Exception as e:
            logger.warning(f"Could not generate explanation for {name}: {str(e)}")
    
    # Save evaluation results
    import json
    with open("reports/metrics/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()