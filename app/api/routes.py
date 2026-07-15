from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.api.schemas import (
    RecommendationResponse,
    MovieInfo,
    SimilarMoviesResponse,
    PopularMoviesResponse,
    FeedbackRequest
)
from app.api.main import recommender
from src.movie_recommendation.utils.logger import logger

router = APIRouter()

@router.get("/recommendations/{user_id}", response_model=List[MovieInfo])
async def get_recommendations(
    user_id: int,
    model: str = Query("hybrid", description="Model type: popularity, content_based, collaborative, mf, neural, hybrid"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get personalized recommendations for a user
    """
    try:
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            model_type=model,
            top_k=top_k
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{movie_id}", response_model=List[MovieInfo])
async def get_similar_movies(
    movie_id: int,
    model: str = Query("content_based", description="Model type for similarity"),
    top_k: int = Query(10, ge=1, le=50, description="Number of similar movies")
):
    """
    Get movies similar to a given movie
    """
    try:
        similar_movies = recommender.get_similar_movies(
            movie_id=movie_id,
            model_type=model,
            top_k=top_k
        )
        
        if not similar_movies:
            raise HTTPException(status_code=404, detail="No similar movies found")
        
        return similar_movies
        
    except Exception as e:
        logger.error(f"Error getting similar movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/popular", response_model=List[MovieInfo])
async def get_popular_movies(
    top_k: int = Query(10, ge=1, le=50, description="Number of popular movies")
):
    """
    Get popular movies
    """
    try:
        popular_movies = recommender.get_popular_movies(top_k=top_k)
        
        if not popular_movies:
            raise HTTPException(status_code=404, detail="No popular movies found")
        
        return popular_movies
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending", response_model=List[MovieInfo])
async def get_trending_movies(
    top_k: int = Query(10, ge=1, le=50, description="Number of trending movies")
):
    """
    Get trending movies (based on recent ratings)
    """
    # Placeholder - implement trending logic
    # For now, return popular movies
    return await get_popular_movies(top_k)

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback for recommendations
    """
    try:
        # Log feedback for analysis
        logger.info(f"Feedback received: {feedback.dict()}")
        
        # Store feedback (e.g., in database)
        # This would require database integration
        
        return {
            "status": "success",
            "message": "Feedback received",
            "feedback_id": "feedback_" + str(hash(feedback.user_id + feedback.movie_id))
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explain/{user_id}/{movie_id}")
async def explain_recommendation(user_id: int, movie_id: int):
    """
    Get explanation for a recommendation
    """
    try:
        # This would require more sophisticated explanation logic
        explanation = {
            "user_id": user_id,
            "movie_id": movie_id,
            "factors": [
                {"factor": "genre_match", "value": 0.85},
                {"factor": "user_similarity", "value": 0.72},
                {"factor": "popularity", "value": 0.65}
            ],
            "summary": "Movie recommended based on your preference for Action and Comedy genres"
        }
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))