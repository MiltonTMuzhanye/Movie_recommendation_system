from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class MovieInfo(BaseModel):
    """Movie information model"""
    movie_id: int
    title: str
    genres: str
    score: Optional[float] = None
    year: Optional[str] = None
    similarity_score: Optional[float] = None
    popularity_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    """Recommendation response model"""
    user_id: int
    recommendations: List[MovieInfo]
    model_used: str
    timestamp: datetime

class SimilarMoviesResponse(BaseModel):
    """Similar movies response model"""
    movie_id: int
    similar_movies: List[MovieInfo]
    model_used: str

class PopularMoviesResponse(BaseModel):
    """Popular movies response model"""
    movies: List[MovieInfo]
    timestamp: datetime

class FeedbackRequest(BaseModel):
    """Feedback request model"""
    user_id: int
    movie_id: int
    rating: Optional[float] = None
    liked: Optional[bool] = None
    feedback_text: Optional[str] = None
    timestamp: datetime

class ModelInfo(BaseModel):
    """Model information model"""
    name: str
    type: str
    version: str
    metrics: Dict[str, float]
    trained_date: datetime

class SystemStatus(BaseModel):
    """System status model"""
    status: str
    models_loaded: bool
    data_loaded: bool
    cache_status: str
    uptime: float