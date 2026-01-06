from pydantic import BaseModel
from typing import List, Optional

class Movie(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    year: Optional[int] = None

class Recommendation(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    score: float
    score_type: str  # "similarity", "predicted_rating", "hybrid"

class UserRecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10
    include_explanations: bool = True

class ContentRecommendationRequest(BaseModel):
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    top_n: int = 10
    min_similarity: float = 0.1

class HybridRecommendationRequest(BaseModel):
    user_id: int
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    top_n: int = 10
    content_weight: float = 0.5
    collab_weight: float = 0.5