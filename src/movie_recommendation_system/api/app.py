from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

app = FastAPI(title="Movie Recommendation API", 
              description="Hybrid recommendation engine combining content-based and collaborative filtering")

# Initialize models (in production, load from disk)
models_initialized = False
content_rec = None
collab_rec = None
hybrid_rec = None

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    recommendation_type: str = "hybrid"  # "content", "collaborative", "hybrid"
    top_n: int = 10

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    recommendation_type: str
    user_id: Optional[int]
    reference_movie: Optional[dict]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global content_rec, collab_rec, hybrid_rec, models_initialized
    
    # Load data
    from src.data.load_data import DataLoader
    from src.data.clean_data import DataCleaner
    from src.features.text_vectorizer import TextVectorizer
    
    try:
        loader = DataLoader("data/raw")
        movies, ratings = loader.load_movielens_data()
        
        cleaner = DataCleaner()
        movies_clean = cleaner.preprocess_movies(movies)
        ratings_clean = cleaner.preprocess_ratings(ratings)
        
        # Initialize content-based model
        vectorizer = TextVectorizer()
        title_matrix = vectorizer.fit_transform(movies_clean['title'])
        similarity_matrix = vectorizer.compute_similarity_matrix()
        
        content_rec = ContentBasedRecommender(similarity_matrix, movies_clean)
        
        # Initialize collaborative model
        collab_rec = CollaborativeRecommender(model_type='svd')
        collab_rec.train(ratings_clean)
        
        # Initialize hybrid model
        hybrid_rec = HybridRecommender(
            content_rec,
            collab_rec,
            movies_clean,
            weights={'content': 0.5, 'collab': 0.5}
        )
        
        models_initialized = True
        print("Models initialized successfully")
        
    except Exception as e:
        print(f"Error initializing models: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Movie Recommendation API", "status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations"""
    if not models_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        if request.recommendation_type == "content":
            if request.movie_id is None and request.movie_title is None:
                raise HTTPException(status_code=400, detail="movie_id or movie_title required for content-based recommendations")
            
            # Find movie if title provided
            if request.movie_id is None:
                movie_matches = content_rec.find_movie_by_title(request.movie_title)
                if len(movie_matches) == 0:
                    raise HTTPException(status_code=404, detail="Movie not found")
                request.movie_id = movie_matches.iloc[0]['movieId']
            
            recommendations = content_rec.recommend_similar_movies(
                request.movie_id, 
                top_n=request.top_n
            )
            recommendations = recommendations.to_dict('records')
            
            return RecommendationResponse(
                recommendations=recommendations,
                recommendation_type="content",
                user_id=request.user_id,
                reference_movie={"movie_id": request.movie_id}
            )
        
        elif request.recommendation_type == "collaborative":
            if request.user_id is None:
                raise HTTPException(status_code=400, detail="user_id required for collaborative recommendations")
            
            # Load data for collaborative recommendations
            loader = DataLoader("data/raw")
            movies, ratings = loader.load_movielens_data()
            
            recommendations = collab_rec.recommend_for_user(
                request.user_id,
                movies,
                ratings,
                top_n=request.top_n
            )
            recommendations = recommendations.to_dict('records')
            
            return RecommendationResponse(
                recommendations=recommendations,
                recommendation_type="collaborative",
                user_id=request.user_id,
                reference_movie=None
            )
        
        elif request.recommendation_type == "hybrid":
            if request.user_id is None:
                raise HTTPException(status_code=400, detail="user_id required for hybrid recommendations")
            if request.movie_id is None and request.movie_title is None:
                raise HTTPException(status_code=400, detail="movie_id or movie_title required for hybrid recommendations")
            
            loader = DataLoader("data/raw")
            movies, ratings = loader.load_movielens_data()
            
            recommendations = hybrid_rec.recommend(
                user_id=request.user_id,
                reference_movie_id=request.movie_id,
                reference_movie_title=request.movie_title,
                ratings_df=ratings,
                top_n=request.top_n
            )
            recommendations = recommendations.to_dict('records')
            
            return RecommendationResponse(
                recommendations=recommendations,
                recommendation_type="hybrid",
                user_id=request.user_id,
                reference_movie={"movie_id": request.movie_id, "movie_title": request.movie_title}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid recommendation_type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popular")
async def get_popular_movies(top_n: int = 10):
    """Get popular movies (cold-start fallback)"""
    loader = DataLoader("data/raw")
    movies, ratings = loader.load_movielens_data()
    
    avg_ratings = ratings.groupby('movieId')['rating'].mean()
    rating_counts = ratings.groupby('movieId')['rating'].count()
    
    popular_movies = pd.DataFrame({
        'avg_rating': avg_ratings,
        'rating_count': rating_counts
    }).sort_values(['rating_count', 'avg_rating'], ascending=False).head(top_n)
    
    popular_with_details = pd.merge(
        popular_movies,
        movies,
        left_index=True,
        right_on='movieId'
    )[['movieId', 'title', 'genres', 'avg_rating', 'rating_count']]
    
    return popular_with_details.to_dict('records')