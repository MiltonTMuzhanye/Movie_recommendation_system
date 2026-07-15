from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import mlflow
from app.api.routes import router
from app.api.schemas import RecommendationResponse, MovieInfo
from src.movie_recommendation.pipelines.recommendation_pipeline import RecommendationPipeline
from src.movie_recommendation.utils.logger import logger
import os

# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Real-time movie recommendation system with hybrid approach",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation pipeline
recommender = RecommendationPipeline({})

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Movie Recommendation API")
    
    # Load models
    recommender.load_models()
    recommender.load_data()
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    logger.info("API ready to serve requests")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(recommender.models) > 0,
        "data_loaded": recommender.movies_df is not None
    }

# Include routes
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )