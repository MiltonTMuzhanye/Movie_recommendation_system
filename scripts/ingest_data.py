import sys
import os
from pathlib import Path
import pandas as pd
import requests
import zipfile
import io
from src.movie_recommendation.utils.logger import logger
from src.movie_recommendation.data.ingestion import DataIngestion
from src.movie_recommendation.utils.config import config

def download_movielens():
    """Download MovieLens dataset"""
    logger.info("Downloading MovieLens dataset")
    
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    data_dir = Path(config.get('data.raw_path', 'data/raw'))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
        
        logger.info(f"Dataset downloaded and extracted to {data_dir}")
        
        # Move files to raw directory
        extracted_dir = data_dir / "ml-latest-small"
        if extracted_dir.exists():
            for file in extracted_dir.glob("*.csv"):
                file.rename(data_dir / file.name)
            extracted_dir.rmdir()
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False

def main():
    """Main ingestion function"""
    logger.info("Starting data ingestion")
    
    # Download dataset if not present
    raw_path = Path(config.get('data.raw_path', 'data/raw'))
    if not (raw_path / "movies.csv").exists():
        logger.info("Dataset not found, downloading...")
        success = download_movielens()
        if not success:
            logger.error("Failed to download dataset")
            sys.exit(1)
    
    # Load and validate data
    ingestion = DataIngestion(config)
    try:
        movies_df, ratings_df = ingestion.load_movielens(str(raw_path))
        
        logger.info(f"Successfully loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        # Save basic stats
        stats = {
            'movies_count': len(movies_df),
            'ratings_count': len(ratings_df),
            'users_count': ratings_df['userId'].nunique(),
            'avg_rating': ratings_df['rating'].mean(),
            'rating_std': ratings_df['rating'].std()
        }
        
        # Save stats
        import json
        with open('data/stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Data ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()