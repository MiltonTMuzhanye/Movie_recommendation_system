import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .movie-genres {
        color: #666;
        font-size: 0.9rem;
    }
    .movie-score {
        color: #FF4B4B;
        font-weight: bold;
    }
    .recommendation-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def call_api(endpoint: str, method: str = "GET", params: dict = None):
    """Make API call to recommendation service"""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=params)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_movies(movies: list, title: str = "Movies"):
    """Display a list of movies"""
    if not movies:
        st.info("No movies found")
        return
    
    st.subheader(title)
    
    cols = st.columns(4)
    for idx, movie in enumerate(movies):
        with cols[idx % 4]:
            with st.container():
                st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown Title')}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div class='movie-genres'>🎭 {movie.get('genres', 'N/A')}</div>", 
                           unsafe_allow_html=True)
                
                if 'score' in movie:
                    st.markdown(f"<div class='movie-score'>⭐ Score: {movie['score']:.3f}</div>", 
                               unsafe_allow_html=True)
                
                if 'similarity_score' in movie:
                    st.markdown(f"<div class='movie-score'>🔗 Similarity: {movie['similarity_score']:.3f}</div>", 
                               unsafe_allow_html=True)
                
                if 'popularity_score' in movie:
                    st.markdown(f"<div class='movie-score'>🔥 Popularity: {movie['popularity_score']:.3f}</div>", 
                               unsafe_allow_html=True)
                
                st.markdown(f"<small>ID: {movie['movie_id']}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Year: {movie.get('year', 'Unknown')}</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application"""
    st.markdown("<h1 class='main-header'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Home", "Get Recommendations", "Similar Movies", "Popular Movies", "About"]
    )
    
    # Model selection
    st.sidebar.subheader("Settings")
    model_type = st.sidebar.selectbox(
        "Recommendation Model",
        ["hybrid", "content_based", "collaborative", "popularity", "mf", "neural"]
    )
    
    top_k = st.sidebar.slider("Number of recommendations", 5, 50, 10)
    
    # Home page
    if page == "Home":
        st.markdown("""
        ## Welcome to the Movie Recommendation System! 🎬
        
        This system uses multiple recommendation algorithms to provide you with personalized movie recommendations.
        
        ### Features:
        - **Personalized Recommendations**: Get movies tailored to your taste
        - **Similar Movies**: Find movies similar to your favorites
        - **Popular Movies**: Discover what's trending
        - **Multiple Models**: Choose from different recommendation approaches
        
        ### How to use:
        1. Navigate to **Get Recommendations** to see personalized picks
        2. Try **Similar Movies** to find movies like your favorites
        3. Check **Popular Movies** for the most popular picks
        """)
        
        # System status
        try:
            health = requests.get(f"{API_URL}/../health")
            if health.status_code == 200:
                status = health.json()
                st.success(f"✅ System Status: {status['status']}")
                st.info(f"Models Loaded: {status['models_loaded']}")
                st.info(f"Data Loaded: {status['data_loaded']}")
        except:
            st.error("⚠️ Cannot connect to API server")
    
    # Get Recommendations page
    elif page == "Get Recommendations":
        st.header("Get Personalized Recommendations")
        
        user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = call_api(
                    f"/recommendations/{user_id}",
                    params={"model": model_type, "top_k": top_k}
                )
                
                if recommendations:
                    display_movies(recommendations, f"Recommendations for User {user_id}")
                    
                    # Explanation (optional)
                    if st.checkbox("Show explanation"):
                        explanation = call_api(f"/explain/{user_id}/{recommendations[0]['movie_id']}")
                        if explanation:
                            st.subheader("Recommendation Explanation")
                            st.write(explanation.get('summary', 'No explanation available'))
    
    # Similar Movies page
    elif page == "Similar Movies":
        st.header("Find Similar Movies")
        
        movie_id = st.number_input("Enter Movie ID", min_value=1, value=1, step=1)
        similarity_model = st.selectbox(
            "Similarity Model",
            ["content_based", "collaborative"],
            index=0
        )
        
        if st.button("Find Similar Movies", type="primary"):
            with st.spinner("Finding similar movies..."):
                similar = call_api(
                    f"/similar/{movie_id}",
                    params={"model": similarity_model, "top_k": top_k}
                )
                
                if similar:
                    display_movies(similar, f"Similar Movies to Movie ID {movie_id}")
    
    # Popular Movies page
    elif page == "Popular Movies":
        st.header("Popular Movies")
        
        if st.button("Show Popular Movies", type="primary"):
            with st.spinner("Loading popular movies..."):
                popular = call_api(
                    "/popular",
                    params={"top_k": top_k}
                )
                
                if popular:
                    display_movies(popular, "Popular Movies")
    
    # About page
    elif page == "About":
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Goal
        Provide personalized movie recommendations using multiple algorithms.
        
        ### 🧠 Models Used
        - **Popularity Model**: Simple popularity-based recommendations
        - **Content-Based**: Recommendations based on movie features
        - **Collaborative Filtering**: User-item interaction patterns
        - **Matrix Factorization**: Latent factor model
        - **Neural Recommender**: Deep learning approach
        - **Hybrid Model**: Combined approach for better results
        
        ### 📊 Data
        - Dataset: MovieLens Latest Small
        - 9,742 movies
        - 100,836 ratings
        - 610 users
        
        ### 🏗️ Architecture
        - FastAPI backend
        - Streamlit frontend
        - Redis cache
        - MLflow tracking
        - Docker deployment
        
        ### 📈 Metrics
        - Precision@10
        - Recall@10
        - MAP@10
        - NDCG@10
        
        ### 🔧 Technologies
        - Python
        - FastAPI
        - Streamlit
        - Scikit-learn
        - TensorFlow
        - Redis
        - Docker
        - MLflow
        """)

if __name__ == "__main__":
    main()