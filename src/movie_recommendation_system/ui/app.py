import streamlit as st
import pandas as pd
import requests
import json

# Page config
st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Hybrid Movie Recommendation Engine")
st.markdown("""
This system combines **content-based filtering** (movie similarities) with **collaborative filtering** (user behavior) 
to provide personalized movie recommendations.
""")

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Hybrid Recommendations", 
    "📊 Content-Based", 
    "👥 Collaborative", 
    "📈 Popular Movies"
])

with tab1:
    st.header("Hybrid Recommendations")
    st.markdown("Get personalized recommendations based on both your preferences and movie content.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1, step=1, key="hybrid_user")
        movie_title = st.text_input("Movie you liked", "Toy Story", key="hybrid_movie")
    
    with col2:
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="hybrid_top_n")
        content_weight = st.slider("Content weight", 0.0, 1.0, 0.5, 0.1, key="hybrid_weight")
        collab_weight = st.slider("Collaborative weight", 0.0, 1.0, 0.5, 0.1, key="collab_weight")
    
    if st.button("Get Hybrid Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    f"{api_url}/recommend",
                    json={
                        "user_id": user_id,
                        "movie_title": movie_title,
                        "recommendation_type": "hybrid",
                        "top_n": top_n
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data['recommendations']
                    
                    st.subheader(f"Recommendations for User {user_id} who liked '{movie_title}'")
                    
                    # Display recommendations in a nice format
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec['title']} (Score: {rec.get('hybrid_score', rec.get('similarity_score', 0)):.3f})"):
                            st.write(f"**Genres:** {rec['genres']}")
                            if 'content_score' in rec and 'collab_score' in rec:
                                st.write(f"**Content score:** {rec['content_score']:.3f}")
                                st.write(f"**Collaborative score:** {rec['collab_score']:.3f}")
                            elif 'similarity_score' in rec:
                                st.write(f"**Similarity score:** {rec['similarity_score']:.3f}")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error fetching recommendations: {str(e)}")

with tab2:
    st.header("Content-Based Recommendations")
    st.markdown("Find movies similar to a given movie based on content (title, genres).")
    
    movie_query = st.text_input("Enter a movie title", "Inception", key="content_movie")
    top_n_content = st.slider("Number of recommendations", 5, 20, 10, key="content_top_n")
    
    if st.button("Find Similar Movies"):
        with st.spinner("Finding similar movies..."):
            try:
                response = requests.post(
                    f"{api_url}/recommend",
                    json={
                        "movie_title": movie_query,
                        "recommendation_type": "content",
                        "top_n": top_n_content
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data['recommendations']
                    
                    st.subheader(f"Movies similar to '{movie_query}'")
                    
                    # Create a DataFrame for better display
                    df = pd.DataFrame(recommendations)
                    if 'similarity_score' in df.columns:
                        df['similarity_score'] = df['similarity_score'].round(3)
                    
                    st.dataframe(
                        df[['title', 'genres', 'similarity_score']] if 'similarity_score' in df.columns else df[['title', 'genres']],
                        use_container_width=True
                    )
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Collaborative Filtering Recommendations")
    st.markdown("Get recommendations based on users with similar tastes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        collab_user_id = st.number_input("User ID", min_value=1, max_value=1000, value=5, step=1, key="collab_user")
    
    with col2:
        top_n_collab = st.slider("Number of recommendations", 5, 20, 10, key="collab_top_n")
    
    if st.button("Get Personalized Recommendations"):
        with st.spinner("Generating personalized recommendations..."):
            try:
                response = requests.post(
                    f"{api_url}/recommend",
                    json={
                        "user_id": collab_user_id,
                        "recommendation_type": "collaborative",
                        "top_n": top_n_collab
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data['recommendations']
                    
                    st.subheader(f"Personalized recommendations for User {collab_user_id}")
                    
                    df = pd.DataFrame(recommendations)
                    if 'predicted_rating' in df.columns:
                        df['predicted_rating'] = df['predicted_rating'].round(3)
                    
                    st.dataframe(
                        df[['title', 'genres', 'predicted_rating']] if 'predicted_rating' in df.columns else df[['title', 'genres']],
                        use_container_width=True
                    )
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab4:
    st.header("Popular Movies")
    st.markdown("Top movies based on ratings and popularity (good for new users).")
    
    top_n_popular = st.slider("Number of movies", 5, 50, 10, key="popular_top_n")
    
    if st.button("Show Popular Movies"):
        with st.spinner("Loading popular movies..."):
            try:
                response = requests.get(
                    f"{api_url}/popular",
                    params={"top_n": top_n_popular}
                )
                
                if response.status_code == 200:
                    popular_movies = response.json()
                    
                    df = pd.DataFrame(popular_movies)
                    df['avg_rating'] = df['avg_rating'].round(2)
                    
                    st.dataframe(
                        df[['title', 'genres', 'avg_rating', 'rating_count']],
                        use_container_width=True
                    )
                    
                    # Visualization
                    st.subheader("Popularity Distribution")
                    fig, ax = plt.subplots()
                    ax.barh(df['title'].head(10), df['rating_count'].head(10))
                    ax.set_xlabel('Number of Ratings')
                    st.pyplot(fig)
                    
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This recommendation engine uses:
- **Content-based filtering**: TF-IDF on movie titles
- **Collaborative filtering**: SVD matrix factorization
- **Hybrid approach**: Weighted combination of both methods
""")