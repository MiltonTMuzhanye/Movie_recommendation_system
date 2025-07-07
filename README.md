# Movie_recommendation_system (Content Based, Collaborative, and Hybrid)

This project is a complete movie recommendation system built using the [MovieLens Dataset (ml-latest-small)](https://grouplens.org/datasets/movielens/). It demonstrates multiple recommendation techniques including:

- Content Based Filtering
- Collaborative Filtering (User Based KNN & Matrix Factorization with SVD)
- Hybrid Recommendation Approach
- Cold Start Strategy using Popular Movies
- Evaluation with RMSE & MAE

 ## Dataset
- Source: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)
- Movies: 9,742 records
- Ratings: 100,836 ratings from 610 users

 ## Techniques Used

  ### Content-Based Filtering
  - TF-IDF Vectorization on movie titles
  - Cosine Similarity to find similar movies
  - Example: Recommending movies similar to **"Toy Story"**
  
  ### Collaborative Filtering
  - **User Based KNN** using cosine similarity
  - **SVD** (Singular Value Decomposition) for matrix factorization
  - Predicts ratings a user would give to unseen movies
  
  ### Hybrid Recommendation System
  - Combines normalized content similarity and collaborative scores
  - Returns personalized, content aware recommendations
  
  ### Cold Start Handling
  - Recommends the most popular movies when the user has no prior history
  - Uses average rating and rating count


## Model Evaluation

| Model | RMSE | MAE | CV RMSE | CV MAE |
|-------|------|-----|---------|--------|
| KNN   | ~0.976 | ~0.752 | ~0.980 | ~0.755 |
| SVD   | ~0.879 | ~0.674 | ~0.880 | ~0.677 |

Evaluation was done using train_test_split and 3 fold cross validation.
