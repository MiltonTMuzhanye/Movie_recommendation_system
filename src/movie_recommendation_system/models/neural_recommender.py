import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from src.movie_recommendation.utils.logger import logger

class NeuralRecommender:
    """
    Neural collaborative filtering recommendation model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize neural recommender
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.user_ids = None
        self.movie_ids = None
        self.user_mapping = {}
        self.movie_mapping = {}
        
    def build_model(self, n_users: int, n_movies: int):
        """
        Build neural network architecture
        
        Args:
            n_users: Number of users
            n_movies: Number of movies
        """
        # Get embedding dimensions
        embedding_dim = self.config.get('models', {})
            .get('neural_recommender', {})
            .get('embedding_dim', 32)
        
        # Define input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        movie_input = layers.Input(shape=(1,), name='movie_input')
        
        # Embedding layers
        user_embedding = layers.Embedding(
            n_users, embedding_dim, 
            embeddings_initializer='he_normal', 
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='user_embedding'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        movie_embedding = layers.Embedding(
            n_movies, embedding_dim, 
            embeddings_initializer='he_normal', 
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='movie_embedding'
        )(movie_input)
        movie_embedding = layers.Flatten()(movie_embedding)
        
        # Concatenate embeddings
        concatenated = layers.Concatenate()([user_embedding, movie_embedding])
        
        # Hidden layers
        hidden_layers = self.config.get('models', {})
            .get('neural_recommender', {})
            .get('hidden_layers', [64, 32])
        
        x = layers.Dropout(0.2)(concatenated)
        for units in hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = keras.Model(
            inputs=[user_input, movie_input],
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.get('models', {})
                    .get('neural_recommender', {})
                    .get('learning_rate', 0.001)
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        logger.info(f"Built neural recommender with {n_users} users and {n_movies} movies")
    
    def fit(self, ratings_df: pd.DataFrame):
        """
        Fit the neural recommender model
        
        Args:
            ratings_df: Ratings dataframe
        """
        logger.info("Fitting neural recommender model")
        
        # Create user and movie mappings
        self.user_ids = sorted(ratings_df['userId'].unique())
        self.movie_ids = sorted(ratings_df['movieId'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_ids)}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(self.movie_ids)}
        
        # Convert to binary feedback (rating >= 3.5)
        ratings_df['positive'] = (ratings_df['rating'] >= 3.5).astype(int)
        
        # Prepare training data
        user_indices = [self.user_mapping[user] for user in ratings_df['userId']]
        movie_indices = [self.movie_mapping[movie] for movie in ratings_df['movieId']]
        targets = ratings_df['positive'].values
        
        # Split data
        X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = train_test_split(
            user_indices, movie_indices, targets,
            test_size=0.2,
            random_state=42
        )
        
        # Build model
        self.build_model(
            n_users=len(self.user_ids),
            n_movies=len(self.movie_ids)
        )
        
        # Training parameters
        batch_size = self.config.get('models', {})
            .get('neural_recommender', {})
            .get('batch_size', 256)
        
        epochs = self.config.get('models', {})
            .get('neural_recommender', {})
            .get('epochs', 50)
        
        # Train model
        history = self.model.fit(
            [X_user_train, X_movie_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_user_test, X_movie_test], y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        logger.info(f"Neural recommender fitted with {len(self.user_ids)} users "
                   f"and {len(self.movie_ids)} movies")
        logger.info(f"Training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    def recommend_for_user(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get predictions for all movies
        movie_indices = list(range(len(self.movie_ids)))
        user_indices = [user_idx] * len(movie_indices)
        
        predictions = self.model.predict(
            [user_indices, movie_indices],
            batch_size=1024,
            verbose=0
        ).flatten()
        
        # Get top k
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            movie_id = self.movie_ids[idx]
            score = float(predictions[idx])
            recommendations.append((movie_id, score))
        
        return recommendations