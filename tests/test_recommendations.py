import unittest
import pandas as pd
import numpy as np
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

class TestRecommendations(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.movies = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Toy Story', 'Jumanji', 'Toy Story 2', 'Toy Soldiers', 'Now and Then'],
            'genres': [['Animation'], ['Adventure'], ['Animation'], ['Action'], ['Drama']]
        })
        
        # Create mock similarity matrix
        self.similarity_matrix = np.array([
            [1.0, 0.1, 0.8, 0.3, 0.2],
            [0.1, 1.0, 0.2, 0.1, 0.1],
            [0.8, 0.2, 1.0, 0.4, 0.3],
            [0.3, 0.1, 0.4, 1.0, 0.1],
            [0.2, 0.1, 0.3, 0.1, 1.0]
        ])
    
    def test_content_recommendations(self):
        recommender = ContentBasedRecommender(self.similarity_matrix, self.movies)
        recommendations = recommender.recommend_similar_movies(1, top_n=2)
        
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations.iloc[0]['movieId'], 3)  # Toy Story 2 should be most similar
        self.assertGreater(recommendations.iloc[0]['similarity_score'], 0.5)
    
    def test_find_movie_by_title(self):
        recommender = ContentBasedRecommender(self.similarity_matrix, self.movies)
        results = recommender.find_movie_by_title('Toy')
        
        self.assertEqual(len(results), 3)  # Should find 3 movies with 'Toy' in title

if __name__ == '__main__':
    unittest.main()