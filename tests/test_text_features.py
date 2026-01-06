import unittest
import pandas as pd
import numpy as np
from src.features.text_vectorizer import TextVectorizer

class TestTextFeatures(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TextVectorizer()
        self.titles = pd.Series([
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ])
    
    def test_fit_transform(self):
        matrix = self.vectorizer.fit_transform(self.titles)
        self.assertEqual(matrix.shape[0], len(self.titles))
        self.assertGreater(matrix.shape[1], 0)
    
    def test_similarity_matrix(self):
        self.vectorizer.fit_transform(self.titles)
        similarity_matrix = self.vectorizer.compute_similarity_matrix()
        
        self.assertEqual(similarity_matrix.shape, (len(self.titles), len(self.titles)))
        self.assertTrue(np.allclose(np.diag(similarity_matrix), 1.0))

if __name__ == '__main__':
    unittest.main()