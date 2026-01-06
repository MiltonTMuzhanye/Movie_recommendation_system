import unittest
from fastapi.testclient import TestClient
from src.api.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    def test_popular_endpoint(self):
        response = self.client.get("/popular?top_n=5")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

if __name__ == '__main__':
    unittest.main()