import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'model_training', 'model_weights.npz')
MAPPINGS_PATH = os.path.join(BASE_DIR, 'model_training', 'mappings.json')
KNN_PATH = os.path.join(BASE_DIR, 'model_training', 'knn_similarities.json')

def relu(x):
    return np.maximum(0, x)

class RecommenderService:
    def __init__(self):
        self.weights = None
        self.mappings = None
        self.num_movies = 0
        self.num_users = 0
        self.knn_similarities = {}
        
        self.load_model_and_mappings()

    def load_model_and_mappings(self):
        if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(MAPPINGS_PATH):
            print("Warning: Weights or mappings not found. Please train/export the model first.")
            return

        print("Loading ANN numpy weights and mappings for recommendations...")
        with open(MAPPINGS_PATH, 'r') as f:
            self.mappings = json.load(f)
            self.num_movies = self.mappings['num_movies']
            self.num_users = self.mappings['num_users']
            
        if os.path.exists(KNN_PATH):
            with open(KNN_PATH, 'r') as f:
                self.knn_similarities = json.load(f)
            
        self.weights = np.load(WEIGHTS_PATH)

    def predict(self, user_array, movie_array):
        # Extract weight matrices
        W_u = self.weights['user_embedding']
        W_m = self.weights['movie_embedding']
        
        # Cold start handling: if user index is out of bounds, clip to max
        # (Though our tf embedding had +10000, numpy arrays have exact dimension)
        user_array = np.clip(user_array, 0, W_u.shape[0] - 1)
        movie_array = np.clip(movie_array, 0, W_m.shape[0] - 1)

        # 1. Embedding Lookup
        u_emb = W_u[user_array] # shape (batch, 50)
        m_emb = W_m[movie_array] # shape (batch, 50)
        
        # Flatten is technically a no-op here since seq_len=1, but we concat them
        X = np.concatenate([u_emb, m_emb], axis=1) # shape (batch, 100)
        
        # 2. ANN Forward Pass
        # Dense 1 (128)
        X = np.dot(X, self.weights['dense_1_w']) + self.weights['dense_1_b']
        X = relu(X)
        
        # Dense 2 (64)
        X = np.dot(X, self.weights['dense_2_w']) + self.weights['dense_2_b']
        X = relu(X)
        
        # Dense 3 (32)
        X = np.dot(X, self.weights['dense_3_w']) + self.weights['dense_3_b']
        X = relu(X)
        
        # Output (1)
        X = np.dot(X, self.weights['dense_4_w']) + self.weights['dense_4_b']
        
        return X

    def get_recommendations(self, user_encoded, movie_encoded_list, top_k=10):
        if self.weights is None:
            return []

        user_array = np.array([user_encoded] * len(movie_encoded_list), dtype=np.int32)
        movie_array = np.array(movie_encoded_list, dtype=np.int32)

        predictions = self.predict(user_array, movie_array)
        predictions = predictions.flatten()

        top_indices = predictions.argsort()[-top_k:][::-1]

        results = [(movie_encoded_list[i], float(predictions[i])) for i in top_indices]
        return results

    def get_knn_similar_movies(self, movie_encoded):
        # Retrieve pre-computed mathematically similar movies from the parsed mapping
        similar_encoded_list = self.knn_similarities.get(str(movie_encoded), [])
        return similar_encoded_list

# Singleton instance
recommender = RecommenderService()
