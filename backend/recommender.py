import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model_training', 'model.h5')
MAPPINGS_PATH = os.path.join(BASE_DIR, 'model_training', 'mappings.json')

class RecommenderService:
    def __init__(self):
        self.model = None
        self.mappings = None
        self.num_movies = 0
        self.num_users = 0
        
        self.load_model_and_mappings()

    def load_model_and_mappings(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPINGS_PATH):
            print("Warning: Model or mappings not found. Please train the model first.")
            return

        print("Loading ANN model and mappings for recommendations...")
        # Load mappings
        with open(MAPPINGS_PATH, 'r') as f:
            self.mappings = json.load(f)
            self.num_movies = self.mappings['num_movies']
            self.num_users = self.mappings['num_users']
            
        # Compile=False makes loading faster because we only need it for inference
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def get_recommendations(self, user_encoded, movie_encoded_list, top_k=10):
        if self.model is None:
            return []

        # Prepare inputs for the Keras ANN model
        # We need an array of user_encoded of the same length as candidate movies
        user_array = np.array([user_encoded] * len(movie_encoded_list))
        movie_array = np.array(movie_encoded_list)

        # Predict ratings
        predictions = self.model.predict([user_array, movie_array], batch_size=512, verbose=0)
        # Predictions array is of shape (len, 1) usually. Flatten it.
        predictions = predictions.flatten()

        # Get indices of top k predictions
        top_indices = predictions.argsort()[-top_k:][::-1]

        # Return list of tuples: (movie_encoded, predicted_rating)
        results = [(movie_encoded_list[i], float(predictions[i])) for i in top_indices]
        return results

# Singleton instance
recommender = RecommenderService()
