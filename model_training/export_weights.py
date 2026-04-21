import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
OUT_PATH = os.path.join(BASE_DIR, 'model_weights.npz')

def export_weights():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    weights = {}
    
    dense_idx = 1
    for layer in model.layers:
        name = layer.name.lower()
        if not layer.weights:
            continue
            
        params = layer.get_weights()
        if 'user' in name and 'embedding' in name:
            weights['user_embedding'] = params[0]
        elif 'movie' in name and 'embedding' in name:
            weights['movie_embedding'] = params[0]
        elif 'dense' in name:
            weights[f'dense_{dense_idx}_w'] = params[0]
            weights[f'dense_{dense_idx}_b'] = params[1]
            dense_idx += 1
            
    print("Exporting numpy arrays:", list(weights.keys()))
    np.savez_compressed(OUT_PATH, **weights)
    print(f"Successfully saved compressed weights to {OUT_PATH}! Size is roughly {os.path.getsize(OUT_PATH) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    export_weights()
