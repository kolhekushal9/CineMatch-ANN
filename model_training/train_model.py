import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
MAPPINGS_PATH = os.path.join(BASE_DIR, 'mappings.json')
MOVIES_PROCESSED_PATH = os.path.join(BASE_DIR, 'movies_processed.csv')

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)

# MovieLens 100k latest small dataset
DATASET_URL = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
ZIP_PATH = os.path.join(DATA_DIR, 'ml-latest-small.zip')
EXTRACTED_DIR = os.path.join(DATA_DIR, 'ml-latest-small')

def download_data():
    if not os.path.exists(EXTRACTED_DIR):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Download and extraction complete.")

def prepare_data():
    print("Preparing data...")
    ratings_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'ratings.csv'))
    movies_df = pd.read_csv(os.path.join(EXTRACTED_DIR, 'movies.csv'))

    # Encode user and movie ids to start from 0
    user_ids = ratings_df['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    
    movie_ids = ratings_df['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    
    ratings_df['user_encoded'] = ratings_df['userId'].map(user2user_encoded)
    ratings_df['movie_encoded'] = ratings_df['movieId'].map(movie2movie_encoded)
    
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    
    # Save the processed movies so we can insert them into our DB later
    # Keep the encoded map included as well
    movies_df['movie_encoded'] = movies_df['movieId'].map(movie2movie_encoded)
    # Some movies might not have ratings and won't get encoded. We assign an arbitrary id.
    max_encoded = ratings_df['movie_encoded'].max()
    movies_df['movie_encoded'] = movies_df['movie_encoded'].fillna(-1).astype(int)
    # Filter out movies with no ratings to keep the system simple
    movies_df = movies_df[movies_df['movie_encoded'] != -1]
    
    movies_df.to_csv(MOVIES_PROCESSED_PATH, index=False)

    mappings = {
        'num_users': num_users,
        'num_movies': num_movies,
        'user2user_encoded': user2user_encoded,
        'movie2movie_encoded': movie2movie_encoded,
        # Inverse mapping to recover movie id
        'movie_encoded2movie': {i: x for x, i in movie2movie_encoded.items()}
    }
    
    with open(MAPPINGS_PATH, 'w') as f:
        json.dump(mappings, f)
        
    return ratings_df, num_users, num_movies

def build_model(num_users, num_movies, embedding_size=50):
    user_input = layers.Input(shape=(1,), name='user_encoded')
    # Adding +10000 to vocab size to allow new users created in the web app
    # to be simply mapped to higher integer indices and use the cold start embeddings.
    user_embedding = layers.Embedding(
        input_dim=num_users + 10000, 
        output_dim=embedding_size, 
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name='user_embedding'
    )(user_input)
    user_vector = layers.Flatten(name='flatten_users')(user_embedding)

    movie_input = layers.Input(shape=(1,), name='movie_encoded')
    movie_embedding = layers.Embedding(
        input_dim=num_movies + 100, 
        output_dim=embedding_size, 
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name='movie_embedding'
    )(movie_input)
    movie_vector = layers.Flatten(name='flatten_movies')(movie_embedding)

    concat = layers.Concatenate()([user_vector, movie_vector])

    # Dense layers for the Artificial Neural Network
    dense_1 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(concat)
    dropout_1 = layers.Dropout(0.2)(dense_1)
    dense_2 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(dropout_1)
    dropout_2 = layers.Dropout(0.2)(dense_2)
    dense_3 = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(dropout_2)

    # Output layer emitting a predicted rating
    output = layers.Dense(1)(dense_3)

    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

if __name__ == '__main__':
    download_data()
    # To save time and keep it simple, we use a sample of ratings data to train quickly
    ratings_df, num_users, num_movies = prepare_data()
    
    print(f"Num Users: {num_users}, Num Movies: {num_movies}")
    print("Normalizing target ratings to be faster / more stable training...")
    # Min-Max scale ratings to [0, 1] using known min/max (0.5 - 5.0) for MovieLens
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()
    ratings_df['rating_normalized'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    
    # Validation split
    ratings_df = ratings_df.sample(frac=1, random_state=42)
    train_size = int(0.8 * len(ratings_df))
    train_df = ratings_df.iloc[:train_size]
    val_df = ratings_df.iloc[train_size:]
    
    x_train = [train_df['user_encoded'].values, train_df['movie_encoded'].values]
    y_train = train_df['rating_normalized'].values

    x_val = [val_df['user_encoded'].values, val_df['movie_encoded'].values]
    y_val = val_df['rating_normalized'].values
    
    print("Building model...")
    model = build_model(num_users, num_movies)
    model.summary()
    
    print("Training Artificial Neural Network Model...")
    # Training for just 3 epochs to make setup fast.
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=256,
        epochs=3,
        validation_data=(x_val, y_val)
    )
    
    print("Saving model and artifacts...")
    model.save(MODEL_PATH)
    print("Done! ANN model is trained and ready for recommendations.")
