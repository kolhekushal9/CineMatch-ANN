import os
import pandas as pd
from app import app
from models import db, Movie

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_CSV_PATH = os.path.join(BASE_DIR, 'model_training', 'movies_processed.csv')

def init_db():
    if not os.path.exists(MOVIES_CSV_PATH):
        print(f"Error: {MOVIES_CSV_PATH} not found. Please run train_model.py first.")
        return

    with app.app_context():
        # Create all tables safely
        print("Creating DB tables...")
        db.create_all()
        
        # Check if database is already populated
        if Movie.query.first():
            print("Database already initialized with movies.")
            return

        print(f"Loading movies from {MOVIES_CSV_PATH}...")
        df = pd.read_csv(MOVIES_CSV_PATH)
        
        # Optimize by batch inserting
        movies_to_insert = []
        for index, row in df.iterrows():
            movie = Movie(
                movie_id=row['movieId'],
                movie_encoded=row['movie_encoded'],
                title=row['title'],
                genres=row['genres']
            )
            movies_to_insert.append(movie)
            
            if len(movies_to_insert) > 1000:
                db.session.bulk_save_objects(movies_to_insert)
                db.session.commit()
                movies_to_insert = []
                print(f"Inserted up to index {index}...")
                
        if movies_to_insert:
            db.session.bulk_save_objects(movies_to_insert)
            db.session.commit()
            
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()
