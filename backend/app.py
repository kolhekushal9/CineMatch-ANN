import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# JWT Auth removed since login feature is removed
from models import db, User, Movie, Rating
from recommender import recommender

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# -----------------
# STATIC FILES INFO
# -----------------
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# -----------------
# API ENDPOINTS
# -----------------

def get_or_create_default_user():
    user = User.query.filter_by(username='guest_user').first()
    if not user:
        new_encoded_id = recommender.num_users if recommender.mappings else 0
        import random
        if recommender.mappings:
            new_encoded_id = recommender.num_users + random.randint(1, 9000)
            
        user = User(username='guest_user', encoded_id=new_encoded_id)
        user.password_hash = 'default_password'
        db.session.add(user)
        db.session.commit()
    return user

@app.route('/api/movies', methods=['GET'])
def get_movies():
    search = request.args.get('search', '')
    if search:
        movies = Movie.query.filter(Movie.title.ilike(f'%{search}%')).limit(20).all()
    else:
        # Just return some popular/random movies initially
        movies = Movie.query.limit(20).all()

    movies_list = [{"id": m.movie_id, "encoded": m.movie_encoded, "title": m.title, "genres": m.genres} for m in movies]
    return jsonify(movies_list), 200

@app.route('/api/ratings', methods=['POST', 'GET'])
def manage_ratings():
    user = get_or_create_default_user()
    user_id = user.id

    if request.method == 'POST':
        data = request.get_json()
        movie_encoded = data.get('movie_encoded')
        rating_value = float(data.get('rating'))

        rating = Rating.query.filter_by(user_id=user_id, movie_id=movie_encoded).first()
        if rating:
            rating.rating = rating_value
        else:
            rating = Rating(user_id=user_id, movie_id=movie_encoded, rating=rating_value)
            db.session.add(rating)
        db.session.commit()

        return jsonify({"msg": "Rated successfully"}), 200

    else:
        ratings = Rating.query.filter_by(user_id=user_id).all()
        ratings_list = [{"movie_encoded": r.movie_id, "rating": r.rating} for r in ratings]
        return jsonify(ratings_list), 200

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    user = get_or_create_default_user()
    user_id = user.id
    
    if not recommender.mappings:
        return jsonify({"msg": "Model not ready"}), 500

    # Get all movies the user has already rated
    rated_movies = [r.movie_id for r in Rating.query.filter_by(user_id=user_id).all()]
    
    # Get a pool of unrated movies. To save compute in a real app, you wouldn't query ALL movies.
    # For this college project, we can query a sample of 1000 distinct movies.
    all_movies = Movie.query.limit(2000).all()
    unrated_encoded = [m.movie_encoded for m in all_movies if m.movie_encoded not in set(rated_movies)]
    
    # Get recommendations from the ANN model
    top_recommendations = recommender.get_recommendations(user.encoded_id, unrated_encoded, top_k=10)
    
    # Map back to movie details
    results = []
    for m_encoded, pred_rating in top_recommendations:
        movie = Movie.query.filter_by(movie_encoded=int(m_encoded)).first()
        if movie:
            # Re-scale from [0,1] normalization used in training back to 1-5 scale format for UI
            formatted_rating = pred_rating * 4.5 + 0.5 
            results.append({
                "id": movie.movie_id,
                "encoded": movie.movie_encoded,
                "title": movie.title,
                "genres": movie.genres,
                "predicted_rating": round(formatted_rating, 1)
            })
            
    return jsonify(results), 200

if __name__ == '__main__':
    # Initialize the DB if it hasn't been initialized
    if not os.path.exists(os.path.join(BASE_DIR, 'app.db')):
        from init_db import init_db
        init_db()
        
    app.run(host='0.0.0.0', port=5001, debug=True)
