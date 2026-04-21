import sys
import os

# Add the backend directory to the Python path so internal imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the Flask application instance from our backend directory
from backend.app import app

# Expose it so hosting services (like Vercel, Render, Heroku) can find the entrypoint
if __name__ == "__main__":
    app.run()
