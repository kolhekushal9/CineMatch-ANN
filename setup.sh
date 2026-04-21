#!/bin/bash
echo "Setting up the Movie Recommendation System..."

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies (TensorFlow, Flask, etc.)..."
pip install -r requirements.txt

echo "Dependencies installed successfully!"

echo "Downloading dataset and training the Artificial Neural Network..."
# We will run the python training script
python model_training/train_model.py

echo "Model trained successfully!"
echo "Initializing the database..."
python backend/init_db.py

echo "Setup Complete! You can now start the application with ./run.sh"
