#!/bin/bash
# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found! Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

echo "Starting Backend Flask Server..."
# Run the flask backend on port 5000
python backend/app.py
