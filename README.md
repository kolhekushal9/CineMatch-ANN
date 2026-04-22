# CineMatch - ANN Movie Recommendation System

This project is a complete full-stack Movie Recommendation engine utilizing an **Artificial Neural Network (ANN)** based on the deeply acclaimed Neural Collaborative Filtering (NCF) architecture.

## 🧠 Artificial Neural Network Concepts Used

This application doesn't just recommend random popular movies; it builds mathematical profiles for every user and every movie using Deep Learning.

### 1. Neural Collaborative Filtering (NCF)
Unlike traditional matrix factorization, this application uses a Deep Neural Network to predict a user's rating for a movie. It works by treating the User ID and Movie ID as continuous vectors in a latent space and learning the non-linear interaction between them.

### 2. Embedding Layers
The input to our network consists of discrete User IDs and Movie IDs. These are fed directly into **TensorFlow Embedding Layers**:
*   The **User Embedding** maps an individual user to a dense 50-dimensional vector representing their personal preferences (e.g., love for action, disdain for long movies).
*   The **Movie Embedding** maps a movie to a dense 50-dimensional vector representing its latent features (e.g., dark, sci-fi, fast-paced).

### 3. Dense Hidden Layers (The "Deep" in Deep Learning)
The user and movie vectors are concatenated to form a 100-dimensional sequence, which is then passed through several fully connected Dense layers:
*   **Layer 1:** 128 Neurons (detects broad combinations)
*   **Layer 2:** 64 Neurons (extracts refined non-linear patterns)
*   **Layer 3:** 32 Neurons (final feature extraction)

### 4. Activation Functions (ReLU)
We use the **ReLU (Rectified Linear Unit)** activation function `f(x) = max(0, x)`. This function is critical for allowing the network to solve highly complex, non-linear problems without encountering the vanishing gradient problem.

### 5. Pure Linear Algebra Inference (Vercel Cloud Compatibility)
To overcome strict cloud hosting data limits, the massive TensorFlow model (`.h5`) was mathematically extracted into purely raw NumPy representations (`.npz`). When a user asks for recommendations, the backend runs the literal forward-propagation algebraic equations:
$$Z = ReLU(W \cdot X + b)$$
This is extremely fast and entirely drops the massive TensorFlow dependency overhead.

## 🚀 Setup Instructions

1. `bash setup.sh` - Installs packages, downloads MovieLens, trains the ANN over thousands of epochs, exports the weights, and builds the database.
2. `bash run.sh` - Starts the Flask Server. Open `localhost:5001`.
