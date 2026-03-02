# Deep Q-Network Tetris Visualizer

A complete full-stack web application that visualizes a Deep Q-Network (DQN) artificial intelligence learning to play Tetris in real-time.(The backend does take a bit time to get active since render turns off the instance after 15 minutes of inactivity)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Next.js](https://img.shields.io/badge/Next.js-15.1-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)

## 🌐 Live Demo

- **Frontend Application**: [https://deep-q-tetris-visualizer.vercel.app](https://deep-q-tetris-visualizer.vercel.app) 
- **Backend API Server**: `https://deep-q-tetris-visualizer.onrender.com`

**Note on the Live Demo:** The backend is hosted on Render's free tier. If the application has not been used recently, the backend server will spin down. It may take up to 60 seconds for the server to spin back up on your first visit. If it initially says "Disconnected", simply wait a moment and refresh the page.

## 🏗️ Architecture

The project consists of three main components:

1. **Python / PyTorch RL Engine**: A custom continuous learning Tetris engine and DQN memory buffer implemented entirely from scratch in PyTorch.
2. **FastAPI Backend (`server.py`)**: A high-performance asynchronous API that runs the training loop on a background thread and streams the live neural network weights, activations, and game state via WebSockets.
3. **Next.js Frontend (`frontend/`)**: A React 19 application leveraging the Canvas API to visualize the incoming WebSocket data, rendering both the classic Tetris board and a multi-layer neural network firing in real-time.

## 🚀 Local Development Setup

To run this application locally, you will need to start both the Python backend and the Next.js frontend.

### 1. Backend Setup (FastAPI & PyTorch)

Make sure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/GuptaHarsh7706/Deep-Q-Tetris-Visualizer.git
cd Deep-Q-Tetris-Visualizer

# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
# Server will run at http://0.0.0.0:8000 and the WebSocket will listen at ws://0.0.0.0:8000/ws/train
```

### 2. Frontend Setup (Next.js & React)

Open a **new terminal window** and navigate to the frontend directory. Make sure you have Node.js installed.

```bash
cd frontend

# Install Node dependencies
npm install

# (Optional) Tell the frontend to connect to your local backend
# By default, it will fall back to connect to the live Render backend
# Create a .env.local file in the frontend/ folder with:
# NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/train

# Start the development server
npm run dev
# The website will be available at http://localhost:3000
```

---

## 🧠 How the AI Works (Deep Q-Network)

The agent learns to play Tetris entirely via Reinforcement Learning. Instead of hardcoded rules, the agent learns through trial and error over thousands of games—being rewarded for strategic moves and penalized for detrimental ones.

### State Representation (Input)
The neural network "sees" the game not as pixels, but through a condensed 27-dimensional mathematical feature vector to significantly speed up training:
- **10 normalized column heights**: How tall each of the 10 columns are.
- **1 normalized hole count**: The total number of empty spaces buried underneath solid blocks.
- **1 normalized bumpiness**: The variation in height between adjacent columns (rougher surfaces are harder to build on).
- **1 normalized lines cleared**: Total number of lines cleared so far in the current game.
- **7-value one-hot vector**: Which of the 7 pieces (`I, J, L, O, S, T, Z`) is currently falling.
- **7-value one-hot vector**: Which of the 7 pieces is coming up next.

### Neural Network Architecture / The Brain
The brain of the agent is a multi-layer perceptron (MLP) built from scratch via PyTorch.
- **Input Layer**: 27 neurons (representing the state vector above)
- **Hidden Layer 1**: 128 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)
- **Output Layer**: 5 neurons (representing the unactivated Q-value / expected future reward for taking each of the 5 possible actions)

### Actions (Output)
At any given millisecond step, the agent can blindly choose 1 of 5 actions:
1. Move Left
2. Move Right
3. Rotate
4. Soft Drop
5. Hard Drop

### Reward System
The agent optimizes strictly for the highest long-term expected reward based heavily on this custom penalty system:
- **Line Clear**: `+1.0` * (lines_cleared^2) (Clearing 4 lines at once via a "Tetris" yields exponentially higher reward than a single line).
- **Adding a Hole**: `-0.5` per new hole created.
- **Increasing Height**: `-0.3` per unit of height increase.
- **Increasing Bumpiness**: `-0.1` per unit of bumpiness increase.
- **Game Over**: `-1.0` instant punishment.
- **Survival**: `+0.001` per step slightly encouraging stalling against Game Overs.

### Training Process Loop
1. **Experience Replay**: The agent stores its game experiences (State, Action, Reward, Next State, Game Over flag) in a memory buffer array sized for 20,000 moves.
2. **Epsilon-Greedy Exploration**: Early in training, the agent intentionally takes random actions (exploration) 100% of the time to learn the environment physics. Over time (as epsilon decays down to 1%), it transitions almost entirely to relying on its neural network's predictions (exploitation).
3. **Q-Learning Update**: Periodically, the PyTorch agent randomly samples a batch of past experiences and runs mathematical gradient descent on the neural network via an Adam Optimizer to teach it to better predict long-term rewards.

---

## 🖥️ Legacy Local Training

If you are a machine learning researcher who does not want to run the web application to visualize the network graph, but instead just wants to run headless continuous training on your local GPU with standard Pygame rendering, you can bypass FastAPI and run the engine directly:

```bash
# Basic Training (Headless Mode) - Extremely fast
python tetris_dqn.py --episodes 5000

# Training with classical Pygame Visualization
python tetris_dqn.py --episodes 5000 --render
```

### Legacy Output Files

The local script training process automatically creates:
- **`checkpoints/`**: PyTorch `.pt` model state dictionaries saved every 500 episodes containing the precise weight matrices, optimizer state, and epsilon tracking.
- **`results/`**: Training metrics and visualization plots, such as loss-graphs over time (`training_metrics.png`).
