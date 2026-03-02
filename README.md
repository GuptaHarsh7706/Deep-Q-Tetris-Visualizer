# Tetris DQN Agent

A complete Deep Q-Network (DQN) implementation for playing Tetris, built from scratch in Python using PyTorch, Pygame, and Matplotlib.

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch>=2.0.0 pygame>=2.4.0 numpy>=1.24.0 matplotlib>=3.7.0
```

## Usage

### Basic Training (Headless Mode)
Train the agent without visualization (faster):
```bash
python tetris_dqn.py --episodes 2000
```

### Training with Visualization
Train with game window and visualizations:
```bash
python tetris_dqn.py --episodes 2000 --render
```

### Force Headless Mode
Even if `--render` is specified, force headless:
```bash
python tetris_dqn.py --episodes 2000 --render --headless
```

## Command Line Arguments

- `--episodes N`: Number of training episodes (default: 2000)
- `--render`: Enable visualization windows (game, network architecture, network activity)
- `--headless`: Force headless mode (disable all visualizations)

## Output Files

The training process creates:

- **`checkpoints/`**: Model checkpoints saved every 500 episodes
  - Format: `ep_{episode_number}.pt`
  - Contains: Q-network weights, target network weights, optimizer state, epsilon, training step count

- **`results/`**: Training metrics and visualizations
  - `training_metrics.png`: Final training plots (6 subplots)
  - `architecture_ep_{N}.png`: Network architecture visualizations (saved periodically)
  - `activity_step_{N}.png`: Network activity visualizations (saved periodically)

## Training Progress

During training, you'll see progress updates every 10 episodes:
```
Episode 10/2000 | Score: 100.0 | Lines: 1 | Epsilon: 0.9950 | Buffer: 64/20000 | Avg (10): Score=50.0, Lines=0.5
```

## Expected Performance

- **After ~2,000 episodes**: Agent should consistently clear 1-3 lines per episode
- **After ~10,000 episodes**: Average lines cleared should trend above 5 per episode

## Keyboard Shortcuts

- **Ctrl+C**: Gracefully save checkpoint and exit

## System Requirements

- Python 3.8+
- GPU support (CUDA/MPS) is automatically detected and used if available
- For visualization: Display capable of running Pygame windows

## How It Works (Deep Q-Network)

The agent learns to play Tetris using Reinforcement Learning, specifically a Deep Q-Network (DQN). Instead of hardcoded rules, the agent learns through trial and error—being rewarded for good moves and penalized for bad ones.

### State Representation (Input)
The neural network "sees" the game through a 27-dimensional feature vector:
- **10 normalized column heights**: How tall each column is.
- **1 normalized hole count**: Number of empty spaces buried under blocks.
- **1 normalized bumpiness**: The variation in height between adjacent columns.
- **1 normalized lines cleared**: Total lines cleared so far in the game.
- **7-value one-hot vector**: The current falling piece type.
- **7-value one-hot vector**: The next piece type.

### Neural Network Architecture
The brain of the agent is a multi-layer perceptron (MLP) built with PyTorch:
- **Input Layer**: 27 neurons (the state vector)
- **Hidden Layer 1**: 128 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)
- **Output Layer**: 5 neurons (representing the Q-value/expected future reward for each possible action)

### Actions (Output)
The agent can choose 1 of 5 actions at any given step:
1. Move Left
2. Move Right
3. Rotate
4. Soft Drop
5. Hard Drop

### Reward System
The agent optimizes for the highest expected reward based on this system:
- **Line Clear**: `+1.0` * (lines_cleared^2)
- **Adding a Hole**: `-0.5` per new hole
- **Increasing Height**: `-0.3` per unit of height increase
- **Increasing Bumpiness**: `-0.1` per unit of bumpiness increase
- **Game Over**: `-1.0`
- **Survival**: `+0.001` per step

### Training Process
1. **Experience Replay**: The agent stores its game experiences (State, Action, Reward, Next State, Game Over flag) in a memory buffer.
2. **Epsilon-Greedy Exploration**: Early in training, the agent takes random actions (exploration) to learn how the game works. Over time, it relies more on its neural network (exploitation).
3. **Q-Learning Update**: Periodically, the agent samples a random batch of past experiences and trains the neural network to better predict the long-term reward of its actions.

## Notes

- The agent uses epsilon-greedy exploration (starts at 1.0, decays to 0.01)
- Training uses experience replay with a buffer of 20,000 experiences
- Target network updates every 200 gradient steps
- Visualization windows update every 50 episodes
