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

## Notes

- The agent uses epsilon-greedy exploration (starts at 1.0, decays to 0.01)
- Training uses experience replay with a buffer of 20,000 experiences
- Target network updates every 200 gradient steps
- Visualization windows update every 50 episodes
