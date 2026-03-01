
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path
from typing import Tuple, Optional, List
import signal
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


SEED = 42
INPUT_SIZE = 27
HIDDEN_SIZE = 128
OUTPUT_SIZE = 5
LEARNING_RATE = 3e-4
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 64
TRAIN_EVERY = 4
TARGET_UPDATE = 200

ROWS = 20
COLS = 10
CELL_SIZE = 30
BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
SIDEBAR_WIDTH = 200

ARCH_WINDOW_WIDTH = 800
ARCH_WINDOW_HEIGHT = 800
WINDOW_WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH + ARCH_WINDOW_WIDTH
WINDOW_HEIGHT = max(BOARD_HEIGHT, ARCH_WINDOW_HEIGHT)


def connection_color(weight: float, activation: float, was_correct: bool) -> Tuple[int, int, int]:

    if abs(activation) < 0.01 or abs(weight) < 0.01:
        return (15, 15, 15)  # dormant

    magnitude = math.tanh(abs(weight) * abs(activation) * 4.0)
    brightness = int(magnitude * 255)
    
    if brightness < 30:
        return (15, 15, 15)

    if weight > 0 and was_correct:
        return (0, min(255, brightness + 80), int(brightness * 0.3))
    elif weight > 0 and not was_correct:
        return (min(255, int(brightness * 0.7) + 50), min(255, brightness + 50), 0)
    elif weight < 0 and was_correct:
        return (min(255, brightness + 60), int(brightness * 0.5), 0)
    else:
        return (min(255, brightness + 80), 0, int(brightness * 0.15))

VIZ_UPDATE_EVERY = 50
RENDER_EVERY = 1

REWARD_LINE_CLEAR = 1.0
REWARD_HOLE_PENALTY = -0.5
REWARD_HEIGHT_PENALTY = -0.3
REWARD_BUMPINESS_PENALTY = -0.1
REWARD_GAME_OVER = -1.0
REWARD_SURVIVAL = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


TETROMINOES = {
    'I': [
        np.array([[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], dtype=int),
        np.array([[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]], dtype=int),
        np.array([[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]], dtype=int),
        np.array([[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]], dtype=int),
    ],
    'O': [
        np.array([[1,1], [1,1]], dtype=int),  # 2x2 matrix
        np.array([[1,1], [1,1]], dtype=int),
        np.array([[1,1], [1,1]], dtype=int),
        np.array([[1,1], [1,1]], dtype=int),
    ],
    'T': [
        np.array([[0,1,0], [1,1,1], [0,0,0]], dtype=int),
        np.array([[0,1,0], [0,1,1], [0,1,0]], dtype=int),
        np.array([[0,0,0], [1,1,1], [0,1,0]], dtype=int),
        np.array([[0,1,0], [1,1,0], [0,1,0]], dtype=int),
    ],
    'S': [
        np.array([[0,1,1], [1,1,0], [0,0,0]], dtype=int),
        np.array([[0,1,0], [0,1,1], [0,0,1]], dtype=int),
        np.array([[0,0,0], [0,1,1], [1,1,0]], dtype=int),
        np.array([[1,0,0], [1,1,0], [0,1,0]], dtype=int),
    ],
    'Z': [
        np.array([[1,1,0], [0,1,1], [0,0,0]], dtype=int),
        np.array([[0,0,1], [0,1,1], [0,1,0]], dtype=int),
        np.array([[0,0,0], [1,1,0], [0,1,1]], dtype=int),
        np.array([[0,1,0], [1,1,0], [1,0,0]], dtype=int),
    ],
    'J': [
        np.array([[1,0,0], [1,1,1], [0,0,0]], dtype=int),
        np.array([[0,1,1], [0,1,0], [0,1,0]], dtype=int),
        np.array([[0,0,0], [1,1,1], [0,0,1]], dtype=int),
        np.array([[0,1,0], [0,1,0], [1,1,0]], dtype=int),
    ],
    'L': [
        np.array([[0,0,1], [1,1,1], [0,0,0]], dtype=int),
        np.array([[0,1,0], [0,1,0], [0,1,1]], dtype=int),
        np.array([[0,0,0], [1,1,1], [1,0,0]], dtype=int),
        np.array([[1,1,0], [0,1,0], [0,1,0]], dtype=int),
    ],
}

PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
PIECE_COLORS = [
    (0, 255, 255),    # I - Cyan
    (255, 255, 0),    # O - Yellow
    (128, 0, 128),    # T - Purple
    (0, 255, 0),      # S - Green
    (255, 0, 0),      # Z - Red
    (0, 0, 255),      # J - Blue
    (255, 165, 0),    # L - Orange
]


class TetrisEngine:
        
    def __init__(self):
     
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_piece_type = None
        self.current_piece_rotation = 0
        self.current_piece_col = 0
        self.current_piece_row = 0
        self.next_piece_type = None
        self.lines_cleared = 0
        self.score = 0
        self.done = False
        self.step_count = 0
        self._prev_heights = None
        self._prev_holes = None
        self._prev_bumpiness = None
        
    def reset(self) -> np.ndarray:
     
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_piece_type = random.choice(PIECE_NAMES)
        self.next_piece_type = random.choice(PIECE_NAMES)
        self.current_piece_rotation = 0
        self.current_piece_col = COLS // 2 - 1
        self.current_piece_row = 0
        self.lines_cleared = 0
        self.score = 0
        self.done = False
        self.step_count = 0
        self._prev_heights = np.zeros(COLS)
        self._prev_holes = 0
        self._prev_bumpiness = 0
        
        return self.get_state()
    
    def _get_piece_matrix(self, piece_type: str, rotation: int) -> np.ndarray:
             return TETROMINOES[piece_type][rotation % 4]
    
    def _can_place(self, piece_type: str, rotation: int, col: int, row: int) -> bool:
    
        piece_matrix = self._get_piece_matrix(piece_type, rotation)
        h, w = piece_matrix.shape
        
        for py in range(h):
            for px in range(w):
                if piece_matrix[py, px] != 0:
                    board_y = row + py
                    board_x = col + px
                    
                    if board_x < 0 or board_x >= COLS or board_y >= ROWS:
                        return False
                    
                    if board_y >= 0 and self.board[board_y, board_x] != 0:
                        return False
        
        return True
    
    def _lock_piece(self):
   
        piece_matrix = self._get_piece_matrix(self.current_piece_type, self.current_piece_rotation)
        piece_color = PIECE_NAMES.index(self.current_piece_type) + 1
        h, w = piece_matrix.shape
        
        for py in range(h):
            for px in range(w):
                if piece_matrix[py, px] != 0:
                    board_y = self.current_piece_row + py
                    board_x = self.current_piece_col + px
                    
                    if 0 <= board_y < ROWS and 0 <= board_x < COLS:
                        self.board[board_y, board_x] = piece_color
    
    def _clear_lines(self) -> int:
        
        lines_to_clear = []
        
        for row in range(ROWS):
            if np.all(self.board[row, :] != 0):
                lines_to_clear.append(row)
        
        if lines_to_clear:
            for row in reversed(lines_to_clear):
                self.board = np.delete(self.board, row, axis=0)
            
            new_rows = np.zeros((len(lines_to_clear), COLS), dtype=int)
            self.board = np.vstack([new_rows, self.board])
        
        return len(lines_to_clear)
    
    def _get_column_heights(self) -> np.ndarray:
        heights = np.zeros(COLS, dtype=int)
        
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row, col] != 0:
                    heights[col] = ROWS - row
                    break
        
        return heights
    
    def _count_holes(self) -> int:
        holes = 0
        
        for col in range(COLS):
            found_filled = False
            for row in range(ROWS):
                if self.board[row, col] != 0:
                    found_filled = True
                elif found_filled and self.board[row, col] == 0:
                    holes += 1
        
        return holes
    
    def _get_bumpiness(self) -> float:
        heights = self._get_column_heights()
        if len(heights) < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(heights))))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            return self.get_state(), 0.0, True
        
        reward = REWARD_SURVIVAL  # Survival bonus
        
        prev_heights = self._get_column_heights()
        prev_holes = self._count_holes()
        prev_bumpiness = self._get_bumpiness()
        
        if action == 0:  # Move left
            if self._can_place(self.current_piece_type, self.current_piece_rotation,
                             self.current_piece_col - 1, self.current_piece_row):
                self.current_piece_col -= 1
        elif action == 1:  # Move right
            if self._can_place(self.current_piece_type, self.current_piece_rotation,
                             self.current_piece_col + 1, self.current_piece_row):
                self.current_piece_col += 1
        elif action == 2:  # Rotate
            new_rotation = (self.current_piece_rotation + 1) % 4
            if self._can_place(self.current_piece_type, new_rotation,
                             self.current_piece_col, self.current_piece_row):
                self.current_piece_rotation = new_rotation
        elif action == 3:  # Soft drop
            if self._can_place(self.current_piece_type, self.current_piece_rotation,
                             self.current_piece_col, self.current_piece_row + 1):
                self.current_piece_row += 1
        elif action == 4:  # Hard drop
            drop_row = self.current_piece_row
            while self._can_place(self.current_piece_type, self.current_piece_rotation,
                                self.current_piece_col, drop_row + 1):
                drop_row += 1
            self.current_piece_row = drop_row
        
        if self._can_place(self.current_piece_type, self.current_piece_rotation,
                         self.current_piece_col, self.current_piece_row + 1):
            self.current_piece_row += 1
        else:
            self._lock_piece()
            
            lines_cleared = self._clear_lines()
            if lines_cleared > 0:
                reward += REWARD_LINE_CLEAR * (lines_cleared ** 2)
                self.lines_cleared += lines_cleared
                self.score += lines_cleared * 100
            
            self.current_piece_type = self.next_piece_type
            self.next_piece_type = random.choice(PIECE_NAMES)
            self.current_piece_rotation = 0
            self.current_piece_col = COLS // 2 - 1
            self.current_piece_row = 0
            
            if not self._can_place(self.current_piece_type, self.current_piece_rotation,
                                 self.current_piece_col, self.current_piece_row):
                self.done = True
                reward += REWARD_GAME_OVER
        
        new_heights = self._get_column_heights()
        new_holes = self._count_holes()
        new_bumpiness = self._get_bumpiness()
        
        if new_holes > prev_holes:
            reward += REWARD_HOLE_PENALTY * (new_holes - prev_holes)
        
        height_increase = np.sum(new_heights) - np.sum(prev_heights)
        if height_increase > 0:
            reward += REWARD_HEIGHT_PENALTY * height_increase
        
        if new_bumpiness > prev_bumpiness:
            reward += REWARD_BUMPINESS_PENALTY * (new_bumpiness - prev_bumpiness)
        
        self._prev_heights = new_heights
        self._prev_holes = new_holes
        self._prev_bumpiness = new_bumpiness
        self.step_count += 1
        
        return self.get_state(), reward, self.done
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state as a feature vector.
        
        Returns:
            np.ndarray: State vector of shape (27,)
                - 10 column heights (normalized)
                - 1 hole count (normalized)
                - 1 bumpiness (normalized)
                - 1 lines cleared (normalized)
                - 7 current piece one-hot
                - 7 next piece one-hot
        """
        heights = self._get_column_heights()
        holes = self._count_holes()
        bumpiness = self._get_bumpiness()
        
        normalized_heights = heights.astype(np.float32) / ROWS
        normalized_holes = np.array([holes], dtype=np.float32) / (ROWS * COLS)
        normalized_bumpiness = np.array([bumpiness], dtype=np.float32) / (ROWS * COLS)
        normalized_lines = np.array([self.lines_cleared], dtype=np.float32) / 100.0  # Cap at 100
        
        current_piece_onehot = np.zeros(7, dtype=np.float32)
        if self.current_piece_type:
            current_piece_onehot[PIECE_NAMES.index(self.current_piece_type)] = 1.0
        
        next_piece_onehot = np.zeros(7, dtype=np.float32)
        if self.next_piece_type:
            next_piece_onehot[PIECE_NAMES.index(self.next_piece_type)] = 1.0
        
        state = np.concatenate([
            normalized_heights,
            normalized_holes,
            normalized_bumpiness,
            normalized_lines,
            current_piece_onehot,
            next_piece_onehot
        ])
        
        return state.astype(np.float32)
    
    def get_current_piece_info(self) -> Tuple[str, int, int, int]:
        """Get current piece information for rendering."""
        return (self.current_piece_type, self.current_piece_rotation,
                self.current_piece_col, self.current_piece_row)
    
    def get_next_piece_info(self) -> Tuple[str, int]:
        """Get next piece information for rendering."""
        return (self.next_piece_type, 0)  # Next piece always at rotation 0



def get_activations(model, state_np, device):
    with torch.no_grad():
        x = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
        h1_pre = model.fc1(x)
        h1 = torch.relu(h1_pre)
        h2_pre = model.fc2(h1)
        h2 = torch.relu(h2_pre)
        out = model.fc3(h2)
    return [
        state_np.flatten(),
        h1.squeeze().cpu().numpy(),
        h2.squeeze().cpu().numpy(),
        out.squeeze().cpu().numpy(),
    ]

def compute_neuron_positions(layer_sizes, xs, win_h, padding_y, hidden_group_size=4):
    neuron_pos = []
    display_sizes = []
    for l, (n, x) in enumerate(zip(layer_sizes, xs)):
        if l in (1, 2) and n > 32:
            display_n = n // hidden_group_size
        else:
            display_n = n
        
        display_sizes.append(display_n)
        positions = []
        for i in range(display_n):
            if display_n > 1:
                y = padding_y + i * (win_h - 2 * padding_y) // (display_n - 1)
            else:
                y = win_h // 2
            positions.append((x, y))
        neuron_pos.append(positions)
    return neuron_pos, display_sizes

def real_to_display(real_idx, group_size=4):
    return real_idx // group_size

def draw_all_neurons(surface, neuron_pos, activations, chosen_action, was_correct):
    LAYER_GLOW_COLORS = [
        (100, 180, 255),
        (180, 100, 255),
        (100, 255, 180),
        (255, 220, 80),
    ]

    for l, positions in enumerate(neuron_pos):
        for i, (x, y) in enumerate(positions):
            if l in (1, 2):
                start_idx = i * 4
                end_idx = min(start_idx + 4, len(activations[l]))
                act = float(np.mean(activations[l][start_idx:end_idx])) if start_idx < len(activations[l]) else 0.0
            else:
                act = float(activations[l][i]) if i < len(activations[l]) else 0.0
            
            is_output = (l == len(neuron_pos) - 1)
            is_chosen = is_output and (i == chosen_action)
            
            import math
            core_brightness = min(255, max(80, int(math.tanh(abs(act)) * 255) + 80))
            
            if is_chosen:
                SIZE = 14
                fill_color = (0, 255, 80) if was_correct else (255, 40, 40)
                pygame.draw.rect(surface, fill_color,
                                 (x - SIZE//2, y - SIZE//2, SIZE, SIZE))
                pygame.draw.rect(surface, (255, 255, 255),
                                 (x - SIZE//2 - 2, y - SIZE//2 - 2, SIZE + 4, SIZE + 4), 1)
                pygame.draw.rect(surface, fill_color,
                                 (x - SIZE//2 - 4, y - SIZE//2 - 4, SIZE + 8, SIZE + 8), 1)
            
            elif is_output:
                SIZE = 10
                pygame.draw.rect(surface, (core_brightness, core_brightness, core_brightness),
                                 (x - SIZE//2, y - SIZE//2, SIZE, SIZE))
                pygame.draw.rect(surface, LAYER_GLOW_COLORS[l],
                                 (x - SIZE//2 - 2, y - SIZE//2 - 2, SIZE + 4, SIZE + 4), 1)
            
            else:
                SIZE = 8
                fill = (core_brightness, core_brightness, core_brightness)
                pygame.draw.rect(surface, fill,
                                 (x - SIZE//2, y - SIZE//2, SIZE, SIZE))
                
                glow = LAYER_GLOW_COLORS[l]
                glow_intensity = int(math.tanh(abs(act)) * 255)
                if glow_intensity > 30:
                    scaled_glow = tuple(min(255, int(c * glow_intensity / 255)) for c in glow)
                    pygame.draw.rect(surface, scaled_glow,
                                     (x - SIZE//2 - 2, y - SIZE//2 - 2, SIZE + 4, SIZE + 4), 1)

class QNetwork(nn.Module):
    """
    Deep Q-Network with architecture: 27 → 128 → 128 → 5
    """
    
    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE,
                 output_size: int = OUTPUT_SIZE):
        
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output
        return x



Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
   
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
       
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
       
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        
        return len(self.buffer)
    
    def recency_histogram(self) -> np.ndarray:
       
        if len(self.buffer) == 0:
            return np.zeros(10, dtype=int)
        
        cohort_size = max(1, len(self.buffer) // 10)
        counts = np.zeros(10, dtype=int)
        
        for i in range(10):
            start_idx = i * cohort_size
            end_idx = min((i + 1) * cohort_size, len(self.buffer))
            counts[i] = end_idx - start_idx
        
        return counts



class TetrisDQNTrainer:
   
    def __init__(self, render: bool = False):
        
        self.render = render and PYGAME_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        self.buffer = ReplayBuffer()
        
        self.epsilon = EPSILON_START
        self.total_steps = 0
        self.grad_step_count = 0
        self.was_correct_last = False
        self.last_action = 0
        self.last_reward = 0.0
        self.correct_streak = 0
        self.wrong_streak = 0
        
        self.episode_scores = []
        self.episode_lines = []
        self.episode_lengths = []
        self.loss_history = []
        self.epsilon_history = []
        self.buffer_fill_history = []
        
        self.game_screen = None
        self.arch_screen = None
        self.activity_screen = None
        self.force_render = False
        self.arch_window_created = False
        self.activity_window_created = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        
        Path("results").mkdir(exist_ok=True)
        
        if self.render:
            self._init_visualization()
    
    def _signal_handler(self, signum, frame):
        
        print("\nReceived interrupt signal. Exiting after current step...")
        self.force_render = True
    
    def _init_visualization(self):
        if not PYGAME_AVAILABLE:
            return
        
        pygame.init()
        
        self.game_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris DQN - Game")
        
        self.arch_surface = pygame.Surface((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
        self.activity_surface = pygame.Surface((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
        
        self.arch_screen = None
        self.activity_screen = None
        self.arch_window_created = False
        self.activity_window_created = False
        self.show_arch_window = False
        self.show_activity_window = False
        
        
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 12)
        
    
    def select_action(self, state: np.ndarray) -> int:
       
        if random.random() < self.epsilon:
            return random.randint(0, OUTPUT_SIZE - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        
        if len(self.buffer) < BATCH_SIZE:
            return
        
        experiences = self.buffer.sample(BATCH_SIZE)
        
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        predicted_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = F.mse_loss(predicted_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        self.grad_step_count += 1
        if self.grad_step_count % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int):
        
        env = TetrisEngine()
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Render: {self.render}")
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                
                next_state, reward, done = env.step(action)
                episode_reward += reward
                episode_length += 1

                self.last_action = action
                self.last_reward = reward
                self.was_correct_last = reward > 0.0
                if self.was_correct_last:
                    self.correct_streak += 1
                    self.wrong_streak = 0
                else:
                    self.wrong_streak += 1
                    self.correct_streak = 0
                
                self.buffer.push(state, action, reward, next_state, done)
                
                if self.total_steps % TRAIN_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
                    self.train_step()
                
                state = next_state
                self.total_steps += 1
                
                if self.render and (self.total_steps % RENDER_EVERY == 0 or self.force_render):
                    self._render_game(env, episode)
                    self._render_activity(state, action)
                    self.force_render = False
            
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
            
            self.episode_scores.append(env.score)
            self.episode_lines.append(env.lines_cleared)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(self.epsilon)
            self.buffer_fill_history.append(len(self.buffer) / REPLAY_BUFFER_SIZE * 100)
            
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(self.episode_scores[-10:])
                avg_lines = np.mean(self.episode_lines[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Score: {env.score:.1f} | Lines: {env.lines_cleared} | "
                      f"Epsilon: {self.epsilon:.4f} | Buffer: {len(self.buffer)}/{REPLAY_BUFFER_SIZE} | "
                      f"Avg (10): Score={avg_score:.1f}, Lines={avg_lines:.1f}")
            
            if self.render and (episode + 1) % VIZ_UPDATE_EVERY == 0:
                self._render_architecture(episode + 1)
        
        print("Training complete!")
    
    def _render_game(self, env: TetrisEngine, episode: int):
        
        if not self.render or self.game_screen is None:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        
        self.game_screen.fill((20, 20, 20))
        
        for row in range(ROWS):
            for col in range(COLS):
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                
                cell_value = env.board[row, col]
                if cell_value > 0:
                    color = PIECE_COLORS[cell_value - 1]
                    pygame.draw.rect(self.game_screen, color, (x, y, CELL_SIZE - 1, CELL_SIZE - 1))
                else:
                    pygame.draw.rect(self.game_screen, (40, 40, 40), (x, y, CELL_SIZE - 1, CELL_SIZE - 1))
                
                pygame.draw.rect(self.game_screen, (60, 60, 60), (x, y, CELL_SIZE, CELL_SIZE), 1)
        
        piece_type, rotation, col, row = env.get_current_piece_info()
        if piece_type:
            piece_matrix = TETROMINOES[piece_type][rotation]
            piece_color_idx = PIECE_NAMES.index(piece_type)
            piece_color = tuple(min(255, c + 50) for c in PIECE_COLORS[piece_color_idx])  # Brighter
            
            for py in range(piece_matrix.shape[0]):
                for px in range(piece_matrix.shape[1]):
                    if piece_matrix[py, px] != 0:
                        x = (col + px) * CELL_SIZE
                        y = (row + py) * CELL_SIZE
                        if 0 <= y < BOARD_HEIGHT:
                            pygame.draw.rect(self.game_screen, piece_color,
                                           (x, y, CELL_SIZE - 1, CELL_SIZE - 1))
        
        sidebar_x = BOARD_WIDTH
        y_offset = 20
        
        next_type, _ = env.get_next_piece_info()
        text = self.font.render("Next Piece:", True, (255, 255, 255))
        self.game_screen.blit(text, (sidebar_x + 10, y_offset))
        y_offset += 40
        
        if next_type:
            next_matrix = TETROMINOES[next_type][0]
            next_color = PIECE_COLORS[PIECE_NAMES.index(next_type)]
            preview_size = 20
            start_x = sidebar_x + 20
            start_y = y_offset
            
            for py in range(next_matrix.shape[0]):
                for px in range(next_matrix.shape[1]):
                    if next_matrix[py, px] != 0:
                        x = start_x + px * preview_size
                        y = start_y + py * preview_size
                        pygame.draw.rect(self.game_screen, next_color,
                                       (x, y, preview_size - 1, preview_size - 1))
        
        y_offset += 100
        
        stats = [
            f"Score: {env.score}",
            f"Lines: {env.lines_cleared}",
            f"Episode: {episode + 1}",
            f"Epsilon: {self.epsilon:.4f}",
            f"Steps: {self.total_steps}",
            f"Buffer: {len(self.buffer)}/{REPLAY_BUFFER_SIZE}",
        ]
        
        for stat in stats:
            text = self.small_font.render(stat, True, (255, 255, 255))
            self.game_screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25

        viz_x = BOARD_WIDTH + SIDEBAR_WIDTH

        if hasattr(self, "arch_surface") and self.arch_surface is not None:
            try:
                arch_scaled = pygame.transform.smoothscale(
                    self.arch_surface, (ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT)
                )
                self.game_screen.blit(arch_scaled, (viz_x, 0))
            except Exception:
                pass

        if hasattr(self, "activity_surface") and self.activity_surface is not None:
            try:
                activity_scaled = pygame.transform.smoothscale(
                    self.activity_surface, (ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT)
                )
                activity_y = ARCH_WINDOW_HEIGHT if WINDOW_HEIGHT >= 2 * ARCH_WINDOW_HEIGHT else 0
                if activity_y + ARCH_WINDOW_HEIGHT <= WINDOW_HEIGHT:
                    self.game_screen.blit(activity_scaled, (viz_x, activity_y))
            except Exception:
                pass

        pygame.display.flip()
    
    def _handle_visualization_windows(self):
        if not self.render or not PYGAME_AVAILABLE:
            return
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_a] and self.show_arch_window:
            try:
                temp_screen = pygame.display.set_mode((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
                pygame.display.set_caption("Q-Network Architecture (Press any key to return)")
                temp_screen.blit(self.arch_surface, (0, 0))
                pygame.display.flip()
                
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                            waiting = False
                
                self.game_screen = pygame.display.set_mode((BOARD_WIDTH + SIDEBAR_WIDTH, BOARD_HEIGHT))
                pygame.display.set_caption("Tetris DQN - Game")
            except:
                pass
        
        if keys[pygame.K_n] and self.show_activity_window:
            try:
                temp_screen = pygame.display.set_mode((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
                pygame.display.set_caption("Network Activity (Press any key to return)")
                temp_screen.blit(self.activity_surface, (0, 0))
                pygame.display.flip()
                
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                            waiting = False
                
                self.game_screen = pygame.display.set_mode((BOARD_WIDTH + SIDEBAR_WIDTH, BOARD_HEIGHT))
                pygame.display.set_caption("Tetris DQN - Game")
            except:
                pass
    
    def _render_architecture(self, episode: int):
        if not self.render or not PYGAME_AVAILABLE:
            return
        
        
        surface = pygame.Surface((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
        surface.fill((0, 0, 0))
        
        NET_WIN_W = ARCH_WINDOW_WIDTH
        PADDING_X = 80
        num_layers = 4
        layer_x_positions = [PADDING_X + i * (NET_WIN_W - 2 * PADDING_X) // (num_layers - 1) for i in range(num_layers)]
        layer_sizes = [27, 16, 16, 5]  # Input, Hidden1 (grouped), Hidden2 (grouped), Output
        layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
        
        weights = []
        weights.append(self.q_network.fc1.weight.data.cpu().numpy())  # (128, 27)
        weights.append(self.q_network.fc2.weight.data.cpu().numpy())  # (128, 128)
        weights.append(self.q_network.fc3.weight.data.cpu().numpy())  # (5, 128)
        
        neuron_positions = []
        
        for layer_idx in range(4):
            size = layer_sizes[layer_idx]
            x = layer_x_positions[layer_idx]
            y_start = 100
            spacing = 500 / max(size, 1)
            positions = []
            
            for i in range(size):
                y = y_start + i * spacing
                positions.append((x, y))
            neuron_positions.append(positions)
        
        for layer_idx in range(3):
            current_positions = neuron_positions[layer_idx]
            next_positions = neuron_positions[layer_idx + 1]
            weight_matrix = weights[layer_idx]
            
            if layer_idx == 0:  # Input to Hidden1
                for i, (x1, y1) in enumerate(current_positions):
                    for j, (x2, y2) in enumerate(next_positions):
                        group_weights = weight_matrix[j*8:(j+1)*8, i] if j*8+8 <= 128 else weight_matrix[j*8:, i]
                        avg_weight = float(np.mean(group_weights))
                        color = connection_color(avg_weight, activation=1.0, was_correct=self.was_correct_last)
                        pygame.draw.aaline(surface, color, (x1, y1), (x2, y2))
            
            elif layer_idx == 1:  # Hidden1 to Hidden2
                for i in range(16):
                    for j in range(16):
                        x1, y1 = current_positions[i]
                        x2, y2 = next_positions[j]
                        group_weights = weight_matrix[j*8:(j+1)*8, i*8:(i+1)*8] if (j*8+8 <= 128 and i*8+8 <= 128) else weight_matrix[j*8:, i*8:]
                        avg_weight = float(np.mean(group_weights))
                        color = connection_color(avg_weight, activation=1.0, was_correct=self.was_correct_last)
                        pygame.draw.aaline(surface, color, (x1, y1), (x2, y2))
            
            else:  # Hidden2 to Output
                for i, (x1, y1) in enumerate(current_positions):
                    for j, (x2, y2) in enumerate(next_positions):
                        group_weights = weight_matrix[j, i*8:(i+1)*8] if i*8+8 <= 128 else weight_matrix[j, i*8:]
                        avg_weight = float(np.mean(group_weights))
                        color = connection_color(avg_weight, activation=1.0, was_correct=self.was_correct_last)
                        pygame.draw.aaline(surface, color, (x1, y1), (x2, y2))
        
        action_names = ["←", "→", "↻", "↓", "⬇"]
        input_labels = [f"H{i}" for i in range(10)] + ["holes", "bump", "lines"] + [f"p{i}" for i in range(7)] + [f"n{i}" for i in range(7)]
        
        for layer_idx in range(4):
            size = layer_sizes[layer_idx]
            positions = neuron_positions[layer_idx]
            
            for i, (x, y) in enumerate(positions):
                rect = pygame.Rect(0, 0, 6, 6)
                rect.center = (int(x), int(y))
                pygame.draw.rect(surface, (255, 255, 255), rect)
                
                if layer_idx == 0:  # Input layer
                    if i < len(input_labels):
                        label = input_labels[i]
                        text = self.tiny_font.render(label, True, (200, 200, 200))
                        text_rect = text.get_rect(center=(x, y - 25))
                        surface.blit(text, text_rect)
                elif layer_idx == 3:  # Output layer
                    if i < len(action_names):
                        label = action_names[i]
                        text = self.small_font.render(label, True, (200, 200, 200))
                        text_rect = text.get_rect(center=(x, y))
                        surface.blit(text, text_rect)
                else:  # Hidden layers (grouped)
                    text = self.tiny_font.render("×8", True, (150, 150, 150))
                    text_rect = text.get_rect(center=(x, y))
                    surface.blit(text, text_rect)
            
            text = self.font.render(layer_names[layer_idx], True, (255, 255, 255))
            surface.blit(text, (layer_x_positions[layer_idx] - 40, 50))
        
        title = self.font.render(f"Q-Network Weights | Episode {episode}", True, (255, 255, 255))
        surface.blit(title, (250, 20))
        
        self.arch_surface = surface
        
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            pygame.image.save(surface, results_dir / f"architecture_ep_{episode}.png")
            print(f"Architecture visualization saved: results/architecture_ep_{episode}.png")
        except Exception as e:
            print(f"Warning: Could not save architecture visualization: {e}")
        
        self.show_arch_window = True
    
    def _render_activity(self, state: np.ndarray, action: int):
        if not self.render or not PYGAME_AVAILABLE:
            return
        
        surface = pygame.Surface((ARCH_WINDOW_WIDTH, ARCH_WINDOW_HEIGHT))
        surface.fill((0, 0, 0))
        
        activations = get_activations(self.q_network, state, self.device)
        
        weights = [
            self.q_network.fc1.weight.data.cpu().numpy(),
            self.q_network.fc2.weight.data.cpu().numpy(),
            self.q_network.fc3.weight.data.cpu().numpy(),
        ]
        
        NET_WIN_W = ARCH_WINDOW_WIDTH
        NET_WIN_H = ARCH_WINDOW_HEIGHT
        PADDING_X = 80
        PADDING_Y = 30
        
        num_layers = 4
        layer_xs = [
            PADDING_X + i * (NET_WIN_W - 2 * PADDING_X) // (num_layers - 1)
            for i in range(num_layers)
        ]
        layer_sizes = [INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]
        
        neuron_pos, display_sizes = compute_neuron_positions(
            layer_sizes, layer_xs, NET_WIN_H, PADDING_Y, hidden_group_size=4)
        
        for layer_idx in range(3):
            stride = 4 if layer_idx == 1 else 1
            src_acts = activations[layer_idx]
            src_pos = neuron_pos[layer_idx]
            dst_pos = neuron_pos[layer_idx + 1]
            w = weights[layer_idx]
            
            for out_i in range(0, w.shape[0], stride):
                for in_i in range(0, w.shape[1], stride):
                    weight = w[out_i, in_i]
                    
                    src_d = in_i // 4 if w.shape[1] > 32 else in_i 
                    dst_d = out_i // 4 if w.shape[0] > 32 else out_i
                    
                    if src_d >= len(src_pos) or dst_d >= len(dst_pos):
                        continue
                    
                    act = src_acts[in_i] if in_i < len(src_acts) else 0.0
                    
                    if abs(act) < 0.01:
                        continue
                        
                    mag = math.tanh(abs(weight) * abs(act) * 3.0)
                    if mag < 0.1:
                        continue
                    
                    color = connection_color(weight, act, self.was_correct_last)
                    pygame.draw.aaline(surface, color, src_pos[src_d], dst_pos[dst_d])
                    
                    if mag > 0.5:
                        bright = tuple(min(255, c + 80) for c in color[:3])
                        pygame.draw.aaline(surface, bright, src_pos[src_d], dst_pos[dst_d])
        
        draw_all_neurons(surface, neuron_pos, activations, action, self.was_correct_last)
        
        layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
        for l, (label, x) in enumerate(zip(layer_names, layer_xs)):
            font_to_use = getattr(self, 'header_font', self.font)
            surf = font_to_use.render(label, True, (180, 180, 180))
            surface.blit(surf, (x - surf.get_width()//2, 8))
        
        action_names = ["←", "→", "↻", "↓", "⬇"]
        action_name = action_names[action] if action < len(action_names) else str(action)
        title = self.font.render(f"Network Activity | Step {self.total_steps} | Action: {action_name}", True, (255, 255, 255))
        surface.blit(title, (20, 20))
        
        for i, (x, y) in enumerate(neuron_pos[-1]):
            if i < len(action_names):
                label = action_names[i]
                text = self.small_font.render(label, True, (255, 255, 255))
                surface.blit(text, (x + 20, y - 5))
                
                q_values = activations[-1]
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / np.sum(exp_q) if np.sum(exp_q) > 0 else np.ones_like(exp_q) / len(q_values)
                prob = float(probs[i])
                bar_width = int(prob * 100)
                is_chosen = (i == action)
                bar_color = (0, 220, 70) if is_chosen and self.was_correct_last else \
                            (220, 40, 40) if is_chosen and not self.was_correct_last else \
                            (80, 80, 80)
                bar_rect = pygame.Rect(int(x) + 40, int(y) - 3, bar_width, 6)
                pygame.draw.rect(surface, bar_color, bar_rect)
        
        self.activity_surface = surface
        
        if self.total_steps % 1000 == 0:
            try:
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                pygame.image.save(surface, results_dir / f"activity_step_{self.total_steps}.png")
            except Exception:
                pass
    



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Tetris")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    
    args = parser.parse_args()
    
    render = args.render and not args.headless
    
    trainer = TetrisDQNTrainer(render=render)
    trainer.train(args.episodes)


if __name__ == "__main__":
    main()
