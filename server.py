import asyncio
import json
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from tetris_dqn import TetrisDQNTrainer, TetrisEngine, get_activations, SEED, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebTetrisDQNTrainer(TetrisDQNTrainer):
    def __init__(self):
        super().__init__(render=False)
        
    async def train_generator(self, websocket: WebSocket, num_episodes: int):
        env = TetrisEngine()
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_length = 0
            self.correct_streak = 0
            self.wrong_streak = 0
            
            await self._stream_state(websocket, env, state, 0, episode, num_episodes)
            
            while not done:
                action = self.select_action(state)
                
                next_state, reward, done = env.step(action)
                
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    best_action = q_values.argmax().item()
                    self.was_correct_last = (action == best_action)
                
                self.buffer.push(state, action, reward, next_state, done)
                
                if self.total_steps % 4 == 0 and len(self.buffer) >= BATCH_SIZE:
                    self.train_step()
                
                state = next_state
                self.total_steps += 1
                episode_length += 1
                
                self.epsilon = max(0.01, self.epsilon * 0.995)
                
                await self._stream_state(websocket, env, state, action, episode, num_episodes)
                await asyncio.sleep(0.01) # Yield to event loop to prevent blocking
                
            self.episode_scores.append(env.score)
            self.episode_lines.append(env.lines_cleared)
            self.episode_lengths.append(episode_length)
            
        await websocket.send_json({"status": "complete"})

    async def _stream_state(self, websocket: WebSocket, env: TetrisEngine, state: np.ndarray, action: int, episode: int, total_episodes: int):
        activations = get_activations(self.q_network, state, self.device)
        
        weights = [
            self.q_network.fc1.weight.data.cpu().numpy().tolist(),
            self.q_network.fc2.weight.data.cpu().numpy().tolist(),
            self.q_network.fc3.weight.data.cpu().numpy().tolist(),
        ]
        
        payload = {
            "type": "frame",
            "game": {
                "board": env.board.tolist(),
                "current_piece": env.current_piece_type,
                "next_piece": env.next_piece_type,
                "current_piece_col": env.current_piece_col,
                "current_piece_row": env.current_piece_row,
                "current_rotation": env.current_piece_rotation,
                "score": env.score,
                "lines": env.lines_cleared,
                "episode": episode + 1,
                "total_episodes": total_episodes,
                "epsilon": self.epsilon,
                "steps": self.total_steps,
                "buffer_size": len(self.buffer),
                "is_done": env.done
            },
            "network": {
                "activations": [a.tolist() for a in activations],
                "weights": weights,
                "action": action,
                "was_correct": self.was_correct_last
            }
        }
        await websocket.send_json(payload)

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    trainer = WebTetrisDQNTrainer()
    try:
        data = await websocket.receive_text()
        config = json.loads(data)
        episodes = config.get("episodes", 10)
        
        await trainer.train_generator(websocket, num_episodes=episodes)
    except Exception as e:
        print(f"WebSocket disconnected or error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
