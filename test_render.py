from tetris_dqn import TetrisDQNTrainer, TetrisEngine
import numpy as np

trainer = TetrisDQNTrainer(render=True)
env = TetrisEngine()
state = env.reset()
Trainer_render = getattr(trainer, "_render_activity")
try:
    Trainer_render(state, 0)
    print("Render successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
