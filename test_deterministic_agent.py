from connect4 import Connect4Env
from deterministic_agent import DeterministicAgent
import numpy as np
import time

def play_vs_random():
    env = Connect4Env(render=True, wait_time=300)
    state, info = env.reset()

    agent = DeterministicAgent(player=1)

    while True:
        current_player = env.current_player

        if current_player == agent.player:
            # deterministic agent move
            move = agent.get_move(env.board.copy())
            print(f"\nDeterministic Agent chooses column: {move}")
        else:
            # random opponent
            valid = env.get_valid_moves(env.board)
            move = np.random.choice(valid)
            print(f"\nRandom Opponent chooses column: {move}")

        state, reward, terminated, truncated, info = env.step(move)
        env.render()

        if terminated or truncated:
            print("\n=== GAME OVER ===")
            print(info)
            break

if __name__ == "__main__":
    play_vs_random()
