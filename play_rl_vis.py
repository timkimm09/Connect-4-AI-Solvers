import pickle
import numpy as np

from connect4 import Connect4Env
from rl_agent import RLAgent

# trained Q-table file
Q_TABLE_PATH = "Q_table_500000_0.999999_smarter_opp.pickle"

# In training, the agent was "1" and the opponent was -1
HUMAN = -1
AI = 1


def load_agent(q_table_path: str) -> RLAgent:
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    return RLAgent(q_table)


def main():
    env = Connect4Env(render=True, wait_time=0, game_mode="human_vs_ai")
    state, _ = env.reset()

    agent = load_agent(Q_TABLE_PATH)

    print("You are RED (1). AI is YELLOW (-1). Click columns to play!")

    while True:
        env.render()

        player = env.current_player

        if player == HUMAN:
            action = env.get_human_move()
            print(f"Human plays column {action}")

        else:
            valid_moves = env.get_valid_moves()
            if len(valid_moves) == 0:
                print("Draw")
                break

            # Agent's move
            print(player)
            action = agent.select_move(env.board, player, valid_moves)
            print(f"Agent plays column {action}")

        # step action in the environment
        state, reward, term, trunc, info = env.step(action)

        # Check if game over
        if term or trunc:
            env.render()
            if "winner" in info:
                if info["winner"] == HUMAN:
                    print("You win")
                elif info["winner"] == AI:
                    print("AI Agent wins")
            else:
                print("Draw")
            break


if __name__ == "__main__":
    main()
