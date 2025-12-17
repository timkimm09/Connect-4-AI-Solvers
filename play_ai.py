import numpy as np
from connect4 import Connect4Env
from minimax_alphabeta_pruning import choose_best_move

env = Connect4Env(render=True, wait_time=0, game_mode="human_vs_ai")
state, _ = env.reset()

HUMAN = 1
AI = -1

print("You are RED (1). AI is YELLOW (-1). Click columns to play!")

while True:
    env.render()

    player = env.current_player

    # Human turn
    if player == HUMAN:
        action = env.get_human_move()
        print(f"Human plays {action}")

    # AI turn
    else:
        action = choose_best_move(env, depth=5)
        print(f"AI plays {action}")

    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.render()
        if "winner" in info:
            if info["winner"] == HUMAN:
                print("Human wins!")
            else:
                print("AI wins!")
        else:
            print("Draw.")
        break
