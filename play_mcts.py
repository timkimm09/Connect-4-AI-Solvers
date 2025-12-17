from connect4 import Connect4Env
from mcts_approach import choose_best_move

HUMAN = -1
AI = 1
NUM_SIMULATIONS = 2000

def main():
    env = Connect4Env(render=True, wait_time=0, game_mode="human_vs_ai")
    state, _ = env.reset()

    print("You are YELLOW (-1). AI is RED (1). Click columns to play.")
    print(f"MCTS is running {NUM_SIMULATIONS} simulations per move.")

    while True:
        env.render()

        player = env.current_player

        if player == HUMAN:
            action = env.get_human_move()
            print(f"Human plays column {action}")
        else:
            print("Standby")
            action = choose_best_move(env, num_simulations=NUM_SIMULATIONS)
            print(f"AI plays column {action}")

        state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.render()
            if "winner" in info:
                if info["winner"] == HUMAN:
                    print("You win.")
                elif info["winner"] == AI:
                    print("AI wins.")
            else:
                print("Draw.")
            break

if __name__ == "__main__":
    main()