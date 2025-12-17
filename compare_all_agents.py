import random
from tqdm import tqdm
from connect4 import Connect4Env
from mcts_approach import choose_best_move as mcts_move
from minimax_alphabeta_pruning import choose_best_move as minimax_move
from deterministic_agent import DeterministicAgent
from rl_agent import RLAgent, choose_smart_move
import pickle
import time

def load_rl_agent(filename):
    try:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
        return RLAgent(q_table)
    except Exception as e:
        print(f"Couldn't load RL agent from {filename}: {e}")
        return None

def random_move(env):
    valid = env.get_valid_moves()
    return random.choice(valid) if valid else 0

def heuristic_move(env):
    valid = env.get_valid_moves()
    return choose_smart_move(env.board, valid, env.current_player)

def run_match(env, agent1_func, agent2_func, agent1_name, agent2_name):
    state, _ = env.reset()
    
    while True:
        player = env.current_player
        
        if player == 1:
            action = agent1_func(env)
        else:
            action = agent2_func(env)
        
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            if "winner" in info:
                return info["winner"]
            return 0

def compare_agents(agent1_func, agent2_func, agent1_name, agent2_name, num_games=100):
    env = Connect4Env(render=False, wait_time=0)
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    print(f"\n{agent1_name} (Player 1) vs {agent2_name} (Player 2)")
    
    start_time = time.time()
    
    for _ in tqdm(range(num_games)):
        result = run_match(env, agent1_func, agent2_func, agent1_name, agent2_name)
        if result == 1:
            agent1_wins += 1
        elif result == -1:
            agent2_wins += 1
        else:
            draws += 1
    
    elapsed = time.time() - start_time
    
    print(f"\nResults ({num_games} games, {elapsed:.1f}s):")
    print(f"  {agent1_name} wins: {agent1_wins} ({100*agent1_wins/num_games:.1f}%)")
    print(f"  {agent2_name} wins: {agent2_wins} ({100*agent2_wins/num_games:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/num_games:.1f}%)")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'time': elapsed
    }

def main():
    print("Connect 4 AI Comparison")
    
    det_agent_p1 = DeterministicAgent(player=1)
    det_agent_p2 = DeterministicAgent(player=-1)
    
    rl_agent = load_rl_agent("Q_table_500000_0.999999_smarter_opp.pickle")
    
    def mcts_500(env):
        return mcts_move(env, num_simulations=500)
    
    def mcts_1000(env):
        return mcts_move(env, num_simulations=1000)
    
    def minimax_5(env):
        return minimax_move(env, depth=5)
    
    def det_p1(env):
        return det_agent_p1.get_move(env.board.copy())
    
    def det_p2(env):
        return det_agent_p2.get_move(env.board.copy())
    
    def rl_move(env):
        if rl_agent is None:
            return random_move(env)
        valid = env.get_valid_moves()
        return rl_agent.select_move(env.board, env.current_player, valid)
    
    num_games = 50
    
    print("MCTS vs Random")
    compare_agents(mcts_500, random_move, "MCTS-500", "Random", num_games)
    
    print("MCTS vs Heuristic")
    compare_agents(mcts_500, heuristic_move, "MCTS-500", "Heuristic", num_games)
    
    print("MCTS vs Deterministic")
    compare_agents(mcts_500, det_p2, "MCTS-500", "Deterministic", num_games)
    
    print("MCTS vs Minimax (depth 5)")
    compare_agents(mcts_1000, minimax_5, "MCTS-1000", "Minimax-5", num_games)
    
    if rl_agent is not None:
        print("MCTS vs RL Agent")
        compare_agents(mcts_500, rl_move, "MCTS-500", "RL-Agent", num_games)
    
    print("Minimax vs Deterministic")
    compare_agents(minimax_5, det_p2, "Minimax-5", "Deterministic", num_games)
    
    if rl_agent is not None:
        print("Minimax vs RL Agent")
        compare_agents(minimax_5, rl_move, "Minimax-5", "RL-Agent", num_games)

    if rl_agent is not None:
        print("RL Agent vs Deterministic")
        compare_agents(rl_move, det_p2, "RL-Agent", "Deterministic", num_games)

if __name__ == "__main__":
    main()