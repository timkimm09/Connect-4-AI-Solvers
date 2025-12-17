# Play against one of the agents
from connect4 import Connect4Env
from rl_agent import RLAgent
from rl_agent import train_rl_agent
import pickle
# while True:
#     print("Enter 1-4 to play one of the agents:\n")
#     print("1. Minimax + Alpha-Beta")
#     print("2. Reinforcement Learning")
#     print("3. Monte Carlo Tree Search")
#     print("4. Deterministic Search")
#     agentNum = int(input())
#     if 1 <= agentNum <= 4:
#         break
agentNum = 2

if agentNum == 2:
    num_episodes = 100000
    decay_rate = 0.99999
    the_opp='smarter_opp'
    # Q_table = train_rl_agent(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate)
    # with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
    #     pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'_'+the_opp+'.pickle'
    env = Connect4Env(render=False, wait_time=0, game_mode="human vs ai")
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            Q_table = obj
    except Exception:
        print("File doesn't exist")
    
    agent = RLAgent(Q_table)
    obs, info = env.reset()
    done = False
    while not done:
        valid_moves = env.get_valid_moves()
        if env.current_player == 1:
            action = agent.select_move(env.board, env.current_player, valid_moves)
        else:
            action = int(input("Enter 0-6, where 1 is the computer, -1 is you.\n"))
        obs, reward, term, trunc, info = env.step(action)
        print(env.board)
        print("")
        if term or trunc:
            done = True
            break
    if "winner" in info and info["winner"] == 1:
        print("Agent Wins")
    elif  "winner" in info and info["winner"] == -1:
        print("Player Wins")
    elif info.get("draw"):
        print("Draw")
