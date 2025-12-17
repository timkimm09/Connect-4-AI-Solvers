import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from connect4 import Connect4Env

def encode(board, current_player):
    current_state = board.flatten().tolist()
    code = ''.join(str(num) for num in current_state)
    return f"{code}_{int(current_player)}" # q-table only updates when its agent's turn (good for things like in the future if the agent goes second instead)

def simulate_drop(board, col, player):
    new_board = board.copy()
    rows, cols = new_board.shape
    for r in range(rows - 1, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board

def check_win_board(board, player):
    rows, cols = board.shape

    # horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if np.all(board[r, c:c+4] == player):
                return True

    # vertical
    for r in range(rows - 3):
        for c in range(cols):
            if np.all(board[r:r+4, c] == player):
                return True

    # diag down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i, c+i] == player for i in range(4)):
                return True

    # diag up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i, c+i] == player for i in range(4)):
                return True

    return False

def board_score(board, player):
    rows, cols = board.shape
    opp = -player
    score = 0

    AG_THREE  = 50
    AG_TWO    = 5
    AG_ONE    = 1
    OPP_THREE = -40   # blocking opponent is also important
    OPP_TWO   = -4

    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diag down-right
        (-1, 1),  # diag up-right
    ]

    for dr, dc in directions:
        for r in range(rows):
            for c in range(cols):
                # starting point must allow a length-4 window (makes this process easier)
                r_end = r + 3 * dr
                c_end = c + 3 * dc
                if not (0 <= r_end < rows and 0 <= c_end < cols):
                    continue

                window = [board[r + i*dr, c + i*dc] for i in range(4)]
                count_p = sum(1 for v in window if v == player)
                count_o = sum(1 for v in window if v == opp)

                # mixed window: both players present so just ignore
                if count_p > 0 and count_o > 0:
                    continue

                if count_p > 0:
                    if count_p == 3:
                        score += AG_THREE
                    elif count_p == 2:
                        score += AG_TWO
                    elif count_p == 1:
                        score += AG_ONE

                if count_o > 0:
                    if count_o == 3:
                        score += OPP_THREE
                    elif count_o == 2:
                        score += OPP_TWO

    return score

def choose_smart_move(board, valid_moves, player):
    opp = -player
    center_col = 3  # center column

    #1. immediate win for the agent
    for col in valid_moves:
        tmp = simulate_drop(board, col, player)
        if check_win_board(tmp, player):
            return col

    #2. block immediate win for opponent
    for col in valid_moves:
        tmp = simulate_drop(board, col, opp)
        if check_win_board(tmp, opp):
            return col

    #3. scoring the best moves based on the current # 3 in a rows, 2 in a rows
    # for both the agent and to block the opponent
    best_score = -1e9
    best_cols = []

    for col in valid_moves:
        tmp = simulate_drop(board, col, player)
        s = board_score(tmp, player)

        # 4) the center preference (well known heuristic)
        if col == center_col:
            s += 10

        if s > best_score:
            best_score = s
            best_cols = [col]
        elif s == best_score:
            best_cols.append(col)

    # 5) choose one of the best moves calculated from 3 randomly
    return random.choice(best_cols)

def train_rl_agent(num_episodes, gamma, epsilon, decay_rate, opp_type):
    env = Connect4Env(render=False, wait_time=0, game_mode='ai')
    Q_table = {}
    update_count_qsa = {}
    num_of_actions = 7 #7 columns
    
    def initS(board, current_player):
        s = encode(board, current_player)
        if s not in Q_table:
            Q_table[s] = np.zeros(num_of_actions, dtype=float)
        return s
    
    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        # first = random.choice([1, -1])
        # env.current_player = first
        while not done:
            if env.current_player == 1:
                state = initS(env.board, env.current_player)
                valid_moves = env.get_valid_moves()
                agentAction = False
                if len(valid_moves) == 0:
                    agent_reward = 0.0
                    next_state = None
                    done = True
                else:
                    # epsilon-greedy
                    if np.random.rand() > epsilon:
                        q_opt = Q_table[state]
                        # mask illegal moves to make training faster
                        mask = np.full(7, -1e9)
                        mask[valid_moves] = 0.0
                        action = int((q_opt + mask).argmax())
                    else:
                        # action = choose_smart_move(env.board, valid_moves, env.current_player)
                        action = random.choice(valid_moves)
                    agentAction = True
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc

                    # default agent reward
                    agent_reward = 0.0
                    next_state = None
                    if done:
                        if "winner" in info and info["winner"] == 1:
                            agent_reward = 1.0    # agent wins
                        elif "winner" in info and info["winner"] == -1:
                            agent_reward = -1.0   # rare, but random opponent wins
                        else:
                            agent_reward = 0.0    # draw, illegal, truncated, error
                        next_state = None
                    else:
                        # opponent's turn
                        opp_valid = env.get_valid_moves()
                        if len(opp_valid) == 0:
                            agent_reward = 0.0
                            done = True
                        else:
                            if opp_type == 1:
                                opp_action = random.choice(opp_valid)
                            else:
                                opp_action = choose_smart_move(env.board, opp_valid, env.current_player) # plays against itself (random moves)
                            next_obs, reward2, term2, trunc2, info2 = env.step(opp_action)
                            done = term2 or trunc2
                            if done:
                                if "winner" in info2 and info2["winner"] == -1:
                                    agent_reward = -1.0    # the opp playing random moves wins (agent loses)
                                else:
                                    agent_reward = 0.0 # game continues
                                    next_state = initS(env.board, env.current_player)

                if agentAction:
                #q-update for agent
                    update_count_qsa[(state, action)] = update_count_qsa.get((state, action), 0) + 1
                    learning_rate = float(1 / (1 + update_count_qsa[(state, action)]))
                
                    # V_old_opt (s') = max Q_old_opt
                    V_old_opt = 0 # if done, its terminated, there's not a future value
                    if next_state is not None: # game is not done yet, calculate q-value
                        next_valid = env.get_valid_moves()
                        if len(next_valid) > 0:
                            next_q_vals = Q_table[next_state]
                            # mask next invlid moves
                            next_mask = np.full(7, -1e9)
                            next_mask[next_valid] = 0.0
                            V_old_opt = np.max(next_q_vals + next_mask)
                
                    Q_old_opt = Q_table[state][action]
                    Q_table[state][action] = (1 - learning_rate)*Q_old_opt + learning_rate*(agent_reward+gamma*V_old_opt)
                if done:
                    break
            else:
                valid_moves = env.get_valid_moves()
                if len(valid_moves) == 0:
                    done = True
                    break
                if opp_type == 1:
                    opp_action = random.choice(valid_moves)
                else:
                    opp_action = choose_smart_move(env.board, valid_moves, env.current_player)
                next_obs, reward, term, trunc, info = env.step(opp_action)
                done = term or trunc
                if done:
                    break
                continue # agent's turn now
            obs = next_obs
        epsilon = epsilon * decay_rate
    
    return Q_table

class RLAgent:
    def __init__(self, Q_table):
        self.Q_table = Q_table
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"
        
        # self.model = Connect4Net().to(self.device)

        # if model_path is not None:
        #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # self.model.eval()
    def select_move(self, board, current_player, valid_moves):
        state = encode(board, current_player)
        if state in self.Q_table:
            q_opt = self.Q_table[state]
        else:
            q_opt = None
        
        if q_opt is None or np.allclose(q_opt, 0.0):
            return int(np.random.choice(valid_moves))
        
        mask = np.full(7, -1e9)
        mask[valid_moves] = 0.0
        return int(np.argmax(q_opt + mask))

if __name__== "__main__":
    num_episodes=10000
    decay_rate=0.9999
    opp_type = 2
    if opp_type == 1:
        the_opp = "random_opp"
    elif opp_type == 2:
        the_opp = "smarter_opp"
    Q_table = train_rl_agent(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate, opp_type=opp_type)
    with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'_'+the_opp+'.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

# some of the old cnn code which didn't work because
# for some reason I could not figure out the shape mismatching
# # with torch.no_grad():
#     if not isinstance(board, np.ndarray):
#         board = np.array(board, dtype=np.int8)
            
    #     state_tensor = board_to_tensor(board, current_player)

    #     # q values calculated with CNN instead of like PA2
    #     # cpu mode for a numpy array, and then flattens (1, 7) into (7,)
    #     # the output shape was (1,7)--one score per column
    #     q_values = self.model(state_tensor)[0].cpu().numpy().flatten()

    #     mask = np.full(7, -1e9, dtype=np.float32) # mask that punishes invalid moves
    #     mask[valid_moves] = 0.0 # no penalty for valid moves
            
    #     # valid move = the real neural network score
    #     # invalid move = an extremely low value (very bad so it will never play an invalid move in real-time)
    #     masked_q = q_values + mask 
            
        # return the column (1-7) with the highest q-score
        # return int(masked_q.argmax()) 

