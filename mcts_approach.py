import numpy as np
import math
import random
from connect4 import Connect4Env

ROWS = 6
COLS = 7
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

def board_to_bitboards(board):
    p1 = 0
    p2 = 0
    heights = [0] * COLS
    for c in range(COLS):
        for r in range(ROWS - 1, -1, -1):
            if board[r][c] != 0:
                bit = 1 << (c * 7 + (ROWS - 1 - r))
                if board[r][c] == 1:
                    p1 |= bit
                else:
                    p2 |= bit
                heights[c] = ROWS - r
    return p1, p2, heights

def bitboards_to_board(p1, p2):
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    for c in range(COLS):
        for r in range(ROWS):
            bit = 1 << (c * 7 + (ROWS - 1 - r))
            if p1 & bit:
                board[r][c] = 1
            elif p2 & bit:
                board[r][c] = -1
    return board

def check_win_bitboard(bitboard):
    directions = [1, 7, 6, 8]
    for d in directions:
        bb = bitboard & (bitboard >> d)
        if bb & (bb >> (2 * d)):
            return True
    return False

def drop_piece_bitboard(p1, p2, heights, col, player):
    if heights[col] >= ROWS:
        return None, None, None
    row = heights[col]
    bit = 1 << (col * 7 + row)
    new_heights = heights.copy()
    new_heights[col] += 1
    if player == 1:
        return p1 | bit, p2, new_heights
    else:
        return p1, p2 | bit, new_heights

def get_valid_moves_bitboard(heights):
    return [c for c in MOVE_ORDER if heights[c] < ROWS]

class MCTSNode:
    __slots__ = ['p1', 'p2', 'heights', 'player', 'parent', 'move', 'children', 
                 'visits', 'wins', 'untried_moves', 'is_terminal_cached', 'winner_cached']
    
    def __init__(self, p1, p2, heights, player, parent=None, move=None):
        self.p1 = p1
        self.p2 = p2
        self.heights = heights
        self.player = player
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.wins = 0.0
        self.is_terminal_cached = None
        self.winner_cached = None
        self._compute_terminal()
        if self.is_terminal_cached:
            self.untried_moves = []
        else:
            self.untried_moves = get_valid_moves_bitboard(heights)
    
    def _compute_terminal(self):
        if check_win_bitboard(self.p1):
            self.is_terminal_cached = True
            self.winner_cached = 1
        elif check_win_bitboard(self.p2):
            self.is_terminal_cached = True
            self.winner_cached = -1
        elif all(h >= ROWS for h in self.heights):
            self.is_terminal_cached = True
            self.winner_cached = 0
        else:
            self.is_terminal_cached = False
            self.winner_cached = None
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.is_terminal_cached
    
    def get_winner(self):
        return self.winner_cached
    
    def ucb1(self, exploration):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration):
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))
    
    def most_visited_child(self):
        return max(self.children.values(), key=lambda c: c.visits)

class MCTS:
    def __init__(self, num_simulations=500, exploration=0.5):
        self.num_simulations = num_simulations
        self.exploration = exploration
    
    def search(self, board, player):
        p1, p2, heights = board_to_bitboards(board)
        root = MCTSNode(p1, p2, heights, player)
        
        if root.is_terminal():
            valid = get_valid_moves_bitboard(heights)
            return valid[0] if valid else 0
        
        immediate = self._check_immediate_wins(p1, p2, heights, player)
        if immediate is not None:
            return immediate
        
        for _ in range(self.num_simulations):
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.exploration)
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result, root.player)
        
        if not root.children:
            valid = get_valid_moves_bitboard(heights)
            return valid[0] if valid else 0
        
        return root.most_visited_child().move
    
    def _check_immediate_wins(self, p1, p2, heights, player):
        valid_moves = get_valid_moves_bitboard(heights)
        my_board = p1 if player == 1 else p2
        opp_board = p2 if player == 1 else p1
        
        for col in valid_moves:
            new_p1, new_p2, _ = drop_piece_bitboard(p1, p2, heights, col, player)
            new_my = new_p1 if player == 1 else new_p2
            if check_win_bitboard(new_my):
                return col
        for col in valid_moves:
            new_p1, new_p2, _ = drop_piece_bitboard(p1, p2, heights, col, -player)
            new_opp = new_p2 if player == 1 else new_p1
            if check_win_bitboard(new_opp):
                return col
        
        return None
    
    def _expand(self, node):
        move = node.untried_moves.pop(0)
        new_p1, new_p2, new_heights = drop_piece_bitboard(node.p1, node.p2, node.heights, move, node.player)
        child = MCTSNode(new_p1, new_p2, new_heights, -node.player, parent=node, move=move)
        node.children[move] = child
        return child
    
    def _simulate(self, node):
        if node.is_terminal():
            return node.get_winner()
        p1, p2, heights = node.p1, node.p2, node.heights.copy()
        player = node.player
        
        while True:
            valid = [c for c in range(COLS) if heights[c] < ROWS]
            if not valid:
                return 0
            col = random.choice(valid)
            row = heights[col]
            bit = 1 << (col * 7 + row)
            heights[col] += 1
            if player == 1:
                p1 |= bit
                if check_win_bitboard(p1):
                    return 1
            else:
                p2 |= bit
                if check_win_bitboard(p2):
                    return -1
            
            player = -player
    
    def _backpropagate(self, node, result, root_player):
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                if result == -node.player:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            else:
                if result == root_player:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            node = node.parent

def choose_best_move(env, num_simulations=1000):
    mcts = MCTS(num_simulations=num_simulations)
    return mcts.search(env.board.copy(), env.current_player)

if __name__ == "__main__":
    env = Connect4Env()
    state, info = env.reset()
    
    symbols = {0: '.', 1: 'X', -1: 'O'}
    move_num = 0
    
    while True:
        move_num += 1
        player_symbol = 'X' if env.current_player == 1 else 'O'
        
        print(f"\nMove {move_num} - Player {player_symbol}")
        
        for row in env.board:
            print(' '.join(symbols[cell] for cell in row))
        print('0 1 2 3 4 5 6')
        
        action = choose_best_move(env, num_simulations=500)
        print(f"MCTS plays column: {action}")
        
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"\nFinal Board:")
            for row in state:
                print(' '.join(symbols[cell] for cell in row))
            print('0 1 2 3 4 5 6')
            
            if 'winner' in info:
                winner = 'X' if info['winner'] == 1 else 'O'
                print(f"\nPlayer {winner} wins")
            else:
                print("\nDraw")
            break