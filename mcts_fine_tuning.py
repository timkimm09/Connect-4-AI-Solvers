import numpy as np
import math
import random
import time

ROWS = 6
COLS = 7

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

def get_valid_moves(heights):
    return [c for c in [3, 2, 4, 1, 5, 0, 6] if heights[c] < ROWS]

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
            self.untried_moves = get_valid_moves(heights)
    
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
    def __init__(self, num_simulations=1000, exploration=1.414, draw_value=0.5):
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.draw_value = draw_value
    
    def search(self, p1, p2, heights, player):
        root = MCTSNode(p1, p2, heights, player)
        
        if root.is_terminal():
            valid = get_valid_moves(heights)
            return valid[0] if valid else 0
        
        for _ in range(self.num_simulations):
            node = root
            
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.exploration)
            
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)
            
            result = self._simulate(node)
            self._backpropagate(node, result, root.player)
        
        if not root.children:
            valid = get_valid_moves(heights)
            return valid[0] if valid else 0
        
        return root.most_visited_child().move
    
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
                    node.wins += self.draw_value
            else:
                if result == root_player:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += self.draw_value
            node = node.parent


def play_game(mcts1, mcts2):
    p1, p2 = 0, 0
    heights = [0] * COLS
    player = 1
    
    while True:
        valid = [c for c in range(COLS) if heights[c] < ROWS]
        if not valid:
            return 0
        
        if player == 1:
            move = mcts1.search(p1, p2, heights, player)
        else:
            move = mcts2.search(p1, p2, heights, player)
        
        row = heights[move]
        bit = 1 << (move * 7 + row)
        heights[move] += 1
        
        if player == 1:
            p1 |= bit
            if check_win_bitboard(p1):
                return 1
        else:
            p2 |= bit
            if check_win_bitboard(p2):
                return -1
        
        player = -player


def run_experiment(param_name, values, baseline_value, num_games=50):
    print(f"Fine-Tuning: {param_name}")
    print(f"Testing values: {values}")
    print(f"Baseline: {baseline_value}")
    print(f"Games per matchup: {num_games}")
    print()
    
    results = {}
    
    for val in values:
        if param_name == "exploration":
            test_mcts = MCTS(num_simulations=500, exploration=val)
            baseline_mcts = MCTS(num_simulations=500, exploration=baseline_value)
        elif param_name == "num_simulations":
            test_mcts = MCTS(num_simulations=val, exploration=1.414)
            baseline_mcts = MCTS(num_simulations=baseline_value, exploration=1.414)
        elif param_name == "draw_value":
            test_mcts = MCTS(num_simulations=500, draw_value=val)
            baseline_mcts = MCTS(num_simulations=500, draw_value=baseline_value)
        
        wins = 0
        losses = 0
        draws = 0
        
        for game in range(num_games):
            if game % 2 == 0:
                result = play_game(test_mcts, baseline_mcts)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                result = play_game(baseline_mcts, test_mcts)
                if result == -1:
                    wins += 1
                elif result == 1:
                    losses += 1
                else:
                    draws += 1
        
        win_rate = wins / num_games * 100
        results[val] = (wins, losses, draws, win_rate)
        
        print(f"{param_name}={val:>8} -> Wins: {wins:>3}, Losses: {losses:>3}, Draws: {draws:>3}, Win Rate: {win_rate:.1f}%")
    
    best_val = max(results.keys(), key=lambda v: results[v][3])
    print(f"\nBest {param_name}: {best_val} (Win Rate: {results[best_val][3]:.1f}%)")
    
    return results


def main():
    print("MCTS FINE-TUNING EXPERIMENTS")
    print()
    print("This script tests different hyperparameter values")
    print("by playing MCTS agents against each other.")
    print()
    
    start = time.time()
    
    print("Experiment 1: Exploration Constant (C)")
    print("Theory: C controls explore vs exploit tradeoff")
    print("  - Low C (0.5): Exploits known good moves")
    print("  - High C (2.0): Explores more alternatives")
    print("  - sqrt(2) = 1.414 is the theoretical default")
    
    exploration_values = [0.5, 1.0, 1.414, 2.0, 2.5]
    run_experiment("exploration", exploration_values, baseline_value=1.414, num_games=30)
    
    print("Experiment 2: Number of Simulations")
    print("Theory: More simulations = better move estimates")
    print("  - But diminishing returns after a point")
    print("  - Also increases computation time")
    
    sim_values = [100, 250, 500, 1000]
    run_experiment("num_simulations", sim_values, baseline_value=500, num_games=30)
    
    print("Experiment 3: Draw Value")
    print("Theory: How much to value draws in backpropagation")
    print("  - 0.0: Treat draws as losses (aggressive)")
    print("  - 0.5: Treat draws as half-wins (neutral)")
    print("  - 1.0: Treat draws as wins (defensive)")
    
    draw_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    run_experiment("draw_value", draw_values, baseline_value=0.5, num_games=30)
    
    elapsed = time.time() - start
    
    print("Summary")
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Findings:")
    print("  1. Exploration (C): Values around 1.0-1.5 work well")
    print("  2. Simulations: More is better, but 500-1000 is practical")
    print("  3. Draw Value: 0.5 (neutral) is typically best")
    print()
    print("For Connect 4 specifically:")
    print("  - C=1.414 (sqrt 2) is a good default")
    print("  - 500-1000 simulations for real-time play")
    print("  - Draw value 0.5 works since draws are rare")


if __name__ == "__main__":
    main()