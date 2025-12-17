import numpy as np
from connect4 import Connect4Env

WIN_SCORE = 10**9
FORCED_WIN_BONUS = 10**6
OPEN3_SCORE = 50
OPEN2_SCORE = 10
CENTER_BONUS = 3
MAX_NEG = -1e18

def get_valid_moves(board):
    return [c for c in range(board.shape[1]) if board[0, c] == 0]


def drop_piece(board, col, player):
    new_board = board.copy()
    rows = new_board.shape[0]
    for r in range(rows - 1, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            return new_board
    raise ValueError("drop_piece called on full column")


def check_win(board, player):
    rows, cols = board.shape

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if np.all(board[r, c:c+4] == player):
                return True

    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            if np.all(board[r:r+4, c] == player):
                return True

    # Diagonal down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i, c+i] == player for i in range(4)):
                return True

    # Diagonal up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i, c+i] == player for i in range(4)):
                return True

    return False


def board_is_full(board):
    return not (board == 0).any()


# Heuristic function
def evaluate_position(board, player):
    """
    Pure evaluation function (no env calls):
    - Rewards open 3-in-a-row with one empty end (higher)
    - Rewards open 2-in-a-row with two empties (smaller)
    - Penalizes symmetric opponent patterns
    - Adds small center-column bias
    - Adds huge bonus/penalty for immediate forced wins/losses (soft)
    """
    opp = -player
    rows, cols = board.shape
    score = 0

    def score_line_segment(segment):
        """Score a length-4 segment (numpy array)."""
        s = 0
        cnt_p = int(np.count_nonzero(segment == player))
        cnt_o = int(np.count_nonzero(segment == opp))
        cnt_e = int(np.count_nonzero(segment == 0))

        # Only score if the segment is not already blocked for that player.
        # For open3: exactly 3 of player's and 1 empty => good if not opponent nearby blocking ends
        if cnt_p == 3 and cnt_e == 1:
            s += OPEN3_SCORE
        if cnt_p == 2 and cnt_e == 2:
            s += OPEN2_SCORE

        if cnt_o == 3 and cnt_e == 1:
            s -= OPEN3_SCORE
        if cnt_o == 2 and cnt_e == 2:
            s -= OPEN2_SCORE

        return s

    # Evaluate all 4-length segments on board
    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            seg = board[r, c:c+4]
            score += score_line_segment(seg)

    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            seg = board[r:r+4, c]
            score += score_line_segment(seg)

    # Diagonals
    for r in range(rows - 3):
        for c in range(cols - 3):
            seg = np.array([board[r+i, c+i] for i in range(4)])
            score += score_line_segment(seg)

    for r in range(3, rows):
        for c in range(cols - 3):
            seg = np.array([board[r-i, c+i] for i in range(4)])
            score += score_line_segment(seg)

    # Center column preference
    center_col = cols // 2
    center_count = int(np.count_nonzero(board[:, center_col] == player))
    score += CENTER_BONUS * center_count

    # Soft forced win / loss check: simulate single-step drops
    valid_moves = get_valid_moves(board)
    for m in valid_moves:
        b2 = drop_piece(board, m, player)
        if check_win(b2, player):
            score += FORCED_WIN_BONUS

    for m in valid_moves:
        b2 = drop_piece(board, m, opp)
        if check_win(b2, opp):
            score -= FORCED_WIN_BONUS

    return score


# Minimax (Negamax) Search with Alpha-Beta pruning
def minimax(board, depth, alpha, beta, player):
    """
    Negamax implementation with alpha-beta.
    Operates on pure boards only (no env mutation).
    Returns (score, best_move)
    """
    valid_moves = get_valid_moves(board)

    # check if player has already won on this board or opponent has won
    if check_win(board, player):
        return WIN_SCORE, None
    if check_win(board, -player):
        return -WIN_SCORE, None

    # Draw
    if depth == 0 or not valid_moves or board_is_full(board):
        return evaluate_position(board, player), None

    best_score = -float("inf")
    best_move = None

    # Move ordering --> try center-first to improve pruning
    cols = board.shape[1]
    center = cols // 2
    # order moves by distance to center
    ordered_moves = sorted(valid_moves, key=lambda c: abs(c - center))

    for move in ordered_moves:
        new_board = drop_piece(board, move, player)

        # Recurse for opponent; negamax flip
        score, _ = minimax(new_board, depth - 1, -beta, -alpha, -player)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break  # beta cutoff

    return best_score, best_move


# Top-level chooser that uses the environment only for state read
def choose_best_move(env, depth=6):
    """
    Returns best column index for current env state.
    Uses pure board minimax; uses env only to read current board and current_player.
    """
    board = env.board.copy()
    player = env.current_player

    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return None

    best_score = -float("inf")
    best_move = None

    # order moves center-first for quick pruning
    cols = board.shape[1]
    center = cols // 2
    ordered_moves = sorted(valid_moves, key=lambda c: abs(c - center))

    for move in ordered_moves:
        new_board = drop_piece(board, move, player)
        score, _ = minimax(new_board, depth - 1, -float("inf"), float("inf"), -player)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# example
if __name__ == "__main__":
    env = Connect4Env(render=True, wait_time=300)
    state, _ = env.reset()
    env.render()

    total_steps = {1: 0, -1: 0}
    moves = []

    while True:
        mover = env.current_player

        if env.game_mode == "human_vs_ai" and mover == 1:
            action = env.get_human_move()
        else:
            action = choose_best_move(env, depth=6)

        if action is None:
            print("No valid moves left.")
            break

        print(f"Player {mover} chooses: {action}")
        state, reward, terminated, truncated, info = env.step(action)

        total_steps[mover] += 1
        moves.append((mover, action))

        env.render()
        print(state)

        if terminated or truncated:
            print("Game over:", info)
            break

    # Summary
    print("\n=== GAME METRICS ===")
    print(f"Total moves: {total_steps[1] + total_steps[-1]}")
    print(f"Player 1 -> Steps: {total_steps[1]}")
    print(f"Player 2 -> Steps: {total_steps[-1]}")
