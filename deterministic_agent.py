import numpy as np
from connect4 import Connect4Env

class DeterministicAgent:
    def __init__(self, player=1):
        """
        player = 1 or -1
        """
        self.player = player
        self.env = Connect4Env() 

    #  check if a move wins immediately
    def _move_wins(self, board, col, player):
        new_board = self.env.drop_piece(board, col, player)
        # temporarily assign the board for check_win
        saved = self.env.board
        self.env.board = new_board
        win = self.env.check_win(player)
        self.env.board = saved
        return win
      
    #  check if opponent can win after this move
    def _opponent_can_win_next(self, board, col, player):
        opponent = -player
        new_board = self.env.drop_piece(board, col, player)
        opp_moves = self.env.get_valid_moves(new_board)

        for opp_col in opp_moves:
            opp_board = self.env.drop_piece(new_board, opp_col, opponent)

            saved = self.env.board
            self.env.board = opp_board
            opp_wins = self.env.check_win(opponent)
            self.env.board = saved

            if opp_wins:
                return True

        return False
      
    #  heuristic scoring function
    def evaluate_board(self, board, player):
        score = 0
        opp = -player
        rows, cols = board.shape

        # center column bonus
        center_col = cols // 2
        center_array = board[:, center_col]
        score += 3 * np.count_nonzero(center_array == player)

        # helper : score a 4-cell window
        def score_window(window):
            nonlocal score

            # your 3 / your 2
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
                score += 5
            if np.count_nonzero(window == player) == 2 and np.count_nonzero(window == 0) == 2:
                score += 2

            # opponent 3 / opponent 2
            if np.count_nonzero(window == opp) == 3 and np.count_nonzero(window == 0) == 1:
                score -= 5
            if np.count_nonzero(window == opp) == 2 and np.count_nonzero(window == 0) == 2:
                score -= 2

        # horizontal check
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c:c+4]
                score_window(window)

        # vertical check
        for r in range(rows - 3):
            for c in range(cols):
                window = board[r:r+4, c]
                score_window(window)

        # diagonal down-right
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r+i][c+i] for i in range(4)])
                score_window(window)

        # diagonal up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r-i][c+i] for i in range(4)])
                score_window(window)

        return score

    #  main move selection logic
    def get_move(self, board):
        """
        Given the current board (numpy array), return the best column index.
        """

        valid_moves = self.env.get_valid_moves(board)
        best_score = -float("inf")
        best_move = None

        # evaluate each legal move
        for col in valid_moves:

            # if this move wins immediately → choose it
            if self._move_wins(board, col, self.player):
                return col

            # if this move allows the opponent to win next turn → give very bad score
            if self._opponent_can_win_next(board, col, self.player):
                move_score = -100
            else:
                # otherwise evaluate the resulting board using heuristic
                future_board = self.env.drop_piece(board, col, self.player)
                move_score = self.evaluate_board(future_board, self.player)

            # track best
            if move_score > best_score:
                best_score = move_score
                best_move = col

        return best_move
