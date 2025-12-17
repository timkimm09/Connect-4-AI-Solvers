import gymnasium as gym
import numpy as np
import pygame

# CITATION: ChatGPT used here to assist in creating Connect4 Environment
class Connect4Env(gym.Env):
    """Gymnasium-compatible Connect 4 environment."""

    def __init__(self, render=False, wait_time=250, game_mode='ai'):
        super().__init__()
        self.render_mode = render
        self.wait_time = wait_time
        self.game_mode = game_mode

        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        self.pygame_initialized = False
        self.rows = 6
        self.cols = 7

        # 0 = empty, 1 = player, -1 = opponent
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8
        )

        self.action_space = gym.spaces.Discrete(self.cols)

        self.board = None
        self.current_player = 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        if self.render_mode:
            self.render()
        return self.board.copy(), {}
    
    # get valid moves at the current state
    def get_valid_moves(self, board=None):
        if board is None:
            board = self.board
        return [c for c in range(self.cols) if board[0][c] == 0]


    def drop_piece(self, board, col, player):
        new_board = board.copy()
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                return new_board
        return new_board

    def step(self, action):
        # Invalid column or full column
        if action not in self.get_valid_moves():
            if self.render_mode:
                self.render()
            return self.board.copy(), -10.0, True, False, {"illegal_move": True}

        # Apply move
        self.board = self.drop_piece(self.board, action, self.current_player)

        # Win check
        if self.check_win(self.current_player):
            if self.render_mode:
                self.render()
                winner = "Player 1 Wins!" if self.current_player == 1 else "Player 2 Wins!"
                self._render_end_screen(winner)
            return self.board.copy(), 1.0, True, False, {"winner": self.current_player}

        # Draw check
        if len(self.get_valid_moves()) == 0:
            if self.render_mode:
                self.render()
                self._render_end_screen("Draw!")
            return self.board.copy(), 0.0, True, False, {"draw": True}

        # Switch player turn
        self.current_player *= -1

        # Render
        if self.render_mode:
            self.render()

        return self.board.copy(), 0.0, False, False, {}

    # Win detection
    def check_win(self, p):
        b = self.board

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(b[r, c:c+4] == p):
                    return True

        # Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if np.all(b[r:r+4, c] == p):
                    return True

        # Diagonal down-right
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(b[r+i][c+i] == p for i in range(4)):
                    return True

        # Diagonal up-right
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(b[r-i][c+i] == p for i in range(4)):
                    return True

        return False
    

    def get_human_move(self):
        """Wait for the human to click a column. Returns column index."""
        if not self.render_mode:
            raise RuntimeError("Human play requires render_mode=True")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x = event.pos[0]
                    col = x // self.square_size
                    # If the move is valid, return it
                    if col in self.get_valid_moves():
                        return col
                    else:
                        print("Invalid column. Try again.")
    

    def render(self):
        if not self.render_mode:
            return 

        if not self.pygame_initialized:
            pygame.init()

            self.square_size = 80
            self.width = self.cols * self.square_size
            self.height = (self.rows + 1) * self.square_size

            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Connect 4")

            self.font = pygame.font.SysFont("Times New Roman", 24)
            self.pygame_initialized = True

        # Draw background
        self.screen.fill(self.BLUE)

        # Draw board slots
        for r in range(self.rows):
            for c in range(self.cols):

                # Blue board rectangle
                pygame.draw.rect(
                    self.screen,
                    self.BLUE,
                    (c * self.square_size,
                     (r + 1) * self.square_size,
                     self.square_size,
                     self.square_size)
                )

                # Piece
                piece = self.board[r][c]
                color = self.BLACK
                if piece == 1:
                    color = self.RED
                elif piece == -1:
                    color = self.YELLOW

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        int(c * self.square_size + self.square_size / 2),
                        int((r + 1) * self.square_size + self.square_size / 2)
                    ),
                    self.square_size // 2 - 5
                )

        pygame.display.update()

        if self.wait_time > 0:
            pygame.time.wait(self.wait_time)


    def _render_end_screen(self, message):
        """Show final end-game message on screen."""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0)) 
        self.screen.blit(overlay, (0,0))

        font = pygame.font.SysFont("Times New Roman", 60, bold=True)
        text = font.render(message, True, (255, 255, 255))
        rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, rect)

        pygame.display.update()

        pygame.time.wait(10000)


# Example usage
if __name__ == "__main__":
    env = Connect4Env(render=True, wait_time=500)
    s, _ = env.reset()

    pygame.time.wait(5000)

    done = False
    while not done:
        action = np.random.choice(env.get_valid_moves())
        s, r, done, _, info = env.step(action)
