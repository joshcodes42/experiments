import game
import random
X, O, EMPTY = game.X, game.O, game.EMPTY

class TemporalDifferenceAgent:
    def __init__(self, alpha = 0.1, epsilon = 0.1, player = X):
        self.value_table = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.player = player

    def value(self, board) -> float:
        if board not in self.value_table:
            outcome = game.outcome(board)
            if outcome == self.player:
                self.value_table[board] = 1
            elif outcome == -self.player or outcome == EMPTY:
                self.value_table[board] = 0
            else:
                self.value_table[board] = 0.5

        return self.value_table[board]
                
    def choose(self, board) -> tuple[int, bool]:
        to_explore = random.random() <= self.epsilon
        legal_moves = game.legal_moves(board)
        best_move = -1

        if not to_explore: # greedy
            best_val = -1

            for move in legal_moves:
                val = self.value(game.play(board, move, self.player))
                if val > best_val:
                    best_move = move
                    best_val = val
        else:
            best_move = random.choice(legal_moves)

        return best_move, to_explore

    def update(self, prev_board, next_board):
        self.value_table[prev_board] += self.alpha * (self.value(next_board) - self.value(prev_board))

