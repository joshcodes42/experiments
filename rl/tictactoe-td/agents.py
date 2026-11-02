from abc import ABC, abstractmethod
import game
import random
X, O = game.X, game.O

class Agent(ABC):
    def __init__(self, player):
        self.player = player

    @abstractmethod
    def choose(self, board):
        pass

class TemporalDifferenceAgent(Agent):
    def __init__(self, player, alpha = 0.1, epsilon = 0.1, seed = 42):
        super().__init__(player)
        self.value_table = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.random = random.Random(seed)

    def value(self, board) -> float:
        if board not in self.value_table:
            outcome = game.outcome(board)
            if outcome == self.player:
                self.value_table[board] = 1
            elif outcome == -self.player or outcome == 0:
                self.value_table[board] = 0
            else:
                self.value_table[board] = 0.5

        return self.value_table[board]
                
    def choose(self, board):
        to_explore = self.random.random() < self.epsilon
        legal_moves = game.legal_moves(board)
        best_move = -1

        if not to_explore: # greedy
            best_val = -1
            self.random.shuffle(legal_moves)
            for move in legal_moves:
                val = self.value(game.play(board, move, self.player))
                if val > best_val:
                    best_move = move
                    best_val = val
        else:
            best_move = self.random.choice(legal_moves)

        return best_move, to_explore

    def update(self, prev_board, next_board):
        prev_value = self.value(prev_board)
        next_value = self.value(next_board)

        self.value_table[prev_board] += self.alpha * (next_value - prev_value)

class RandomAgent(Agent):
    def __init__(self, player, seed):
        super().__init__(player)
        self.random = random.Random(seed)

    def choose(self, board):
        return self.random.choice(game.legal_moves(board))