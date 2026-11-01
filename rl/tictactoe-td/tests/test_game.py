import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from game import *
import numpy as np
from tqdm import tqdm
import random

def random_game():
    board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    player = X

    while True:
        player_move = random.choice(legal_moves(board))
        board = play(board, player_move, player)

        o = outcome(board)
        if o != None:
            return o

        player = -player

def many_random_games(n: int):
    rates = {
        X: 0,
        O: 0,
        EMPTY: 0
    }

    for i in tqdm(range(n)):
        o = random_game()
        rates[o] += 1

    rates = np.array(list(rates.values()))
    rates = rates / np.sum(rates)
    return rates

if __name__ == "__main__":
    print(many_random_games(100000))