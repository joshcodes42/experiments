import numpy as np

## Constants
X, O = 1, -1

WIN_POSITIONS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6)
)

def legal_moves(board) -> list[int]:
    return [i for i in range(9) if board[i] == 0]

def play(board, square, player) -> tuple:
    board = list(board)
    board[square] = player
    return tuple(board)

def outcome(board) -> int | None:
    for one, two, three in WIN_POSITIONS:
        if board[one] != 0 and board[one] == board[two] and board[two] == board[three]:
            return board[one]

    if 0 in board:
        return None

    return 0