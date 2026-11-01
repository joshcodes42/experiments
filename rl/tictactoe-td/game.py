import numpy as np

## Constants
EMPTY, X, O = 0, 1, -1

def legal_moves(board) -> list[int]:
    return [i for i in range(9) if board[i] == EMPTY]

def play(board, square, player) -> tuple:
    board = list(board)
    board[square] = player
    return tuple(board)

def outcome(board) -> int | None:
    board = np.array(board).reshape(3, 3)
    row_summed = np.sum(board, axis = 1)
    if ((row_summed == 3) | (row_summed == -3)).any():
        return int(row_summed[np.argmax(np.abs(row_summed))] / 3)

    col_summed = np.sum(board, axis = 0)
    if ((col_summed == 3) | (col_summed == -3)).any():
        return int(col_summed[np.argmax(np.abs(col_summed))] / 3)

    diag1_sum = np.trace(board)
    diag2_sum = np.trace(np.flip(board, axis=1))

    if diag1_sum == 3 or diag1_sum == -3:
        return int(diag1_sum / 3)

    if diag2_sum == 3 or diag2_sum == -3:
        return int(diag2_sum / 3)

    return None