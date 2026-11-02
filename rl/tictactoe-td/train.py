import game
import agents
from tqdm import tqdm
import numpy as np

def TD_vs_random(num_episodes: int, starting_player: int, window_length: int = 100, eval_len: int = 2000, verbose = False):
    TD_agent = agents.TemporalDifferenceAgent(game.X)
    random_agent = agents.RandomAgent(game.O, seed = 42)

    outcomes = []

    # TRAIN LOOP
    for i in tqdm(range(1, num_episodes + 1)):
        if i % window_length == 0 and verbose:
            X_rate = np.mean(np.array(outcomes) == game.X)
            O_rate = np.mean(np.array(outcomes) == game.O)
            tie_rate = 1 - X_rate - O_rate

            print(f"\nTraining iteration {i}: X_rate: {X_rate} | O_rate: {O_rate} | Draw Rate: {tie_rate:.02f}")

        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        player = starting_player
        prev_board = board

        while True:
            if player == game.X:
                # print(board, " its X turn")
                TD_move, was_explore = TD_agent.choose(board)
                board = game.play(board, TD_move, player)

                if not was_explore:
                    TD_agent.update(prev_board, board)

                out = game.outcome(board)
                if out != None:
                    outcomes = outcomes[-window_length + 1:] + [out]
                    break

                prev_board = board

            else:
                # print(board, " its O turn")
                random_move = random_agent.choose(board)
                board = game.play(board, random_move, player)

                out = game.outcome(board)
                if out != None:
                    TD_agent.update(prev_board, board)
                    outcomes = outcomes[-window_length + 1:] + [out]
                    break

            player = -player

    # EVAL LOOP
    TD_agent.epsilon = 0
    eval_outcomes = []
    for i in range(eval_len):
        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        player = starting_player
        prev_board = board

        while True:
            if player == game.X:
                TD_move, was_explore = TD_agent.choose(board)
                board = game.play(board, TD_move, player)

                out = game.outcome(board)
                if out != None:
                    eval_outcomes.append(out)
                    break

                prev_board = board

            else:
                random_move = random_agent.choose(board)
                board = game.play(board, random_move, player)

                out = game.outcome(board)
                if out != None:
                    eval_outcomes.append(out)
                    break

            player = -player

    X_rate = np.mean(np.array(eval_outcomes) == game.X)
    O_rate = np.mean(np.array(eval_outcomes) == game.O)
    tie_rate = 1 - X_rate - O_rate

    print(f"\n Frozen Evaluation: X_rate: {X_rate} | O_rate: {O_rate} | Draw Rate: {tie_rate:.02f}")


if __name__ == "__main__":
    TD_vs_random(20000, game.X, 500)