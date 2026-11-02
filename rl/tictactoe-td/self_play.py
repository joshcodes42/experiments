import game
import agents
from tqdm import tqdm
import numpy as np
import time

def TD_vs_TD(starting_player: int, window_length: int = 100, verbose = False, seed = 42, alpha = 0.1, epsilon = 0.1):
    agent1 = agents.TemporalDifferenceAgent(game.X, seed = seed, alpha = alpha, epsilon = epsilon)
    agent2 = agents.TemporalDifferenceAgent(game.O, seed = seed + 1, alpha = alpha, epsilon = epsilon)

    outcomes = []

    # TRAIN LOOP
    for i in tqdm(range(1, 1000000 + 1)):
        if i % window_length == 0 and verbose:
            X_rate = np.mean(np.array(outcomes) == game.X)
            O_rate = np.mean(np.array(outcomes) == game.O)
            tie_rate = 1 - X_rate - O_rate

            print(f"\nTraining iteration {i}: X_rate: {X_rate} | O_rate: {O_rate} | Draw Rate: {tie_rate:.02f}")

        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        player = starting_player
        prev_board1 = board
        prev_board2 = board

        while True:
            if player == game.X:
                # print(board, " its X turn")
                TD_move, was_explore = agent1.choose(board)
                board = game.play(board, TD_move, player)

                if not was_explore:
                    agent1.update(prev_board1, board)

                out = game.outcome(board)
                if out != None:
                    agent2.update(prev_board2, board)
                    outcomes = outcomes[-window_length + 1:] + [out]
                    break

                prev_board1 = board

            else:
                # print(board, " its O turn")
                TD_move, was_explore = agent2.choose(board)
                board = game.play(board, TD_move, player)

                if not was_explore:
                    agent2.update(prev_board2, board)

                out = game.outcome(board)
                if out != None:
                    agent1.update(prev_board1, board)
                    outcomes = outcomes[-window_length + 1:] + [out]
                    break

                prev_board2 = board

            player = -player

        agent1.alpha *= 0.9999977
        agent2.alpha *= 0.9999977
        agent1.epsilon *= 0.9999977
        agent2.epsilon *= 0.9999977

    return agent2

if __name__ == "__main__":
    def print_board():

        mapping = {
            0: "-",
            1: "X",
            -1: "O"
        }
        
        for i in range(3):
            print(mapping[board[i * 3]], mapping[board[i * 3 + 1]], mapping[board[i * 3 + 2]])

    print("AI is training hella hard rn to beat yo ass")
    final_agent = TD_vs_TD(game.X, seed = 42, alpha = 0.01, epsilon = 0.1, verbose=False)
    final_agent.epsilon = 0
    while True:
            
        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)

        while True:
            print_board()

            while True:
                player_move = int(input("Make a move!\n"))
                if player_move in game.legal_moves(board):
                    break

            board = game.play(board, player_move, game.X)
            print_board()
            
            out = game.outcome(board)
            if out == 0:
                print("TIE!")
                break
            elif out == game.O:
                print("the ai won haha you loser")
                break
            elif out == game.X:
                print("CONGRATS YOU WON!!")
                break
            print("AI is making a move...")
            time.sleep(2)
            board = game.play(board, final_agent.choose(board)[0], game.O)

            out = game.outcome(board)
            if out == 0:
                print("TIE!")
                break
            elif out == game.O:
                print("the ai won haha you loser")
                break
            elif out == game.X:
                print("CONGRATS YOU WON!!")
                break

        print("loading new game..")
        time.sleep(5)