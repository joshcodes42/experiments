import game
import agents
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def TD_vs_random(num_episodes: int, starting_player: int, window_length: int = 100, eval_len: int = 2000, verbose = False, seed = 42, alpha = 0.1, epsilon = 0.1):
    TD_agent = agents.TemporalDifferenceAgent(game.X, seed = seed, alpha = alpha, epsilon= epsilon)
    random_agent = agents.RandomAgent(game.O, seed = seed + 1)

    outcomes = []
    train_outcomes = []

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
                    train_outcomes.append(out)
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
                    train_outcomes.append(out)
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
    return np.array(train_outcomes), np.array(eval_outcomes)

def random_vs_random(num_episodes: int, starting_player: int, window_length: int = 100, eval_len: int = 2000, verbose = False, seed = 42):
    random_agent1 = agents.RandomAgent(game.X, seed = seed)
    random_agent2 = agents.RandomAgent(game.O, seed = seed + 1)

    eval_outcomes = []
    for i in range(eval_len):
        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        player = starting_player
        prev_board = board

        while True:
            if player == game.X:
                move = random_agent1.choose(board)
                board = game.play(board, move, player)

                out = game.outcome(board)
                if out != None:
                    eval_outcomes.append(out)
                    break

            else:
                random_move = random_agent2.choose(board)
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
    return np.array(eval_outcomes)

if __name__ == "__main__":
    def rolling_avg(data, window_size):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    trainout1_all = []
    trainout2_all = []
    trainout3_all = []

    for s in [42, 24, 67, 104, 777, 2049, 20000, 2933895, 3952458243]:
        baseline = random_vs_random(20000, game.X, seed = s)    
        trainout1, evalout1 = TD_vs_random(20000, game.X, seed = s, alpha = 0.01, epsilon = 0.1)
        trainout2, evalout2 = TD_vs_random(20000, game.X, seed = s, alpha = 0.1, epsilon = 0.1)
        trainout3, evalout3 = TD_vs_random(20000, game.X, seed = s, alpha = 0.5, epsilon = 0.1)

        win_rates1 = rolling_avg(np.where(trainout1 == game.X, 1, 0), 500)
        win_rates2 = rolling_avg(np.where(trainout2 == game.X, 1, 0), 500)
        win_rates3 = rolling_avg(np.where(trainout3 == game.X, 1, 0), 500)

        trainout1_all.append(win_rates1)
        trainout2_all.append(win_rates2)
        trainout3_all.append(win_rates3)

    trainout1_all = np.array(trainout1_all).mean(axis=0)
    trainout2_all = np.array(trainout2_all).mean(axis=0)
    trainout3_all = np.array(trainout3_all).mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(trainout1_all)), trainout2_all)
    ax.set_title("Alpha = 0.1, TD vs Random")
    ax.set_ylabel("Win rate")
    ax.set_xlabel("Training episodes")

    fig.savefig(
        'trainout2.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False
    )

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(trainout1_all)), trainout1_all)
    ax.plot(np.arange(len(trainout1_all)), trainout2_all)
    ax.plot(np.arange(len(trainout1_all)), trainout3_all)
    ax.legend(["Alpha = 0.01", "Alpha = 0.1", "Alpha = 0.5"])

    ax.set_title("Increasing levels of Alpha, TD vs Random")
    ax.set_ylabel("Win rate")
    ax.set_xlabel("Training episodes")

    fig.savefig(
        'all_trainouts.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False
    )