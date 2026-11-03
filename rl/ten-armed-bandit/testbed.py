import numpy as np
import random
import matplotlib.pyplot as plt

def make_problem(rng, k = 10, runs = 2000):
    return rng.normal(loc = 0.0, scale = 1.0, size = (runs, k))

def pull(rng, q_star):
    return rng.normal(loc = q_star, scale = 1.0)

def do_runs(rng, epsilon, steps = 1000, k = 10, runs = 2000):
    all_do_greedy = rng.random(size=(runs, steps)) >= epsilon
    q_star = make_problem(rng, k = k, runs = runs)

    Q = np.zeros(shape=(runs, k))
    N = np.zeros(shape=(runs, k))
    for i in range(steps): # for each run, at this time step
        rewards = pull(rng, q_star)

        do_greedy = all_do_greedy[:, i]

        greedy_move = np.argmax(Q + np.random.random(Q.shape) * 1e-9, axis=1)
        exploratory_move = rng.choice(np.arange(k), size = runs)

        move = greedy_move * do_greedy + exploratory_move * (1 - do_greedy)

        N[np.arange(runs), move] += 1
        Q[np.arange(runs), move] += (1/N[np.arange(runs), move]) * (rewards[np.arange(runs), move] - Q[np.arange(runs), move])

    print(rewards.shape)
    # return np.mean(rewards, axis=0), np.mean(took_optimal, axis=0)

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    r1, o1 = do_runs(rng, epsilon=0)
    r2, o2 = do_runs(rng, epsilon=0.01)
    r3, o3 = do_runs(rng, epsilon=0.1)

    print(r1.shape)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(r1.shape[1], r1)
    axs[0].plot(np.arange(len(r2)), r2)
    axs[0].plot(np.arange(len(r3)), r3)
    axs[0].set_title("Average Reward over Time")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Avg Reward")

    plt.show()