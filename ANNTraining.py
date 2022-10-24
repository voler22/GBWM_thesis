import numpy as np
import random
from itertools import count
from statistics import median, mean, stdev
from collections import Counter
from scipy.stats import norm

import env
import reinforce_algo
from OOSTesting import OOS, OOS_grid


def train_one_batch(batch_size, Model, optimizer, T, increments, initial_value, goal, means_grid, stds_grid):
    t_sdize = np.linspace(0, T - 1, (T * increments))
    t_mu = t_sdize.mean()

    returns = []
    batch_losses = []

    for _ in range(batch_size):
        W, W_mean = initial_value, initial_value
        for i in range(T):
            t = (i - t_mu)
            W_nzed = (W - W_mean)
            state = np.array([t, float(W_nzed)])
            action = reinforce_algo.action_select(state, Model)
            W_mean = W*np.exp(means_grid[action])
            W = env.traj(W, action, means_grid, stds_grid, h)

            if i == (T - 1) and W >= goal:
                r = 1
            else:
                r = 0

        for _ in range(T):
            returns.append(r)

    for x1, x2 in zip(returns, Model.saved_log_probs):
        batch_losses.append(-x1 * x2)

    optimizer.zero_grad()
    avg_loss = T * torch.cat(batch_losses).mean()
    avg_loss.backward()
    optimizer.step()

    del Model.saved_log_probs[:]


def train_model(nepochs, verifier_nepochs, nepochs_OOS, batch_size, Model, optimizer, T, increments, initial_value, goal,
                means_grid, stds_grid):
    pstar = 0
    for i in range(nepochs):
        train_one_batch(batch_size, Model, optimizer, T, increments, initial_value, goal, means_grid, stds_grid)

        if nepochs % verifier_nepochs == 0:
            p = OOS(initial_value, goal, increments, Model, nepochs_OOS, means_grid, stds_grid)
            if p > pstar:
                best_model = Model

        if nepochs % (nepochs / 100):
            print(f'Completion rate: {(i+1)/nepochs * 100}%  ***  p* = {pstar}')

    print("------------------")
    print("Training complete")
