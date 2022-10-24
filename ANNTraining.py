import numpy as np
import random
from itertools import count
from statistics import median, mean, stdev
from collections import Counter
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import env # imports the grid and transition probabilities functions
from sampleWidx import sampleWidx
from NeuralNet import NeuralNet
import reinforce_algo

# without batches (not really needed)
def no_batch_train(initial_value, goal, time, increments, nepochs, value_grid, means_grid, stds_grid, probs_grid, Model, opt):
    n_success = 0
    neps = 0
    stopper = 0 # for the stopping condition

    # to standardize inputs
    t_sdize = np.linspace(0, time - 1, (time*increments))
    t_mu, t_std = t_sdize.mean(), t_sdize.std()
    w_mu, w_std = value_grid.mean(), value_grid.std()

    for _ in range(nepochs):
        W0 = initial_value
        idx = np.where(value_grid == W0)[0]
        returns = []

        for i in range(time):
            t = (i - t_mu) / t_std # standardized time period
            wstzed = (value_grid[idx] - w_mu) / w_std # standardized wealth
            state = np.array([t, float(wstzed)])
            action = reinforce_algo.action_select(state, Model)
            idx = sampleWidx(idx, value_grid, action, means_grid, stds_grid, probs_grid)
            if i == (time - 1) and value_grid[idx] >= goal:
                r = 1  # return if path is successful
            else:
                r = 0
            returns.append(r)

        returns = torch.tensor(returns)
        returns = torch.flip(returns, dims=(0,))
        returns = torch.cumsum(returns, dim=0)
        returns = torch.flip(returns, dims=(0,))

        policy_loss = []

        for x1, x2 in zip(returns, Model.saved_log_probs):
            policy_loss.append(-x1 * x2)

        reinforce_algo.refine(policy_loss, opt)

        Model.saved_log_probs = []

        n_success = n_success + r
        neps = neps + 1
        train_success = n_success / neps

        if neps % (nepochs/100) == 0:
            print(f'Completion rate: {100*neps/nepochs}% & success in training: {train_success}')

        if neps % (nepochs/10) == 0 and train_success < stopper:
            break
        elif neps % (nepochs/10) == 0 and train_success >= stopper:
            stopper = train_success


# with batches
def train_one_batch(batch_size, Model, optimizer, T, increments, value_grid, initial_value, goal, probs_grid,
                    means_grid, stds_grid):
    t_sdize = np.linspace(0, T - 1, (T * increments))
    t_mu, t_std = t_sdize.mean(), t_sdize.std()
    w_mu, w_std = value_grid.mean(), value_grid.std()

    returns = []
    batch_losses = []

    n_success = 0

    for _ in range(batch_size):
        W0 = initial_value
        idx = np.where(value_grid == W0)[0]

        for i in range(T):
            t = (i - t_mu) / t_std
            wszted = (value_grid[idx] - w_mu) / w_std
            state = np.array([t, float(wszted)])
            action = reinforce_algo.action_select(state, Model)
            idx = sampleWidx(idx, value_grid, action, means_grid, stds_grid, probs_grid)
            if i == (T - 1) and value_grid[idx] >= goal:
                r = 1  # return if path is successful
                n_success = n_success + 1
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

    return n_success


def train_model(nepochs, batch_size, Model, optimizer, T, increments, value_grid, initial_value, goal,
                probs_grid, means_grid, stds_grid):
    stopper = 0
    for i in range(nepochs):
        # print(f"Epoch {i+1}")
        n_success = train_one_batch(batch_size, Model, optimizer, T, increments, value_grid, initial_value,
                                    goal, probs_grid, means_grid, stds_grid)

        if (i + 1) % (nepochs / 100) == 0:
            print(f'Completion rate: {100 * (i + 1) / nepochs}% & success in training: {n_success / batch_size}')

        # if (i+1) % (nepochs/10) == 0 and (n_success/batch_size) < stopper and (i+1)/nepochs > 0.6:
        #  break
        # elif (i+1) % (nepochs/10) == 0 and (n_success/batch_size) >= stopper:
        #  stopper = (n_success/batch_size)

    print("---------------")
    print("training complete")
