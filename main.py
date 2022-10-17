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

portfolio_means = np.array([0.15, 0.10, 0.05])
portfolio_stds = np.array([0.3, 0.15, 0.07])
len_grid = 100
min_value_grid, max_Value_grid = 0.02, 3
h = 1 # time increments
w_array = env.wealth_grid(len_grid, min_value_grid, max_Value_grid, h) # wealth grid
probs_array = env.transition_probs(portfolio_means, portfolio_stds, w_array, 1)

n_inputs = 2 # time and wealth
n_weights = 8 # for neural network
n_actions = len(portfolio_means)

# Checking device
if torch.cuda.is_available():
    Device = "cuda"
else:
    Device = "cpu"

model = NeuralNet(n_inputs, n_weights, n_actions).to(device=Device)
# Defining the optimizer
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

# training
def main(initial_value, goal, time, increments, nepochs, value_grid, means_grid, stds_grid, probs_grid):
    n_success = 0
    neps = 0

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
            action = reinforce_algo.action_select(state, model)
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

        for x1, x2 in zip(returns, model.saved_log_probs):
            policy_loss.append(-x1 * x2)

        reinforce_algo.refine(policy_loss, optimizer)

        model.saved_log_probs = []

        n_success = n_success + r
        neps = neps + 1
        train_success = n_success / neps

        if neps % (nepochs/100) == 0:
            print(f'Completion rate: {100*neps/nepochs}% & success in training: {train_success}')


# Out-of-sample-testing
def OOS(initial_value, goal, time, increments, nepochs, value_grid, means_grid, stds_grid, probs_grid):
    n_success = 0
    neps = 0

    # to standardize inputs
    t_sdize = np.linspace(0, time - 1, (time * increments))
    t_mu, t_std = t_sdize.mean(), t_sdize.std()
    w_mu, w_std = value_grid.mean(), value_grid.std()

    paths = np.zeros((nepochs, time+1))
    paths[:, 0] = initial_value

    for j in range(nepochs):
        W0 = initial_value
        idx = np.where(value_grid == W0)[0]

        for i in range(time):
            t = (i - t_mu) / t_std
            wstzed = (value_grid[idx] - w_mu) / w_std
            state = np.array([t, float(wstzed)])
            action = reinforce_algo.action_select_OOS(state, model)
            idx = sampleWidx(idx, value_grid, action, means_grid, stds_grid, probs_grid)
            paths[j, i+1] = value_grid[idx]

        if value_grid[idx] >= goal:
            n_success = n_success + 1

        neps = neps + 1

        success_rate = n_success / neps
        print(success_rate)
    return(paths)


main(1,1.2,2,1,100000,w_array,portfolio_means,portfolio_stds,probs_array)

paths = OOS(1,1.2,2,1,100000,w_array,portfolio_means,portfolio_stds,probs_array)


###########  Plots  ################
a1 = []
a2 = []
a0 = []
w_mu, w_std = w_array.mean(), w_array.std()
done = False
w=0.8
while w < 1.4:
    t = 1
    w_pr = (w-w_mu)/w_std
    state = np.array([t, float(w_pr)])
    state = torch.from_numpy(state).float().unsqueeze(0)
    pred = model(state)
    a0.append(float(pred[0][0]))
    a1.append(float(pred[0][1]))
    a2.append(float(pred[0][2]))

    w = w + 0.01

a0, a1, a2 = np.array(a0), np.array(a1), np.array(a2)
wealths = np.linspace(0.8, 1.3, len(a0))
plt.figure()
plt.plot(wealths, a0)
