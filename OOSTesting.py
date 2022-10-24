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

def OOS(initial_value, goal, time, increments, Model, nepochs, value_grid, means_grid, stds_grid, probs_grid):
    n_success = 0
    neps = 0

    # to standardize inputs
    t_sdize = np.linspace(0, time - 1, (time * increments))
    t_mu, t_std = t_sdize.mean(), t_sdize.std()
    w_mu, w_std = value_grid.mean(), value_grid.std()

    for j in range(nepochs):
        W0 = initial_value
        idx = np.where(value_grid == W0)[0]

        for i in range(time):
            t = (i - t_mu) / t_std
            wstzed = (value_grid[idx] - w_mu) / w_std
            state = np.array([t, float(wstzed)])
            action = reinforce_algo.action_select_OOS(state, Model)
            idx = sampleWidx(idx, value_grid, action, means_grid, stds_grid, probs_grid)

        if value_grid[idx] >= goal:
            n_success = n_success + 1

        neps = neps + 1

        success_rate = n_success / neps
    print(success_rate)