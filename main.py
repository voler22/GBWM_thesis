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
from ANNTraining import no_batch_train, train_one_batch, train_model
from OOSTesting import OOS

portfolio_means = np.array([0.15, 0.10, 0.05])
portfolio_stds = np.array([0.3, 0.15, 0.07])
len_grid = 100
min_value_grid, max_Value_grid = 0.02, 3
h = 1 # time increments

w_array = env.wealth_grid(len_grid, min_value_grid, max_Value_grid, h) # wealth grid
probs_array = env.transition_probs(portfolio_means, portfolio_stds, w_array, h)

######### To train model ###########

n_inputs = 2 # time and wealth
n_weights = 8 # for neural network
n_actions = len(portfolio_means)

train_nepochs = 1000

# Checking device
if torch.cuda.is_available():
    Device = "cuda"
else:
    Device = "cpu"


model = NeuralNet(n_inputs, n_weights, n_actions).to(device = Device)

optimizer = optim.Adam(model.parameters(), lr = 1e-2)
nepochs = 1
batch_size = 1000
T = 2
h = 1
W0 = 1
G = 1.2

# OOS
n_OOS_paths = 1000

##################################

# training no batch
no_batch_train(W0,G,T,h,train_nepochs,w_array,portfolio_means,portfolio_stds,probs_array,model,optimizer)

# training with batches
train_model(nepochs, batch_size, model, optimizer, T, h, w_array, W0, G, probs_array, portfolio_means, portfolio_stds)

# Out-of-sample-testing
OOS(W0, G, T, h, model, n_OOS_paths, w_array, portfolio_means, portfolio_stds, probs_array)



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
