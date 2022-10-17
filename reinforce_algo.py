import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

def action_select(state, model):
  state = torch.from_numpy(state).float().unsqueeze(0)
  pred = model(state)
  predcat = Categorical(pred)
  action = predcat.sample()
  logprob = predcat.log_prob(action)
  model.saved_log_probs.append(logprob)
  return action

def action_select_OOS(state, model):
  state = torch.from_numpy(state).float().unsqueeze(0)
  pred = model(state)
  action = torch.argmax(pred)
  return action

# This function actually applies the gradient
def refine(losses, optimizer):
  optimizer.zero_grad()
  tot_losses = torch.cat(losses).sum()
  tot_losses.backward()
  optimizer.step()