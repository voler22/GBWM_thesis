import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, num_inputs, num_weights, num_outputs):
        # intitialisation
        super(NeuralNet, self).__init__()

        self.affine1 = nn.Linear(num_inputs, num_weights)
        #self.dropout = nn.Dropout(p = 0.6)
        self.affine2 = nn.Linear(num_weights, num_outputs)

        self.saved_log_probs = []

    def forward(self, position):
        position = self.affine1(position)
        #position = self.dropout(position)
        position = F.relu(position)
        action_probs = self.affine2(position)

        action_probs = F.softmax(action_probs, dim=1)

        return action_probs