import torch
import torch.nn as nn
import torch.nn.functional as F

'''class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        state = None
        output, state = self.rnn(x, state)
        x = F.relu(self.fc1(output[:,-1,:]))
        x = self.fc2(x)
        return x'''

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x