import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        return self.max_action * torch.tanh(self.layer3(a))

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinCritic, self).__init__()
        
        # Q1 architecture
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        
        # Q2 architecture (Identical but independent)
        self.layer4 = nn.Linear(state_dim + action_dim, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        q1 = self.layer3(x1)
        
        x2 = F.relu(self.layer4(xu))
        x2 = F.relu(self.layer5(x2))
        q2 = self.layer6(x2)
        
        return q1, q2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        return self.layer3(x1)
