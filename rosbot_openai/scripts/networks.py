import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, in_dim):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.linear_vel = nn.Linear(512, 1)
        self.ang_vel = nn.Linear(512,1)

    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        lin_vel_activation = F.sigmoid(self.linear_vel(activation3))
        ang_vel_activation = F.tanh(self.ang_vel(activation3))
        #Das concatenate funktioniert irgendwie falsch wenn man mehrere observations reingibt. es konkatteniert dann alle lin und ang vels.
        # -> Doppelt so gross wie es eigentlich sein sollte
        output = torch.cat((lin_vel_activation, ang_vel_activation), dim=-1)
        #print("Actor Network Output: %s", output)

        return output

class CriticNetwork(nn.Module):
    def __init__(self, in_dim):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512,1)

    def forward(self, obs):
        #if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        output = self.layer4(activation3)

        return output


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 512)
        self.layer2 = nn.Linear(512,512)
        self.layer3 = nn.Linear(512,out_dim)

    def forward(self, obs):
        #if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float)

        #x = self.flatten(obs)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output