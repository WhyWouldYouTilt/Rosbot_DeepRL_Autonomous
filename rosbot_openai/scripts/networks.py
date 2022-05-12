import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, in_dim):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.linear_vel = nn.Linear(128, 1)
        self.ang_vel = nn.Linear(128,1)

    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).cuda()
        activation1 = F.relu(self.layer1(obs)).cuda()
        activation2 = F.relu(self.layer2(activation1)).cuda()
        activation3 = F.relu(self.layer3(activation2)).cuda()
        lin_vel_activation = F.sigmoid(self.linear_vel(activation3)).cuda()
        ang_vel_activation = F.tanh(self.ang_vel(activation3)).cuda()
        output = torch.cat((lin_vel_activation, ang_vel_activation), dim=-1).cuda()
        #print("Actor Network Output: %s", output)

        return output

class CriticNetwork(nn.Module):
    def __init__(self, in_dim):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128,1)

    def forward(self, obs):
        #if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float).cuda()
        activation1 = F.relu(self.layer1(obs)).cuda()
        activation2 = F.relu(self.layer2(activation1)).cuda()
        activation3 = F.relu(self.layer3(activation2)).cuda()
        output = self.layer4(activation3).cuda()

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