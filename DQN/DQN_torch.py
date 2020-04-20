#Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import gym
import numpy as np
from common.ReplayBuffer import Buffer
from common.Saver import Saver

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = (self.fc3(x))

        return output

    def q_value(self, x):
        pass

    def best_action(self, x):
        pass

class DQN:
    def __init__(self, state_dim, action_dim, save=False, load=False, batch_size=100, gamma=0.99, learning_rate=0.001, epsilon=0.2, training_start=200, copy_iter=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon
        self.training_start = training_start
        self.copy_iter = copy_iter

        self.network = Network(self.state_dim, self.action_dim).to(self.device)
        self.target_network = Network(self.state_dim, self.action_dim).to(self.device)

        self.optim = optim.Adam(self.network.parameters(), lr=self.lr)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def get_action(self, state):
        if np.random.random() < self.eps:
            return torch.tensor(np.random.randint(low=0, high=self.action_dim), device=self.device, dtype=torch.float32)
        else:
            return self.network.best_action(state)[0]

    def train(self, s, a, r, ns, d):
        pass

    def run(self):
        pass





if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")
    # env = gym.make("Acrobot-v1")
    # env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    dtype = torch.FloatTensor
    s = env.reset()
    s = np.expand_dims(s, axis=1)
    s = torch.from_numpy(s).type(torch.float32).view(1, -1)#so, to use nn.Module, use tensor and view (1, -1)?
    network = Network(state_dim, action_dim)
    print(network(s))




