#Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import numpy as np
from common.ReplayBuffer import Buffer


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

class DQN:
    def __init__(self, state_dim, action_dim, save=False, load=False, batch_size=100, gamma=0.99, learning_rate=0.001, epsilon=0.2, training_start=200, copy_iter=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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
        self.buffer = Buffer(self.batch_size)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.copy_weight(self.network, self.target_network)
        self.target_network.eval()

    def copy_weight(self, network, target_network):
        target_network.load_state_dict(network.state_dict())


    def get_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=1)
                state = torch.from_numpy(state).type(torch.float32).view(1, -1)
                action = self.network(state).data.max(1)[1].cpu()#why max(1)[1]? Answer: torch.return_types.max(values=tensor([0.0348]), indices=tensor([0])) So [0] is max, [1] is argmax
            return action.numpy()[0]

    def train(self, s, a, r, ns, d):
        s = torch.from_numpy(s).type(torch.float32)
        a = torch.from_numpy(a).type(torch.long)
        r = torch.from_numpy(r).type(torch.long)
        ns = torch.from_numpy(ns).type(torch.float32)
        d = torch.from_numpy(d).type(torch.long)

        target_q = self.target_network(ns).max(1)[0].detach().view(-1, 1)
        target_value = r + self.gamma*(1-d)*target_q
        #print(r.size(), (1-d).size(), target_q.size())
        selected_values = self.network(s).gather(1, a)
        loss = self.loss(selected_values, target_value)
        self.optimizer.zero_grad()
        loss.backward()

        #for param in self.network.parameters():
        #    param.grad.data.clamp_(-1, 1)

        self.optimizer.step()





    def run(self):

        episode = 0
        total_step = 0

        while True:
            observation = env.reset()
            done = False
            episode += 1
            episode_reward = 0
            local_step = 0

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                action = self.get_action(observation)
                next_observation, reward, done, _ = env.step(action)

                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if total_step > self.training_start:
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

                    if total_step % self.copy_iter == 0:
                        self.copy_weight(self.network, self.target_network)

                if done:
                    print("episode: {}, reward: {},  total_step: {}".format(episode, episode_reward, total_step))





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
    '''
    dtype = torch.FloatTensor
    s = env.reset()
    s = np.expand_dims(s, axis=1)
    s = torch.from_numpy(s).type(torch.float32).view(1, -1)#so, to use nn.Module, use tensor and view (1, -1)?
    
    network = Network(state_dim, action_dim)
    print(network(s))
    '''

    dqn = DQN(state_dim, action_dim)
    dqn.run()





