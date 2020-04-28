#Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions.normal as normal

import gym
import numpy as np
from common.ReplayBuffer import Buffer

class Q_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = (self.fc3(x))

        return output


class V_network(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = (self.fc3(x))

        return output


class Policy_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2*self.action_dim)

    def forward(self, x, deterministic=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))

        mu = x[:, :self.action_dim]
        sigma = torch.exp(torch.clamp(x[:, self.action_dim:], min=-20.0, max=2.0))

        distribution = normal.Normal(loc=mu, scale=sigma)
        sample_action = distribution.rsample()
        torch_mean = torch.tanh(mu)
        if deterministic == True:
            return torch_mean
        tanh_sample = torch.tanh(sample_action)

        return tanh_sample

    def log_pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))

        mu = x[:, :self.action_dim]
        sigma = torch.exp(torch.clamp(x[:, self.action_dim:], min=-20.0, max=2.0))

        distribution = normal.Normal(loc=mu, scale=sigma)
        sample_action = distribution.rsample()
        tanh_sample = torch.tanh(sample_action)

        log_prob = distribution.log_prob(sample_action + 1e-6)
        log_pi = log_prob - (torch.log(1 + 1e-6 - tanh_sample.pow(2)))

        return log_pi

    def mu_sigma(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))

        mu = x[:, :self.action_dim]
        sigma = torch.exp(torch.clamp(x[:, self.action_dim:], min=-20.0, max=2.0))

        return mu, sigma



class SAC:
    def __init__(self, state_dim, action_dim, max_action, min_action, save, load, batch_size=100, tau=0.995, learning_rate=0.0003, gamma=0.99, alpha=0.2, reward_scale=1, training_start=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.tau = tau
        self.lr = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.training_start = training_start

        self.actor = Policy_network(self.state_dim, self.action_dim)
        self.critic1 = Q_network(self.state_dim, self.action_dim)
        self.critic2 = Q_network(self.state_dim, self.action_dim)
        self.v_network = V_network(self.state_dim)
        self.target_v_network = V_network(self.state_dim)

        self.buffer = Buffer(self.batch_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=self.lr)

        self.value_loss = nn.MSELoss()
        self.critic1_loss = nn.MSELoss()
        self.critic2_loss = nn.MSELoss()

        self.copy_weight(self.v_network, self.target_v_network)

    def copy_weight(self, network, target_network):
        #target_network.load_state_dict(network.state_dict())
        for v1, v2 in zip(network.parameters(), target_network.parameters()):
            v2.data.copy_(v1.data)

    def soft_update(self, network, target_network):
        for v1, v2 in zip(network.parameters(), target_network.parameters()):
            v2.data.copy_(self.tau*v2.data + (1-self.tau)*v1.data)

    def train(self, s, a, r, ns, d):
        s = torch.from_numpy(s).type(torch.float32)
        a = torch.from_numpy(a).type(torch.float32)
        r = torch.from_numpy(r).type(torch.float32)
        ns = torch.from_numpy(ns).type(torch.float32)
        d = torch.from_numpy(d).type(torch.float32)

        min_aq = torch.min(self.critic1(torch.cat((s, self.actor(s)), dim=1)), self.critic2(torch.cat((s, self.actor(s)), dim=1)))

        target_v = (min_aq - self.alpha*self.actor.log_pi(s))
        v_loss = self.value_loss(self.v_network(s), target_v.detach())

        self.v_optimizer.zero_grad()
        v_loss.backward()

        target_q = r + self.gamma * (1 - d) * self.target_v_network(ns)

        critic1_loss = self.critic1_loss(self.critic1(torch.cat((s, a), dim=1)), target_q.detach())
        critic2_loss = self.critic2_loss(self.critic2(torch.cat((s, a), dim=1)), target_q.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()

        mu, sigma = self.actor.mu_sigma(s)
        output = mu + torch.normal(0, 1, tuple(sigma.size()))*sigma

        min_aq_rep = torch.min(self.critic1(torch.cat((s, output), dim=1)), self.critic2(torch.cat((s, output), dim=1)))
        actor_loss = (self.alpha*self.actor.log_pi(s) - min_aq).mean()
        #actor_loss = (self.actor.log_pi(s)*(self.actor.log_pi(s) - (min_aq - self.v_network(s)).detach())).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.v_optimizer.step()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        self.actor_optimizer.step()


        self.soft_update(self.v_network, self.target_v_network)





    def run(self):
        episode = 0
        total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            local_step = 0

            done = False
            observation = env.reset()

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                self.actor.eval()
                action = self.actor(torch.from_numpy(observation).type(torch.float32).view(1, -1)).data.max(1)[0].numpy()

                self.actor.train()

                if total_step <=self.training_start:
                    action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(self.max_action * action)
                episode_reward += reward

                self.buffer.add(observation, action, self.reward_scale * reward, next_observation, done)
                observation = next_observation

            print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                     episode_reward))

            if total_step >= 5 * self.batch_size:
                for i in range(local_step):
                    s, a, r, ns, d = self.buffer.sample()
                    # s, a, r, ns, d = self.buffer.ERE_sample(i, update_len)
                    self.train(s, a, r, ns, d)


if __name__ == '__main__':

    env = gym.make("Pendulum-v0")#around 5000 steps
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedTriplePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps.
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("SAC training of", env.unwrapped.spec.id)

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    sac = SAC(state_dim, action_dim, max_action, min_action, True, True)
    sac.run()