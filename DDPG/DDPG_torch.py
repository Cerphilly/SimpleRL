#Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import numpy as np
from common.ReplayBuffer import Buffer
from common.Saver import Saver


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.max_action*torch.tanh(self.fc3(x))

        return output


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
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



class DDPG:
    def __init__(self, state_dim, action_dim, max_action, min_action, save=False, load=False, batch_size=100, gamma=0.99, tau=0.995, learning_rate=0.001, training_start=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load

        self.gamma = gamma
        self.tau = tau
        self.lr = learning_rate
        self.training_start = training_start
        self.batch_size = batch_size

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.buffer = Buffer(self.batch_size)


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.copy_weight(self.actor, self.target_actor)
        self.copy_weight(self.critic, self.target_critic)

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

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        value_next = self.target_critic(torch.cat((ns, self.target_actor(ns).detach()), dim=1))
        target_value = (r + self.gamma*(1.0-d)*value_next)
        critic_loss = F.mse_loss(self.critic(torch.cat((s, a), dim=1)), target_value.detach())
        critic_loss.backward()
        self.critic_optimizer.step()



        #actor_loss = -(1/self.batch_size)*(torch.sum(self.critic(torch.cat((s, self.actor(s)), dim=1))))
        actor_loss = -self.critic(torch.cat((s, self.actor(s)), dim=1))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)



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

                noise = np.random.normal(loc=0, scale=0.1, size=action_dim)

                self.actor.eval()
                action = self.actor(torch.from_numpy(observation).type(torch.float32).view(1, -1)).data.max(1)[0].numpy() + noise
                action = np.clip(action, self.min_action, self.max_action)
                self.actor.train()

                if total_step < self.training_start:
                    action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if total_step >= self.training_start:
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

            print("episode: {}, total_step: {}, episode_reward: {}".format(episode, total_step, episode_reward))


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("DDPG training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    ddpg = DDPG(state_dim, action_dim, max_action, min_action)
    ddpg.run()