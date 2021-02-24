from Algorithms.TD3 import TD3
from Algorithms.SAC_v1 import SAC_v1
from Algorithms.SAC_v2 import SAC_v2

from Networks.D2RL_Networks import *

from Common.Utils import copy_weight


class D2RL_TD3(TD3):
    def __init__(self, state_dim, action_dim, hidden_dim=256, training_step=100, batch_size=128, buffer_size=1e6,
                 gamma=0.99, tau=0.005, actor_lr=0.001, critic_lr=0.001, policy_delay=2, actor_noise=0.1, target_noise=0.2, noise_clip=0.5, training_start=500):
        super(D2RL_TD3, self).__init__(state_dim, action_dim, hidden_dim, training_step, batch_size, buffer_size, gamma, tau, actor_lr, critic_lr, policy_delay, actor_noise, target_noise, noise_clip, training_start)

        self.actor = D2RL_Policy(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_actor = D2RL_Policy(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_critic1 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_critic2 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'D2RL_TD3'


class D2RL_SAC_v1(SAC_v1):
    def __init__(self, state_dim, action_dim, hidden_dim=256, training_step=1,
                 batch_size=128, buffer_size=1e6, tau=0.005, learning_rate=0.0003, gamma=0.99, alpha=0.2, reward_scale=1, training_start = 500):
        super(D2RL_SAC_v1, self).__init__(state_dim, action_dim, hidden_dim, training_step, batch_size, buffer_size, tau, learning_rate, gamma, alpha, reward_scale, training_start)

        self.actor = D2RL_Squashed_Gaussian(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.v_network = D2RL_V(self.state_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_v_network = D2RL_V(self.state_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))

        copy_weight(self.v_network, self.target_v_network)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'V_network': self.v_network, 'Target_V_network': self.target_v_network}
        self.name = 'D2RL_SAC_v1'

class D2RL_SAC_v2(SAC_v2):
    def __init__(self, state_dim, action_dim, hidden_dim=256, training_step=1, alpha=0.1, train_alpha=True,
                 batch_size=128, buffer_size=1e6, tau=0.005, learning_rate=0.0003, gamma=0.99, reward_scale=1, training_start = 500):
        super(D2RL_SAC_v2, self).__init__(state_dim, action_dim, hidden_dim, training_step, alpha, train_alpha, batch_size, buffer_size, tau, learning_rate, gamma, reward_scale, training_start)

        self.actor = D2RL_Squashed_Gaussian(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_critic1 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.target_critic2 = D2RL_Q(self.state_dim, self.action_dim, (hidden_dim, hidden_dim, hidden_dim, hidden_dim))

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'D2RL_SAC_v2'