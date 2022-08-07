#D2RL: Deep Dense Architectures in Reinforcement Learning, Sinha et al, 2020

from Algorithm.TD3 import TD3
from Algorithm.SAC_v1 import SAC_v1
from Algorithm.SAC_v2 import SAC_v2

from Network.D2RL_Networks import *

from Common.Utils import copy_weight


class D2RL_TD3(TD3):
    def __init__(self, state_dim, action_dim, args):
        super(D2RL_TD3, self).__init__(state_dim, action_dim, args)

        self.actor = D2RL_Policy(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_actor = D2RL_Policy(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic1 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic2 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'D2RL_TD3'


class D2RL_SAC_v1(SAC_v1):
    def __init__(self, state_dim, action_dim, args):
        super(D2RL_SAC_v1, self).__init__(state_dim, action_dim, args)

        self.actor = D2RL_Squashed_Gaussian(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.v_network = D2RL_V(self.state_dim, args.hidden_dim)
        self.target_v_network = D2RL_V(self.state_dim, args.hidden_dim)

        copy_weight(self.v_network, self.target_v_network)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'V_network': self.v_network, 'Target_V_network': self.target_v_network}
        self.name = 'D2RL_SAC_v1'

class D2RL_SAC_v2(SAC_v2):
    def __init__(self, state_dim, action_dim, args):
        super(D2RL_SAC_v2, self).__init__(state_dim, action_dim, args)

        self.actor = D2RL_Squashed_Gaussian(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic1 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic1 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic2 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic2 = D2RL_Q(self.state_dim, self.action_dim, args.hidden_dim)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'D2RL_SAC_v2'