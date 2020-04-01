#Actor(state_dim, action_dim), Critic(state_dim) 하나 씩
#Actor critic을 한 모델로 바꿀 수 있다. 그러면 Loss도 바뀜
#lr: 3e-4?
#run: Actor에서 action 받아와서 env.step, buffer 저장(s, a, r, ns, d)
#주기적 training

'''
training:
s, a, r, ns, d batch sample(but from one episode)
step 1: get returns and GAEs and log probability of old policy
step 2: get value loss and actor loss and update actor & critic
'''
import torch
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import cv2

import gym
import dm_control2gym
from dm_control import suite

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstep, dmstate, dmextendstate, dmextendstep



class V_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units):
        super(V_network, self).__init__()
        self.state_dim = state_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)

        return output



class PPO:
    def __init__(self, state_dim, action_dim, max_action, min_action, gamma = 0.99):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.critic = V_network(self.state_dim, [100, 100])
        self.buffer = Buffer(100)

        self.gamma = gamma
        self.lamda = 0.98

    def get_gae(self, s, r, d):
        '''
        def get_gae(rewards, masks, values):
            rewards = torch.Tensor(rewards)
            masks = torch.Tensor(masks)
            returns = torch.zeros_like(rewards)
            advants = torch.zeros_like(rewards)

            running_returns = 0
            previous_value = 0
            running_advants = 0

            for t in reversed(range(0, len(rewards))):
                running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
                running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                            values.data[t]
                running_advants = running_tderror + hp.gamma * hp.lamda * \
                                  running_advants * masks[t]

                returns[t] = running_returns
                previous_value = values.data[t]
                advants[t] = running_advants

            advants = (advants - advants.mean()) / advants.std()
            return returns, advants
        '''
        returns = np.zeros_like(r)
        advantages = np.zeros_like(r)

        values = self.critic(s)
        running_r = 0
        previous_value = 0
        running_advantages = 0

        for i in reversed(range(0, len(r))):
            running_returns = r[i] + self.gamma*d[i]*running_r
            running_tderror = r[i] + self.gamma*previous_value*d[i] - values[i]
            running_advantages = running_tderror + self.gamma*d[i]*self.lamda*running_advantages

            returns[i] = running_returns
            previous_value = values[i]
            advantages[i] = running_advantages

        return returns, advantages

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

                action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(self.max_action * action)
                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

            print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                     episode_reward))
            #print(self.buffer.s, self.buffer.r, self.buffer.d)
            s, a, r, ns, d = self.buffer.all_sample()
            self.get_gae(s, r, d)




if __name__ == '__main__':


    env = gym.make("Pendulum-v0")#around 5000 steps
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps.
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("PPO training of", env.unwrapped.spec.id)
    '''
    env = suite.load(domain_name="cartpole", task_name="three_poles")  # 300만 스텝 학습: SAC_Test
    # env = suite.load(domain_name="cartpole", task_name="two_poles")
    # env = suite.load(domain_name="acrobot", task_name="swingup")

    # env = suite.load(domain_name="cartpole", task_name="swingup")
    state_spec = env.reset()
    action_spec = env.action_spec()
    state_dim = len(dmstate(state_spec))
    print(dmstate(state_spec))
    action_dim = action_spec.shape[0]  # 1
    max_action = action_spec.maximum[0]  # 1.0
    min_action = action_spec.minimum[0]

    parameters = {'tau': 0.995, "learning_rate": 0.0003, 'gamma': 0.99, 'alpha': 0.2, 'batch_size': 100,
                  'reward_scale': 1, 'save': True, 'load': True}

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    sac = SAC(state_dim, action_dim, max_action, min_action, True, False)
    sac.run_dm()
    '''

    ppo = PPO(state_dim, action_dim, max_action, min_action)
    ppo.run()