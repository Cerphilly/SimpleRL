#Generative Adversarial Imitation Learning, Ho and Ermon, 2016.
import tensorflow as tf
import numpy as np
import gym


import tensorflow_probability as tfp

import cv2

import dm_control2gym
from dm_control import suite

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstep, dmstate, dmextendstate, dmextendstep

from SAC import SAC_v1
from DDPG import DDPG


class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim):
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim + action_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = tf.nn.sigmoid(self.output_layer(z))

        return output

    def log_reward(self, inputs):
        return tf.math.log(self.call(inputs) + 1e-8)

class GAIL:
    def __init__(self, policy, state_dim, action_dim, max_action, min_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.discriminator = Discriminator(self.state_dim, [32, 32], self.action_dim)
        self.policy = policy#SAC, DDPG, TD3 etc
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999)
        self.expert_buffer = Buffer(self.policy.batch_size)

        self.saver = Saver([self.discriminator], ['discriminator'], self.expert_buffer, '/home/cocel/PycharmProjects/SimpleRL/GAIL/expert_Invertedpendulumswing-v2')

    def js_divergence(self, fake_logits, real_logits):
        m = (fake_logits + real_logits)/2
        js_divergence = tf.reduce_mean((fake_logits*tf.math.log(fake_logits/m + 1e-8) + real_logits*tf.math.log(real_logits/m + 1e-8))/2)

        return js_divergence

    def accuracy(self, fake_logits, real_logits):
        accuracy = \
            tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. + \
            tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.

        return accuracy

    def train(self, agent_states, agent_acts, expert_states, expert_acts):
        with tf.GradientTape() as tape:
            real_logits = self.discriminator(tf.concat([expert_states, expert_acts], axis=1))
            fake_logits = self.discriminator(tf.concat([agent_states, agent_acts], axis=1))

            loss = -(tf.reduce_mean(tf.math.log(real_logits + 1e-8)) + tf.reduce_mean(tf.math.log(1. - fake_logits + 1e-8)))
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        js_divergence = self.js_divergence(fake_logits, real_logits)
        accuracy = self.accuracy(fake_logits, real_logits)

        del tape

        return js_divergence, accuracy

    def inference(self, states, actions):
        return self.discriminator.log_reward(tf.concat([states, actions], axis=1))


    def run(self):
        self.saver.buffer_load()

        episode = 0
        total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            episode_js_divergence = 0
            episode_accuracy = 0
            local_step = 0

            done = False
            observation = env.reset()

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                action = np.max(self.policy.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

                if total_step <= 5 * self.policy.batch_size:
                    action = env.action_space.sample()


                next_observation, reward, done, _ = env.step(self.max_action * action)
                episode_reward += reward

                self.policy.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

            if total_step >= 5*self.policy.batch_size:
                for i in range(local_step):
                    s, a, _, ns, d = self.policy.buffer.sample()
                    r = self.inference(s, a)
                    expert_s, expert_a, expert_r, expert_ns, expert_d = self.expert_buffer.sample()
                    js_divergence, accuracy = self.train(s, a, expert_s, expert_a)
                    self.policy.train(s, a, r, ns, d)
                    episode_js_divergence += js_divergence
                    episode_accuracy += accuracy


            print("episode: {}, total_step: {}, step: {}, episode_reward: {}, episode_js_divergence: {}, episode_accuracy: {}".format(episode, total_step, local_step,
                                                                                     episode_reward, episode_js_divergence, episode_accuracy))

if __name__ == '__main__':

    #env = gym.make("Pendulum-v0")  # around 500 steps
    # env = gym.make("MountainCarContinuous-v0")

    # env = gym.make("InvertedDoublePendulumSwing-v2")
    # env = gym.make("InvertedDoublePendulum-v2")
    env = gym.make("InvertedPendulumSwing-v2")
    # env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("SAC training of", env.unwrapped.spec.id)

    '''
    env = suite.load(domain_name="cartpole", task_name="three_poles")#300만 스텝 학습: SAC_Test
    #env = suite.load(domain_name="cartpole", task_name="two_poles")
    #env = suite.load(domain_name="acrobot", task_name="swingup")

    #env = suite.load(domain_name="cartpole", task_name="swingup")
    state_spec = env.reset()
    action_spec = env.action_spec()
    state_dim = len(dmstate(state_spec))
    print(dmstate(state_spec))
    action_dim = action_spec.shape[0]  # 1
    max_action = action_spec.maximum[0]  # 1.0
    min_action = action_spec.minimum[0]
    '''

    parameters = {'tau': 0.995, "learning_rate": 0.0003, 'gamma': 0.99, 'alpha': 0.2, 'batch_size': 100,
                  'reward_scale': 1, 'save': False, 'load': False, 'expert_reward': -200}

    #
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)


    sac = SAC_v1.SAC(state_dim, action_dim, max_action, min_action, False, False)
    ddpg = DDPG.DDPG(state_dim, action_dim, max_action, min_action, False, False)

    gail = GAIL(sac, state_dim, action_dim, max_action, min_action)
    gail.run()