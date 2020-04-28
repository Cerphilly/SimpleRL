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
import tensorflow as tf
import tensorflow_probability as tfp

import gym
import dm_control2gym
from dm_control import suite

import numpy as np
import math

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstep, dmstate
from running_state import ZFilter

class Actor(tf.keras.Model):#Policy_network
    def __init__(self, state_dim, hidden_units, action_dim, max_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(self.action_dim, name="output")

    @tf.function
    def call(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        mu = tf.nn.tanh(self.output_layer(z))
        logstd = tf.zeros_like(mu)
        std = tf.exp(logstd)*0.3

        return mu, std, logstd


class Critic(tf.keras.Model):#V_network
    def __init__(self, state_dim, hidden_units):
        super(Critic, self).__init__()
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
    def __init__(self, state_dim, action_dim, max_action, min_action, batch_size = 20, actor_update=10, critic_update=10, clip=0.2, gamma=0.99, lamda=0.98, actor_lr=0.0001, critic_lr=0.0002):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.batch_size = batch_size
        self.actor_update = actor_update
        self.critic_update = critic_update
        self.clip = clip
        self.gamma = gamma
        self.lamda = lamda

        self.actor = Actor(self.state_dim, [100, 100], self.action_dim, self.max_action)
        self.critic = Critic(self.state_dim, [100, 100])
        self.buffer = Buffer(self.batch_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

    def log_density(self, x, mu, std, logstd):
        var = tf.square(std)
        log_density = -tf.square(x - mu) / (2 * var) \
                      - 0.5 * tf.math.log(2 * math.pi) - logstd
        return log_density

    def surrogate_loss(self, advants, states, old_policy, actions):
        mu, std, logstd = self.actor(states)
        new_policy = self.log_density(actions, mu, std, logstd)
        ratio = tf.exp(new_policy - old_policy)
        surrogate = ratio*advants

        return surrogate, ratio

    def train(self, s, a, discounted_r):

        advantage = (discounted_r - self.critic(s)).numpy()
        #advantage = (advantage - advantage.mean())/(advantage.std()+1e-6)

        mu, std, logstd = self.actor(s)
        oldpi_prob = self.log_density(a, mu, std, logstd)

        oldpi_prob = tf.stop_gradient(oldpi_prob)

        for _ in range(self.actor_update):
            with tf.GradientTape() as tape:
                mu, std, logstd = self.actor(s)
                pi_prob = self.log_density(a, mu, std, logstd)
                ratio = tf.exp(pi_prob - oldpi_prob)

                surr = ratio * advantage
                actor_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2)*advantage))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        for _ in range(self.critic_update):
            with tf.GradientTape() as tape:
                value = self.critic(s)
                critic_loss = tf.reduce_mean(tf.square(discounted_r - value))
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_weights)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    def train2(self, s, a, r, ns, d):
        values = self.critic(s).numpy()
        returns = np.zeros_like(r)
        advants = np.zeros_like(r)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for i in reversed(range(0, len(r))):
            running_returns = r[i] + self.gamma*running_returns*(1-d[i])
            running_tderror = r[i] + self.gamma*previous_value*(1-d[i]) - values[i]
            running_advants = running_tderror + self.gamma*self.lamda*running_advants*(1-d[i])

            returns[i] = running_returns
            previous_value = values[i]
            advants[i] = running_advants

        #advants = (advants - advants.mean())/advants.std()
        with tf.GradientTape(persistent=True) as tape:
            mu, std, logstd = self.actor(s)
            old_policy = self.log_density(a, mu, std, logstd)
            old_values = self.critic(s)

            surr, ratio = self.surrogate_loss(advants, s, old_policy.numpy(), a)


            values = self.critic(s)
            critic_loss = tf.reduce_mean(tf.square(values - returns))

            actor_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.clip, 1. + self.clip)*advants))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))












    def run(self):
        running_state = ZFilter((self.state_dim,), clip=5)
        episode = 0
        total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            local_step = 0

            done = False
            observation = env.reset()
            observation = running_state(observation)

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                mu, std, logstd = np.max(self.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

                distribution = tfp.distributions.Normal(loc=mu, scale=std)
                action = np.clip(distribution.sample().numpy(), -self.max_action, self.max_action)

                next_observation, reward, done, _ = env.step(action)
                next_observation = running_state(next_observation)

                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if local_step % self.batch_size == 0 or done == True:
                    s, a, r, ns, d = self.buffer.all_sample()
                    '''assert len(s) == self.batch_size
                    next_v = self.critic(np.expand_dims(next_observation, axis=0).astype('float32'))
                    discounted_r = []
                    for reward in r[::-1]:
                        next_v = reward + self.gamma * next_v
                        discounted_r.append(next_v.numpy()[0])
                    discounted_r.reverse()
                    r = np.array(discounted_r)
                    '''
                    self.train2(s, a, r, ns, d)
                    self.buffer.delete()

            print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                     episode_reward))



if __name__ == '__main__':
    env = gym.make("Pendulum-v0")


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("PPO training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    ppo = PPO(state_dim, action_dim, max_action, min_action)

    ppo.run()