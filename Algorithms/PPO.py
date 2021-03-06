#Proximal Policy Optimization Algorithms, Schulman et al, 2017
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/ppo.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network, V_network
from Networks.Gaussian_Actor import Gaussian_Actor

class PPO:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, args):

        self.discrete = args.discrete

        self.buffer = Buffer(args.buffer_size)

        self.ppo_mode = args.ppo_mode #mode: 'clip', 'Adaptive KL', 'Fixed KL'
        assert self.ppo_mode is 'clip' or 'Adaptive KL' or 'Fixed KL'

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae
        self.batch_size = args.batch_size
        self.clip = args.clip
        self.beta = args.beta
        self.dtarg = args.dtarg

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = args.training_step

        if self.discrete == True:
            self.actor = Policy_network(self.state_dim, self.action_dim, args.hidden_dim)
        else:
            self.actor = Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim)

        self.critic = V_network(self.state_dim)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'PPO'

    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]

        else:
            action = self.actor(state).numpy()[0]

        return action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()[0]
            action = np.argmax(policy)

        else:
            action = self.actor(state, deterministic=True).numpy()[0]

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c_loss = 0

        s, a, r, ns, d = self.buffer.all_sample()
        old_values = self.critic(s)

        returns = np.zeros_like(r.numpy())
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)
        #GAE
        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t])).numpy()
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - old_values[t]).numpy()
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t])).numpy()

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        if self.discrete == True:
            old_policy = self.actor(s, activation = 'softmax')
            old_a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
            old_log_policy = tf.reduce_sum(tf.math.log(old_policy) * tf.stop_gradient(old_a_one_hot), axis=1, keepdims=True)

        else:
            old_mean, old_std = self.actor.mu_sigma(s)
            old_dist = self.actor.dist(s)
            old_log_policy = old_dist.log_prob(a)

        n = len(s)
        arr = np.arange(n)

        for epoch in range(training_num):

            if n // self.batch_size > 0:#for offline training
                np.random.shuffle(arr)

                batch_index = arr[:self.batch_size]
                batch_index.sort()

            else:
                batch_index = arr

            batch_s = s.numpy()[batch_index]
            batch_a = a.numpy()[batch_index]
            batch_returns = returns[batch_index]
            batch_advantages = advantages[batch_index]
            batch_old_log_policy = old_log_policy.numpy()[batch_index]


            with tf.GradientTape(persistent=True) as tape:
                if self.discrete == True:

                    batch_old_policy = old_policy.numpy()[batch_index]
                    policy = self.actor(batch_s, activation='softmax')
                    a_one_hot = tf.squeeze(tf.one_hot(tf.cast(batch_a, tf.int32), depth=self.action_dim), axis=1)
                    log_policy = tf.reduce_sum(tf.math.log(policy) * tf.stop_gradient(a_one_hot), axis=1, keepdims=True)

                    ratio = tf.exp(log_policy - batch_old_log_policy)
                    surrogate = ratio * batch_advantages

                    if self.ppo_mode == 'clip':
                        clipped_surrogate = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                        actor_loss = tf.reduce_mean(-tf.minimum(surrogate, clipped_surrogate))

                    else:
                        kl_divergence = tfp.distributions.kl_divergence(tfp.distributions.Categorical(probs=policy), tfp.distributions.Categorical(probs=batch_old_policy)).numpy()
                        kl_divergence = np.reshape(kl_divergence, [-1, 1])
                        actor_loss = -tf.reduce_mean(surrogate - self.beta*kl_divergence)

                        if self.ppo_mode == 'Adaptive KL':
                            d = np.mean(kl_divergence)
                            if d < self.dtarg/1.5:
                                self.beta = self.beta/2
                            elif d > self.dtarg*1.5:
                                self.beta = self.beta * 2

                else:
                    dist = self.actor.dist(batch_s)
                    log_policy = dist.log_prob(batch_a)

                    ratio = tf.exp(log_policy - batch_old_log_policy)
                    surrogate = ratio * batch_advantages

                    if self.ppo_mode == 'clip':
                        clipped_surrogate = tf.clip_by_value(ratio, 1-self.clip, 1+self.clip) * batch_advantages
                        actor_loss = tf.reduce_mean(-tf.minimum(surrogate, clipped_surrogate))

                    else:
                        batch_old_mean = old_mean.numpy()[batch_index]
                        batch_old_std = old_std.numpy()[batch_index]
                        batch_old_dist = tfp.distributions.Normal(loc=batch_old_mean, scale=batch_old_std)
                        kl_divergence = tfp.distributions.kl_divergence(dist, batch_old_dist).numpy()

                        actor_loss = -tf.reduce_mean(surrogate - self.beta*kl_divergence)

                        if self.ppo_mode == 'Adaptive KL':
                            d = np.mean(kl_divergence)
                            if d < self.dtarg / 1.5:
                                self.beta = self.beta / 2
                            elif d > self.dtarg * 1.5:
                                self.beta = self.beta * 2

                critic_loss = 0.5 * tf.reduce_mean(tf.square(batch_returns - self.critic(batch_s)))

            actor_variables = self.actor.trainable_variables
            critic_variables = self.critic.trainable_variables

            actor_gradients = tape.gradient(actor_loss, actor_variables)
            critic_gradients = tape.gradient(critic_loss, critic_variables)

            self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_variables))
            self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

            del tape

            total_a_loss += actor_loss.numpy()
            total_c_loss += critic_loss.numpy()

        self.buffer.delete()
        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss]]


