#Proximal Policy Optimization Algorithm, Schulman et al, 2017
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/ppo.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Network.Basic_Networks import Policy_network, V_network
from Network.Gaussian_Actor import Gaussian_Actor
from Network.Encoder import PixelEncoder

class ImagePPO:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, obs_dim, action_dim, args):

        self.discrete = args.discrete
        self.buffer = Buffer(state_dim=obs_dim, action_dim=action_dim if args.discrete == False else 1, max_size=args.buffer_size, on_policy=True)

        self.ppo_mode = args.ppo_mode #mode: 'clip'
        assert self.ppo_mode is 'clip'

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae
        self.batch_size = args.batch_size
        self.clip = args.clip

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = args.training_step

        self.feature_dim = args.feature_dim

        if self.discrete:
            self.actor = Policy_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        else:
            self.actor = Gaussian_Actor(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.critic = V_network(state_dim=self.feature_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, data_format='channels_first', activation=args.activation)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'PPO'

    def get_action(self, observation):
        observation = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)

        if self.discrete:
            feature = self.encoder(observation)
            policy = self.actor(feature, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()
            log_prob = dist.log_prob(action).numpy()
            action = action[0]

        else:
            feature = self.encoder(observation)
            action, log_prob = self.actor(feature)
            action = action.numpy()[0]
            log_prob = log_prob.numpy()[0]

        return action, log_prob

    def eval_action(self, observation):
        observation = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)

        if self.discrete:
            feature = self.encoder(observation)
            policy = self.actor(feature, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()[0]

        else:
            feature = self.encoder(observation)
            action, _ = self.actor(feature, deterministic=True)
            action = action.numpy()[0]

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c_loss = 0

        s, a, r, ns, d, log_prob = self.buffer.all_sample()

        r = r.numpy()
        d = d.numpy()

        old_values = self.critic(self.encoder(s)).numpy()
        returns = np.zeros_like(r)
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)

        #GAE
        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t]))
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - old_values[t])
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t]))

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std())

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        n = len(s)
        arr = np.arange(n)
        training_num2 = max(int(n / self.batch_size), 1)#200/32 = 6
        for i in range(training_num):
            for epoch in range(training_num2):#0, 1, 2, 3, 4, 5
                if epoch < training_num2 - 1:#5
                    batch_index = arr[self.batch_size * epoch : self.batch_size * (epoch + 1)]

                else:
                    batch_index = arr[self.batch_size * epoch: ]

                batch_s = tf.gather(s, batch_index)
                batch_a = tf.gather(a, batch_index)
                batch_returns = tf.gather(returns, batch_index)
                batch_advantages = tf.gather(advantages, batch_index)
                batch_old_log_policy = tf.gather(log_prob, batch_index)

                with tf.GradientTape(persistent=True) as tape:

                    if self.discrete:
                        policy = self.actor(tf.stop_gradient(self.encoder(batch_s)), activation='softmax')
                        dist = tfp.distributions.Categorical(probs=policy)
                        log_policy = tf.reshape(dist.log_prob(tf.squeeze(batch_a)), (-1, 1))
                        ratio = tf.exp(log_policy - batch_old_log_policy)
                        surrogate = ratio * batch_advantages

                        if self.ppo_mode == 'clip':
                            clipped_surrogate = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                            actor_loss = tf.reduce_mean(-tf.minimum(surrogate, clipped_surrogate))

                        else:
                            raise NotImplementedError

                    else:
                        dist = self.actor.dist(self.encoder(batch_s))
                        log_policy = dist.log_prob(batch_a)

                        ratio = tf.exp(log_policy - batch_old_log_policy)
                        surrogate = ratio * batch_advantages
                        if self.ppo_mode == 'clip':
                            clipped_surrogate = tf.clip_by_value(ratio, 1-self.clip, 1+self.clip) * batch_advantages

                            actor_loss = - tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate))

                        else:
                            raise NotImplementedError

                    critic_loss = tf.reduce_mean(tf.square(batch_returns - self.critic(self.encoder(batch_s))))

                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_gradients = tape.gradient(critic_loss, self.encoder.trainable_variables + self.critic.trainable_variables)

                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.encoder.trainable_variables + self.critic.trainable_variables))

                del tape

                total_a_loss += actor_loss.numpy()
                total_c_loss += critic_loss.numpy()

        self.buffer.delete()

        return {'Loss': {'Actor': total_a_loss, 'Critic': total_c_loss},
                'Value': {'Entropy': tf.reduce_mean(dist.entropy()).numpy()}}


