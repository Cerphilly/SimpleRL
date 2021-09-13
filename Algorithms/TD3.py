#Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Networks import Policy_network, Q_network
from Network.D2RL_Networks import D2RL_Policy, D2RL_Q

class TD3:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim, action_dim, args.buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.policy_delay = args.policy_delay
        self.actor_noise = args.actor_noise
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0

        self.actor = Policy_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_actor = Policy_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic1 = Q_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic1 = Q_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.critic2 = Q_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_critic2 = Q_network(self.state_dim, self.action_dim, args.hidden_dim)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'TD3'

    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)
        action = self.actor(state).numpy()[0] + noise
        action = np.clip(action, -1, 1)

        return action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        action = self.actor(state).numpy()[0]
        action = np.clip(action, -1, 1)

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_action = tf.clip_by_value(self.target_actor(ns) + tf.clip_by_value(tf.random.normal(shape=self.target_actor(ns).shape, mean=0, stddev=self.target_noise), -self.noise_clip, self.noise_clip), -1, 1)

            target_value = tf.stop_gradient(r + self.gamma * (1 - d) * tf.minimum(self.target_critic1(ns, target_action), self.target_critic2(ns, target_action)))

            with tf.GradientTape(persistent=True) as tape:
                critic1_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic1(s, a)))
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic2(s, a)))

            critic1_grad = tape.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))

            critic2_grad = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))

            if self.current_step % self.policy_delay == 0:
                with tf.GradientTape() as tape2:
                    actor_loss = -tf.reduce_mean(self.critic1(s, self.actor(s)))

                actor_grad = tape2.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                soft_update(self.actor, self.target_actor, self.tau)
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)

            del tape, tape2
            total_a_loss += actor_loss.numpy()
            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss]]



