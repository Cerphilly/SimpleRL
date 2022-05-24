#Soft Actor-Critic Algorithm and Applications, Haarnoja et al, 2018

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update, remove_argument, modify_default, modify_choices
from Network.Basic_Network import Q_network
from Network.Gaussian_Policy import Gaussian_Policy


class SAC_v2:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim, action_dim, args.buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.critic_update = args.critic_update

        self.log_alpha = tf.Variable(np.log(args.alpha), dtype=tf.float32, trainable=args.train_alpha)
        self.target_entropy = -action_dim
        self.alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr)
        self.train_alpha = args.train_alpha

        self.actor = Gaussian_Policy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units, log_std_min=args.log_std_min, log_std_max=args.log_std_max, squash=True,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.critic1 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_critic1 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.critic2 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_critic2 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'SAC_v2'

    @staticmethod
    def get_config(parser):
        parser.add_argument('--log_std_min', default=-10, type=int, help='For gaussian actor')
        parser.add_argument('--log_std_max', default=2, type=int, help='For gaussian actor')
        parser.add_argument('--tau', default=0.005, type=float, help='Network soft update rate')
        parser.add_argument('--alpha', default=0.2, type=float)
        parser.add_argument('--train-alpha', default=True, type=bool)
        parser.add_argument('--alpha-lr', default=0.0005, type=float)

        parser.add_argument('--critic-update', default=1, type=int)

        remove_argument(parser, ['learning_rate', 'v_lr'])
        modify_choices(parser, 'train_mode', ['offline', 'online'])

        return parser

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        action, _ = self.actor(state)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        action, _ = self.actor(state, deterministic=True)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0

        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            ns_action, ns_logpi = self.actor(ns)

            target_min_aq = tf.minimum(self.target_critic1(ns, ns_action), self.target_critic2(ns, ns_action))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha.numpy() * ns_logpi))

            with tf.GradientTape(persistent=True) as tape1:

                critic1_loss = 0.5 * tf.reduce_mean(tf.square(self.critic1(s, a) - target_q))
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(self.critic2(s, a) - target_q))

            critic1_gradients = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
            critic2_gradients = tape1.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

            del tape1

            with tf.GradientTape() as tape2:
                s_action, s_logpi = self.actor(s)

                min_aq_rep = tf.minimum(self.critic1(s, s_action), self.critic2(s, s_action))

                actor_loss = 0.5 * tf.reduce_mean(self.alpha.numpy() * s_logpi - min_aq_rep)

            actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            del tape2

            if self.train_alpha == True:
                with tf.GradientTape() as tape3:
                    _, s_logpi = self.actor(s)
                    alpha_loss = -(tf.exp(self.log_alpha) * (tf.stop_gradient(s_logpi + self.target_entropy)))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)#from softlearning package

                alpha_grad = tape3.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))

                del tape3

            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)

            total_a_loss += actor_loss.numpy()
            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()
            if self.train_alpha == True:
                total_alpha_loss += alpha_loss.numpy()

        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss], ['Loss/alpha', total_alpha_loss], ['Alpha', tf.exp(self.log_alpha).numpy()]]



