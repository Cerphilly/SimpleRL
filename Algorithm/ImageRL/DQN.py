#Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.
#For OpenAI Gym Atari

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, find_channel
from Network.Basic_Networks import Policy_network
from Network.Encoder import PixelEncoder


class ImageDQN:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(state_dim=obs_dim, action_dim=1, max_size=args.buffer_size)
        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.feature_dim = args.feature_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.copy_iter = args.copy_iter

        self.network = Policy_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_network = Policy_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, data_format=find_channel(args.domain_type), activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, data_format=find_channel(args.domain_type), activation=args.activation)

        copy_weight(self.network, self.target_network)
        copy_weight(self.encoder, self.target_encoder)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}
        self.encoder.summary()
        self.name = 'ImageDQN'

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
            feature = self.encoder(obs)
            q_value = self.network(feature, activation='linear').numpy()
            best_action = np.argmax(q_value, axis=1)[0]
            return best_action

    def eval_action(self, obs):
        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        q_value = self.network(feature, activation='linear').numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        return best_action


    def train(self, training_num):
        total_loss = 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_q = tf.reduce_max(self.target_network(self.target_encoder(ns), activation='linear'), axis=1, keepdims=True)
            target_value = r + self.gamma * (1-d) * target_q
            target_value = tf.stop_gradient(target_value)

            with tf.GradientTape() as tape:
                selected_values = tf.reduce_sum(self.network(self.encoder(s), activation='linear')*tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
                loss = tf.reduce_mean(tf.square(target_value - selected_values))

            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.network.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.network.trainable_variables))

            if self.current_step % self.copy_iter == 0:
                copy_weight(self.network, self.target_network)
                copy_weight(self.encoder, self.target_encoder)

            total_loss += loss.numpy()

            del tape

        return [['Loss/Loss', total_loss]]