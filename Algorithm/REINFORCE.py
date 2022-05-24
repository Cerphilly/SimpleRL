# Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import remove_argument
from Network.Basic_Network import Policy_network
from Network.Gaussian_Policy import Gaussian_Policy


class REINFORCE:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim if args.discrete == False else 1,
                             max_size=args.buffer_size, on_policy=True)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.discrete = args.discrete

        self.gamma = args.gamma
        self.training_start = 0
        self.training_step = 1
        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        if args.discrete == True:
            self.network = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        else:
            self.network = Gaussian_Policy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units, log_std_min=args.log_std_min, log_std_max=args.log_std_max, squash=False,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        self.network_list = {'Network': self.network}
        self.name = 'REINFORCE'

    @staticmethod
    def get_config(parser):
        parser.add_argument('--log_std_min', default=-20, type=int, help='For gaussian actor')
        parser.add_argument('--log_std_max', default=2, type=int, help='For gaussian actor')
        remove_argument(parser, ['actor_lr', 'critic_lr', 'v_lr', 'batch_size'])

        return parser


    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)

        if self.discrete == True:
            policy = self.network(state, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()
            log_prob = dist.log_prob(action).numpy()
            action = action[0]

        else:
            action, log_prob = self.network(state)
            action = action.numpy()[0]
            log_prob = log_prob.numpy()[0]

        return action, log_prob

    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)

        if self.discrete == True:
            policy = self.network(state, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()[0]

        else:
            action, _ = self.network(state, deterministic=True)
            action = action.numpy()[0]

        return action

    def train(self, training_num):
        total_loss = 0
        s, a, r, ns, d, _ = self.buffer.all_sample()

        r = r.numpy()
        d = d.numpy()

        returns = np.zeros_like(r)

        running_return = 0
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - d[t])
            returns[t] = running_return

        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            if self.discrete == True:
                policy = self.network(s, activation='softmax')
                dist = tfp.distributions.Categorical(probs=policy)
                log_policy = tf.reshape(dist.log_prob(tf.squeeze(a)), (-1, 1))

            else:
                dist = self.network.dist(s)
                log_policy = dist.log_prob(a)

            loss = tf.reduce_sum(-log_policy * returns)

        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        total_loss += loss.numpy()

        self.buffer.delete()

        return [['Loss/Loss', total_loss]]

