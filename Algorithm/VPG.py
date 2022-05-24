#Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/vpg.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import remove_argument

from Network.Basic_Network import Policy_network, V_network
from Network.Gaussian_Policy import Gaussian_Policy

class VPG:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim, action_dim if args.discrete == False else 1, args.buffer_size, on_policy=True)
        self.discrete = args.discrete

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = 1

        if self.discrete == True:
            self.actor = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        else:
            self.actor = Gaussian_Policy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units, log_std_min=args.log_std_min, log_std_max=args.log_std_max, squash=False,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)


        self.critic = V_network(state_dim=self.state_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'VPG'


    @staticmethod
    def get_config(parser):
        parser.add_argument('--log_std_min', default=-20, type=int, help='For gaussian actor')
        parser.add_argument('--log_std_max', default=2, type=int, help='For gaussian actor')
        parser.add_argument('--lambda-gae', default=0.96, type=float)
        remove_argument(parser, ['learning_rate', 'v_lr', 'batch_size'])

        return parser

    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()
            log_prob = dist.log_prob(action).numpy()
            action = action[0]

        else:
            action, log_prob = self.actor(state)
            action = action.numpy()[0]
            log_prob = log_prob.numpy()[0]

        return action, log_prob


    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()[0]

        else:
            action, _ = self.actor(state, deterministic=True)
            action = action.numpy()[0]

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c_loss = 0

        s, a, r, ns, d, _ = self.buffer.all_sample()
        values = self.critic(s).numpy()

        r = r.numpy()
        d = d.numpy()

        returns = np.zeros_like(r)
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)

        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t]))
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - values[t])
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t]))

            returns[t] = running_return
            previous_value = values[t]
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / advantages.std()

        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            if self.discrete == True:
                policy = self.actor(s, activation='softmax')
                dist = tfp.distributions.Categorical(probs=policy)
                log_policy = tf.reshape(dist.log_prob(tf.squeeze(a)), (-1, 1))

            else:
                dist = self.actor.dist(s)
                log_policy = dist.log_prob(a)


            actor_loss = -tf.reduce_mean(log_policy * tf.stop_gradient(advantages))
            critic_loss = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(returns) - self.critic(s)))

        actor_variables = self.actor.trainable_variables
        critic_variables = self.critic.trainable_variables

        actor_gradients = tape.gradient(actor_loss, actor_variables)
        critic_gradients = tape.gradient(critic_loss, critic_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

        total_a_loss += actor_loss.numpy()
        total_c_loss += critic_loss.numpy()

        self.buffer.delete()

        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss]]






