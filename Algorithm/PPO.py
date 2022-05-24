#Proximal Policy Optimization Algorithm, Schulman et al, 2017
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/ppo.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import remove_argument
from Network.Basic_Network import Policy_network, V_network
from Network.Gaussian_Policy import Gaussian_Policy
#todo: require overall remake

class PPO:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, args):

        self.discrete = args.discrete

        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim if args.discrete == False else 1, max_size=args.buffer_size, on_policy=True)

        self.ppo_mode = 'clip' #mode: 'clip'
        assert self.ppo_mode == 'clip'

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae
        self.batch_size = args.batch_size
        self.clip = args.clip

        self.beta = 1
        self.dtarg = 0.01

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = args.training_step

        if self.discrete == True:
            self.actor = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        else:
            self.actor = Gaussian_Policy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units, log_std_min=args.log_std_min, log_std_max=args.log_std_max, squash=False,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        self.critic = V_network(state_dim=self.state_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'PPO'

    @staticmethod
    def get_config(parser):
        parser.add_argument('--log_std_min', default=-20, type=int, help='For gaussian actor')
        parser.add_argument('--log_std_max', default=2, type=int, help='For gaussian actor')
        parser.add_argument('--lambda-gae', default=0.96, type=float)
        #parser.add_argument('--ppo-mode', default='clip', choices=['clip'])
        parser.add_argument('--clip', default=0.2, type=float)

        remove_argument(parser, ['learning_rate', 'v_lr'])

        return parser

    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax')
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

        s, a, r, ns, d, log_prob = self.buffer.all_sample()
        old_values = self.critic(s).numpy()

        r = r.numpy()
        d = d.numpy()

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

                    if self.discrete == True:
                        policy = self.actor(batch_s, activation='softmax')
                        dist = tfp.distributions.Categorical(probs=policy)
                        log_policy = tf.reshape(dist.log_prob(tf.squeeze(batch_a)), (len(batch_index), -1))
                        ratio = tf.exp(log_policy - batch_old_log_policy)
                        surrogate = ratio * batch_advantages

                        if self.ppo_mode == 'clip':
                            clipped_surrogate = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                            actor_loss = tf.reduce_mean(-tf.minimum(surrogate, clipped_surrogate))

                        else:
                            raise NotImplementedError

                    else:
                        dist = self.actor.dist(batch_s)
                        log_policy = dist.log_prob(batch_a)

                        ratio = tf.exp(log_policy - batch_old_log_policy)
                        surrogate = ratio * batch_advantages

                        if self.ppo_mode == 'clip':
                            clipped_surrogate = tf.clip_by_value(ratio, 1-self.clip, 1+self.clip) * batch_advantages

                            actor_loss = - tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate))

                        else:
                            raise NotImplementedError

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
        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss], ['Entropy/Actor', tf.reduce_mean(dist.entropy()).numpy()]]


