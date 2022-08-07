#Trust Region Policy Optimization, Schulman et al, 2015.
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/trpo.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy

from Common.Buffer import Buffer
from Network.Basic_Networks import Policy_network, V_network
from Network.Gaussian_Actor import Gaussian_Actor

class TRPO:
    def __init__(self, state_dim, action_dim, args):

        self.discrete = args.discrete

        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim if args.discrete == False else 1, max_size=args.buffer_size, on_policy=True)

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae
        self.batch_size = args.batch_size
        self.backtrack_iter = args.backtrack_iter
        self.backtrack_coeff = args.backtrack_coeff
        self.delta = args.delta

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = args.training_step

        if self.discrete:
            self.actor = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
            self.backup_actor = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        else:
            self.actor = Gaussian_Actor(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
            self.backup_actor = Gaussian_Actor(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.critic = V_network(state_dim=self.state_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'TRPO'

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

        if self.discrete:
            policy = self.actor(state, activation='softmax')
            dist = tfp.distributions.Categorical(probs=policy)
            action = dist.sample().numpy()[0]

        else:
            action, _ = self.actor(state, deterministic=True)
            action = action.numpy()[0]

        return action

    def fisher_vector_product(self, states, p):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                if self.discrete:
                    kl_divergence = tfp.distributions.kl_divergence(
                        tfp.distributions.Categorical(probs=self.actor(states, activation='softmax')),
                        tfp.distributions.Categorical(probs=self.backup_actor(states, activation='softmax')))
                else:
                    dist = self.actor.dist(states)
                    backup_dist = self.backup_actor.dist(states)

                    kl_divergence = tfp.distributions.kl_divergence(dist, backup_dist)
                kl_divergence = tf.reduce_mean(kl_divergence)
            kl_grad = tape1.gradient(kl_divergence, self.actor.trainable_variables)

            flatten_kl_grad = tf.concat([tf.reshape(grad, [-1]) for grad in kl_grad], axis=0)
            kl_grad_p = tf.reduce_sum(flatten_kl_grad * p)

        kl_hessian_p = tape2.gradient(kl_grad_p, self.actor.trainable_variables)
        flatten_kl_hessian_p = tf.concat([tf.reshape(hessian, [-1]) for hessian in kl_hessian_p], axis=0).numpy()

        return flatten_kl_hessian_p + 0.1 * p


    def conjugate_gradient(self, states, b, nsteps):
        x = np.zeros_like(b)
        r = copy.deepcopy(b)
        p = copy.deepcopy(r)
        rdotr = np.dot(r, r)

        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / (np.dot(p, _Avp) + 1e-8)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = np.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr

        return x


    def update_model(self, model, new_variables):
        index = 0
        for variable in model.trainable_variables:
            variable_length = len(tf.reshape(variable, [-1]))
            new_variable = new_variables[index: index + variable_length]
            new_variable = tf.reshape(new_variable, tf.shape(variable))
            variable.assign(new_variable)

            index += variable_length

    def train(self, training_num):
        total_c_loss = 0
        s, a, r, ns, d, old_log_policy = self.buffer.all_sample()

        old_values = self.critic(s).numpy()

        r = r.numpy()
        d = d.numpy()

        returns = np.zeros_like(r)
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)

        for t in reversed(range(len(r))): #General Advantage Estimation
            running_return = (r[t] + self.gamma * running_return * (1 - d[t]))
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - old_values[t])
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t]))

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std())

        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        flattened_actor = tf.concat([tf.reshape(variable, [-1]) for variable in self.actor.trainable_variables], axis=0)
        self.update_model(self.backup_actor, flattened_actor)

        with tf.GradientTape() as tape:
            if self.discrete:
                policy = self.actor(s, activation='softmax')
                dist = tfp.distributions.Categorical(probs=policy)
                log_policy = tf.reshape(dist.log_prob(tf.squeeze(a)), (-1, 1))

                surrogate = tf.reduce_mean(tf.exp(log_policy - tf.stop_gradient(old_log_policy)) * advantages)

            else:
                dist = self.actor.dist(s)
                log_policy = dist.log_prob(a)

                surrogate = tf.reduce_mean(tf.exp(log_policy - tf.stop_gradient(old_log_policy)) * advantages)

        policy_grad = tape.gradient(surrogate, self.actor.trainable_variables)
        flatten_policy_grad = tf.concat([tf.reshape(grad, [-1]) for grad in policy_grad], axis=0)

        step_dir = self.conjugate_gradient(s, flatten_policy_grad.numpy(), 10)

        shs = 0.5 * tf.reduce_sum(step_dir * self.fisher_vector_product(s, step_dir), axis=0)
        step_size = 1 / tf.sqrt(shs / self.delta)
        full_step = step_size * step_dir

        expected_improve = tf.reduce_sum(flatten_policy_grad * full_step, axis=0)

        flag = False
        fraction = 1.0

        for i in range(self.backtrack_iter):
            new_flattened_actor = flattened_actor + fraction * full_step
            self.update_model(self.actor, new_flattened_actor)

            if self.discrete:
                new_policy = self.actor(s, activation='softmax')
                new_a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
                new_log_policy = tf.reduce_sum(tf.math.log(new_policy) * tf.stop_gradient(new_a_one_hot), axis=1, keepdims=True)
            else:
                new_dist = self.actor.dist(s)
                new_log_policy = new_dist.log_prob(a)

            new_surrogate = tf.reduce_mean(tf.exp(new_log_policy - old_log_policy) * advantages)

            loss_improve = new_surrogate - surrogate
            expected_improve *= fraction

            if self.discrete:
                new_kl_divergence = tfp.distributions.kl_divergence(tfp.distributions.Categorical(probs=self.actor(s, activation='softmax')),
                                                                    tfp.distributions.Categorical(probs=self.backup_actor(s, activation='softmax')))
            else:
                new_dist = self.actor.dist(s)
                backup_dist = self.backup_actor.dist(s)

                new_kl_divergence = tfp.distributions.kl_divergence(new_dist, backup_dist)

            new_kl_divergence = tf.reduce_mean(new_kl_divergence)

            #print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  ' 'number of line search: {}'.format(new_kl_divergence.numpy(), loss_improve, expected_improve, i))

            if new_kl_divergence.numpy() <= self.delta and loss_improve >= expected_improve:
                flag = True
                break

            fraction *= self.backtrack_coeff

        if not flag:
            self.update_model(self.actor, flattened_actor)
            print("Policy update failed")

        #critic_train

        n = len(s)
        arr = np.arange(n)
        training_step2 = max(int(n / self.batch_size), 1)

        for epoch2 in range(training_step2):
            if epoch2 < training_step2 - 1:  # 5
                batch_index = arr[self.batch_size * epoch2: self.batch_size * (epoch2 + 1)]

            else:
                batch_index = arr[self.batch_size * epoch2:]

            batch_s = tf.gather(s, batch_index)
            batch_returns = tf.gather(returns, batch_index)

            with tf.GradientTape() as tape:
                critic_loss = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(batch_returns) - self.critic(batch_s)))

            critic_variables = self.critic.trainable_variables
            critic_gradients = tape.gradient(critic_loss, critic_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

            total_c_loss += critic_loss.numpy()

        self.buffer.delete()
        return {'Loss': {'Critic': total_c_loss}}




