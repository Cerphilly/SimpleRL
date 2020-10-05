#Trust Region Policy Optimization, Schulman et al, 2015.
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/trpo.html

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network, V_network

class TRPO:
    def __init__(self, state_dim, action_dim, max_action = 1, min_action=1, discrete=True, actor=None, critic=None, training_step=1, gamma = 0.99,
                 lambda_gae = 0.95, learning_rate = 3e-4, batch_size=64, backtrack_iter=10, backtrack_coeff=0.6, delta=0.05, num_epoch=5):

        self.actor = actor
        self.critic = critic
        self.max_action = max_action
        self.min_action = min_action

        self.discrete = discrete

        self.buffer = Buffer()

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.batch_size = batch_size
        self.backtrack_iter = backtrack_iter
        self.backtrack_coeff = backtrack_coeff
        self.delta = delta
        self.num_epoch = num_epoch

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = training_step

        if self.actor == None:
            if self.discrete == True:
                self.actor = Policy_network(self.state_dim, self.action_dim)
            else:
                self.actor = Policy_network(self.state_dim, self.action_dim*2)

        if self.critic == None:
            self.critic = V_network(self.state_dim)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]
        else:
            output = self.actor(state)
            mean, log_std = self.max_action * (output[:, :self.action_dim]), output[:, self.action_dim:]
            std = tf.exp(log_std)

            eps = tf.random.normal(tf.shape(mean))
            action = (mean + std * eps)[0]
            action = tf.clip_by_value(action, self.min_action, self.max_action)

        return action

    def fisher_vector_product(self, states, p):
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                if self.discrete == True:
                    kl_divergence = tfp.distributions.kl_divergence(
                        tfp.distributions.Categorical(probs=self.actor(states, activation='softmax')),
                        tfp.distributions.Categorical(probs=self.actor(states, activation='softmax')))
                else:
                    policy = self.actor(states)
                    mean, log_std = self.max_action * policy[:, :self.action_dim], policy[:, self.action_dim:]
                    std = tf.exp(log_std)
                    dist = tfp.distributions.Normal(loc=mean, scale=std)
                    kl_divergence = tfp.distributions.kl_divergence(dist, dist)
                kl_divergence = tf.reduce_mean(kl_divergence)
            kl_grad = tape1.gradient(kl_divergence, self.actor.trainable_variables)
            flatten_kl_grad = tf.concat([tf.reshape(grad, [-1]) for grad in kl_grad], axis=0)
            kl_grad_p = tf.reduce_sum(flatten_kl_grad * p)

        kl_hessian_p = tape2.gradient(kl_grad_p, self.actor.trainable_variables)
        flatten_kl_hessian_p = tf.concat([tf.reshape(hessian, [-1]) for hessian in kl_hessian_p], axis=0)

        return flatten_kl_hessian_p + 0.1 * p

    def conjugate_gradient(self, states, b, nsteps, residual_tol=1e-10):
        x = np.zeros_like(b)
        r = b.copy()
        p = b.copy()
        rdotr = np.dot(r, r)

        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / (np.dot(p, _Avp) + 1e-8)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = np.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
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
        s, a, r, ns, d = self.buffer.all_sample()

        old_values = self.critic(s)

        returns = np.zeros_like(r.numpy())
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)

        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t])).numpy()
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - old_values[t]).numpy()
            running_advantage = (
                        running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t])).numpy()

            returns[t] = running_return
            previous_value = old_values[t]
            advantages[t] = running_advantage

        if self.discrete == True:
            old_policy = self.actor(s, activation = 'softmax')
            old_a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
            old_log_policy = tf.reduce_sum(tf.math.log(old_policy) * tf.stop_gradient(old_a_one_hot), axis=1, keepdims=True)

        else:
            old_policy = self.actor(s)
            old_mean, old_log_std = self.max_action * (old_policy[:, :self.action_dim]), old_policy[:, self.action_dim:]
            old_std = tf.exp(old_log_std)
            old_dist = tfp.distributions.Normal(loc=old_mean, scale=old_std)
            old_log_policy = old_dist.log_prob(a)

        with tf.GradientTape() as tape:
            if self.discrete == True:
                policy = self.actor(s, activation='softmax')
                a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
                log_policy = tf.reduce_sum(tf.math.log(policy) * tf.stop_gradient(a_one_hot), axis=1, keepdims=True)

                surrogate = tf.reduce_mean(tf.exp(log_policy - tf.stop_gradient(old_log_policy)) * advantages)

            else:
                policy = self.actor(s)
                mean, log_std = self.max_action * policy[:, :self.action_dim], policy[:, self.action_dim:]
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(loc=mean, scale=std)
                log_policy = dist.log_prob(s)

                surrogate = tf.reduce_mean(tf.exp(log_policy - tf.stop_gradient(old_log_policy)) * advantages)
        policy_grad = tape.gradient(surrogate, self.actor.trainable_variables)
        flatten_policy_grad = tf.concat([tf.reshape(grad, [-1]) for grad in policy_grad], axis=0)

        step_dir = self.conjugate_gradient(s, flatten_policy_grad.numpy(), 10)


        flattened_actor = tf.concat([tf.reshape(variable, [-1]) for variable in self.actor.trainable_variables], axis=0)

        shs = 0.5 * tf.reduce_sum(step_dir * self.fisher_vector_product(s, step_dir), axis=0, keepdims=True) + 1e-8
        step_size = 1 / tf.sqrt(shs / self.delta)[0]
        full_step = step_size * step_dir

        if self.discrete == True:
            self.backup_actor = Policy_network(self.state_dim, self.action_dim)
        else:
            self.backup_actor = Policy_network(self.state_dim, self.action_dim * 2)

        self.update_model(self.backup_actor, flattened_actor)
        #copy_weight(self.actor, self.backup_actor)
        expected_improve = tf.reduce_sum(flatten_policy_grad * full_step, axis=0, keepdims=True)

        flag = False
        fraction = 1.0

        for i in range(self.backtrack_iter):
            new_flattened_actor = flattened_actor + fraction * full_step
            self.update_model(self.actor, new_flattened_actor)
            if self.discrete == True:
                new_policy = self.actor(s, activation='softmax')
                new_a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
                new_log_policy = tf.reduce_sum(tf.math.log(new_policy) * tf.stop_gradient(new_a_one_hot), axis=1, keepdims=True)
            else:
                new_policy = self.actor(s)
                new_mean, new_log_std = self.max_action * new_policy[:, :self.action_dim], new_policy[:, self.action_dim:]
                new_std = tf.exp(new_log_std)
                new_dist = tfp.distributions.Normal(loc=new_mean, scale=new_std)
                new_log_policy = new_dist.log_prob(s)

            new_surrogate = tf.reduce_mean(tf.exp(new_log_policy - old_log_policy) * advantages)

            loss_improve = new_surrogate - surrogate
            expected_improve *= fraction

            if self.discrete == True:
                new_kl_divergence = tfp.distributions.kl_divergence(tfp.distributions.Categorical(probs=self.actor(s, activation='softmax')),
                                                                    tfp.distributions.Categorical(probs=self.backup_actor(s, activation='softmax')))
            else:
                new_policy = self.actor(s)
                new_mean, new_log_std = self.max_action * new_policy[:, :self.action_dim], new_policy[:, self.action_dim:]
                new_std = tf.exp(new_log_std)
                new_dist = tfp.distributions.Normal(loc=new_mean, scale=new_std)

                backup_policy = self.backup_actor(s)
                backup_mean, backup_log_std = self.max_action * backup_policy[:,:self.action_dim], backup_policy[:,self.action_dim:]
                backup_std = tf.exp(backup_log_std)
                backup_dist = tfp.distributions.Normal(loc=backup_mean, scale=backup_std)
                new_kl_divergence = tfp.distributions.kl_divergence(new_dist, backup_dist)

            new_kl_divergence = tf.reduce_mean(new_kl_divergence)

            print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
                  'number of line search: {}'
                  .format(new_kl_divergence.numpy(), loss_improve, expected_improve[0], i))

            if new_kl_divergence.numpy() < self.delta and (loss_improve / expected_improve[0]) > self.backtrack_coeff:
                flag = True
                break

            fraction *= self.backtrack_coeff

        if not flag:
            self.update_model(self.actor, flattened_actor)
            #copy_weight(self.backup_actor, self.actor)
            print("Policy update failed")

        #critic_train

        n = len(s)
        arr = np.arange(n)
        # arr = np.random.choice(arr, self.batch_size, replace=False)

        for epoch in range(self.num_epoch):
            np.random.shuffle(arr)

            if n // self.batch_size > 0:
                batch_index = arr[:self.batch_size]
            else:
                batch_index = arr

            batch_s = s.numpy()[batch_index]
            batch_returns = returns[batch_index]

            with tf.GradientTape() as tape:
                critic_loss = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(batch_returns) - self.critic(batch_s)))

            critic_variables = self.critic.trainable_variables
            critic_gradients = tape.gradient(critic_loss, critic_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))


        if flag:
            self.buffer.delete()




