import numpy as np
import tensorflow as tf


class TFBuffer:
    def __init__(self, max_size=1e6):#1000000: save the last 1000 episode at most
        self.max_size = max_size
        self.s = []
        self.a = []
        self.r = []
        self.ns = []
        self.d = []

    def check(self, element):

        element = np.asarray(element)

        if element.ndim == 0:
            return np.expand_dims(element, axis=0)
        if element.ndim == 1:
            return element
        if element.ndim > 1:
            raise ValueError


    def add(self, s, a, r, ns, d):

        self.s.append(self.check(s))
        self.a.append(self.check(a))
        self.r.append(self.check(r))
        self.ns.append(self.check(ns))
        self.d.append(self.check(d))

        if len(self.s)>=self.max_size:
            del self.s[0]
            del self.a[0]
            del self.r[0]
            del self.ns[0]
            del self.d[0]

    def delete(self):

        self.s = []
        self.a = []
        self.r = []
        self.ns = []
        self.d = []


    def all_sample(self):
        states = np.array(self.s)
        actions = np.array(self.a)
        rewards = np.array(self.r)
        states_next = np.array(self.ns)
        dones = np.array(self.d)

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        return states, actions, rewards, states_next, dones


    def sample(self, batch_size):
        ids = np.random.randint(low=0, high=len(self.s), size=batch_size)
        #ids = np.random.choice(len(self.s), self.batch_size, replace=False)

        states = np.asarray([self.s[i] for i in ids]) # (batch_size, states_dim)
        actions = np.asarray([self.a[i] for i in ids]) # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids]) # (batch_Size, 1)
        states_next = np.asarray([self.ns[i] for i in ids]) #(batch_size, states_dim)
        dones = np.asarray([self.d[i] for i in ids]) #(batch_size, 1)

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)


        return states, actions, rewards, states_next, dones

    def ERE_sample(self, i, update_len, batch_size, cmin = 100, eta = 0.996):
        #Boosting Soft Actor-Critic: Emphasizing Recent Experience without Forgetting the Past, Che Wang, Keith Ross. arXiv:1906.04009T
        N = len(self.s)
        cmin = cmin*batch_size
        ck = max(N*eta**(i*1000/update_len), cmin)
        ck = max(len(self.s) - int(ck), 0)

        ids = np.random.randint(low=ck, high=len(self.s), size=batch_size)
        states = np.asarray([self.s[i] for i in ids])  # (batch_size, states_dim)
        actions = np.asarray([self.a[i] for i in ids])  # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids])  # (batch_Size, 1)
        states_next = np.asarray([self.ns[i] for i in ids])  # (batch_size, states_dim)
        dones = np.asarray([self.d[i] for i in ids])  # (batch_size, 1)

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        return states, actions, rewards, states_next, dones


