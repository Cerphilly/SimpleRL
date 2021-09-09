import numpy as np
import tensorflow as tf
from Common.Utils import random_crop, center_crop_images

class Buffer:
    def __init__(self, state_dim, action_dim, max_size=1e6, on_policy=False):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.on_policy = on_policy

        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, s, a, r, ns, d, log_prob=None):
        np.copyto(self.s[self.idx], s)
        np.copyto(self.a[self.idx], a)
        np.copyto(self.r[self.idx], r)
        np.copyto(self.ns[self.idx], ns)
        np.copyto(self.d[self.idx], d)

        if self.on_policy == True:
            np.copyto(self.log_prob[self.idx], log_prob)

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    def delete(self):
        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def all_sample(self):
        ids = np.arange(self.max_size if self.full else self.idx)
        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def sample(self, batch_size):
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)
        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)


        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def cpc_sample(self, batch_size, image_size=84):
        # ImageRL/CURL
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        pos = states.copy()

        states = random_crop(states, image_size)
        states_next = random_crop(states_next, image_size)
        pos = random_crop(pos, image_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)


        cpc_kwargs = dict(obs_anchor=states, obs_pos=pos, time_anchor=None, time_pos=None)

        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

            return states, actions, rewards, states_next, dones, log_probs, cpc_kwargs

        return states, actions, rewards, states_next, dones, cpc_kwargs

    def rad_sample(self, batch_size, aug_funcs, pre_image_size=100):
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        for aug, func in aug_funcs.items():
            if 'crop' in aug or 'cutout' in aug:
                states = func(states)
                states_next = func(states_next)

            elif 'translate' in aug:
                states = center_crop_images(states, pre_image_size)
                states_next = center_crop_images(states_next, pre_image_size)

                states, random_idxs = func(states, return_random_idxs=True)
                states_next = func(states_next, **random_idxs)

        for aug, func in aug_funcs.items():
            if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                continue
            states = func(states)
            states_next = func(states_next)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones













