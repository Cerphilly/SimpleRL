import numpy as np
import tensorflow as tf
import os

from Common.Utils import random_crop, center_crop_images
from Common import Data_Augmentation as rad

class Buffer:
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
        else:
            return element

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

    def export(self, log = False):
        states = np.array(self.s)
        actions = np.array(self.a)
        rewards = np.array(self.r)
        states_next = np.array(self.ns)
        dones = np.array(self.d)

        if log == True:
            print("States: ", np.shape(states))
            print("Actions: ", np.shape(actions))
            print("Rewards: ", np.shape(rewards))
            print("Next States: ", np.shape(states_next))
            print("Dones: ", np.shape(dones))


        return states, actions, rewards, states_next, dones

    def load(self, buffer_dir, step):
        self.s = np.load(os.path.join(buffer_dir, "s_{}.npy".format(step)))
        self.a = np.load(os.path.join(buffer_dir, "a_{}.npy".format(step)))
        self.r = np.load(os.path.join(buffer_dir, "r_{}.npy".format(step)))
        self.ns = np.load(os.path.join(buffer_dir, "ns_{}.npy".format(step)))
        self.d = np.load(os.path.join(buffer_dir, "d_{}.npy".format(step)))

        print("Buffer of step {} loaded to the buffer".format(step))

    def sample(self, batch_size):
        #ids = np.random.randint(low=0, high=len(self.s), size=batch_size)
        ids = np.random.choice(len(self.s), batch_size, replace=False)

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
        #not very useful
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

    def cpc_sample(self, batch_size, image_size=84):
        #ImageRL/CURL
        ids = np.random.randint(low=0, high=len(self.s), size=batch_size)

        states = np.asarray([self.s[i] for i in ids])
        actions = np.asarray([self.a[i] for i in ids])  # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids])
        states_next = np.asarray([self.ns[i] for i in ids])
        dones = np.asarray([self.d[i] for i in ids])
        pos = states.copy()

        states = random_crop(states, image_size)
        states_next = random_crop(states_next, image_size)
        pos = random_crop(pos, image_size)

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)
        pos = tf.convert_to_tensor(pos, tf.float32)

        cpc_kwargs = dict(obs_anchor=states, obs_pos=pos, time_anchor=None, time_pos=None)

        return states, actions, rewards, states_next, dones, cpc_kwargs


    def rad_sample(self, batch_size, aug_funcs, pre_image_size=100):
        ids = np.random.choice(len(self.s), batch_size, replace=False)

        states = np.asarray([self.s[i] for i in ids]) # (batch_size, states_dim)
        actions = np.asarray([self.a[i] for i in ids]) # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids]) # (batch_Size, 1)
        states_next = np.asarray([self.ns[i] for i in ids]) #(batch_size, states_dim)
        dones = np.asarray([self.d[i] for i in ids]) #(batch_size, 1)


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

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        return states, actions, rewards, states_next, dones


    def dbc_sample(self, batch_size):#not used anymore
        ids = np.random.choice(len(self.s), batch_size, replace=False)

        states = np.asarray([self.s[i] for i in ids])
        actions = np.asarray([self.a[i] for i in ids])
        rewards = np.asarray([self.r[i] for i in ids])
        states_next = np.asarray([self.ns[i] for i in ids])
        dones = np.asarray([self.d[i] for i in ids])

        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        np.random.shuffle(ids)

        states2 = np.asarray([self.s[i] for i in ids])
        actions2 = np.asarray([self.a[i] for i in ids])
        rewards2 = np.asarray([self.r[i] for i in ids])
        states_next2 = np.asarray([self.ns[i] for i in ids])
        dones2 = np.asarray([self.d[i] for i in ids])

        states2 = tf.convert_to_tensor(states2, tf.float32)
        actions2 = tf.convert_to_tensor(actions2, tf.float32)
        rewards2 = tf.convert_to_tensor(rewards2, tf.float32)
        states_next2 = tf.convert_to_tensor(states_next2, tf.float32)
        dones2 = tf.convert_to_tensor(dones2, tf.float32)


        return (states, actions, rewards, states_next, dones), (states2, actions2, rewards2, states_next2, dones2)














