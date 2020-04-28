import numpy as np
import torch

class Buffer:
    def __init__(self, batch_size = 100, max_size=1e6):#1000000: save the last 1000 episode at most
        self.max_size = max_size
        self.batch_size = batch_size
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

        #print(np.shape(self.s), np.shape(self.a), np.shape(self.r), np.shape(self.ns), np.shape(self.d))

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

        #print("Buffer deleted")

    def all_sample(self):
        states = torch.from_numpy(np.array(self.s)).type(torch.float32)
        actions = torch.from_numpy(np.array(self.a)).type(torch.float32)
        rewards = torch.from_numpy(np.array(self.r)).type(torch.float32)
        states_next = torch.from_numpy(np.array(self.ns)).type(torch.float32)
        dones = torch.from_numpy(np.array(self.d)).type(torch.long)

        return states, actions, rewards, states_next, dones


    def sample(self):
        ids = np.random.randint(low=0, high=len(self.s), size=self.batch_size)

        states = torch.from_numpy(np.asarray([self.s[i] for i in ids])).type(torch.float32)  # (batch_size, states_dim)
        actions = torch.from_numpy(np.asarray([self.a[i] for i in ids])).type(torch.float32) # (batch_size, 1)
        rewards = torch.from_numpy(np.asarray([self.r[i] for i in ids])).type(torch.float32) # (batch_Size, 1)
        states_next = torch.from_numpy(np.asarray([self.ns[i] for i in ids])).type(torch.float32) #(batch_size, states_dim)
        dones = torch.from_numpy(np.asarray([self.d[i] for i in ids])).type(torch.long) #(batch_size, 1)

        return states, actions, rewards, states_next, dones

    def ERE_sample(self, i, update_len, cmin = 100, eta = 0.996):
        N = len(self.s)
        cmin = cmin*self.batch_size
        ck = max(N*eta**(i*1000/update_len), cmin)
        ck = max(len(self.s) - int(ck), 0)

        ids = np.random.randint(low=ck, high=len(self.s), size=self.batch_size)
        states = torch.from_numpy(np.asarray([self.s[i] for i in ids])).type(torch.float32)  # (batch_size, states_dim)
        actions = torch.from_numpy(np.asarray([self.a[i] for i in ids])).type(torch.float32)  # (batch_size, 1)
        rewards = torch.from_numpy(np.asarray([self.r[i] for i in ids])).type(torch.float32)  # (batch_Size, 1)
        states_next = torch.from_numpy(np.asarray([self.ns[i] for i in ids])).type(torch.float32)  # (batch_size, states_dim)
        dones = torch.from_numpy(np.asarray([self.d[i] for i in ids])).type(torch.long)  # (batch_size, 1)

        return states, actions, rewards, states_next, dones


