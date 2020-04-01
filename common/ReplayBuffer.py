import numpy as np

class Buffer:
    def __init__(self, batch_size = 100, max_size=1e6):
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
        del self.s, self.a, self.r, self.ns, self.d

        self.s = []
        self.a = []
        self.r = []
        self.ns = []
        self.d = []

        print("Buffer deleted")

    def all_sample(self):
        states = np.asarray(self.s)
        actions = np.asarray(self.a)
        rewards = np.asarray(self.r)
        states_next = np.asarray(self.ns)
        dones = np.asarray(self.d)

        return states, actions, rewards, states_next, dones


    def sample(self):
        ids = np.random.randint(low=0, high=len(self.s), size=self.batch_size)
        #ids = np.random.choice(len(self.s), self.batch_size, replace=False)

        states = np.asarray([self.s[i] for i in ids]).astype('float32')  # (batch_size, states_dim)
        actions = np.asarray([self.a[i] for i in ids]).astype('float32') # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids]).astype('float32') # (batch_Size, 1)
        states_next = np.asarray([self.ns[i] for i in ids]).astype('float32') #(batch_size, states_dim)
        dones = np.asarray([self.d[i] for i in ids]).astype('float32') #(batch_size, 1)

        return states, actions, rewards, states_next, dones

    def ERE_sample(self, i, update_len, cmin = 100, eta = 0.996):
        N = len(self.s)
        cmin = cmin*self.batch_size
        ck = max(N*eta**(i*1000/update_len), cmin)
        ck = max(len(self.s) - int(ck), 0)

        ids = np.random.randint(low=ck, high=len(self.s), size=self.batch_size)
        states = np.asarray([self.s[i] for i in ids]).astype('float32')  # (batch_size, states_dim)
        actions = np.asarray([self.a[i] for i in ids]).astype('float32')  # (batch_size, 1)
        rewards = np.asarray([self.r[i] for i in ids]).astype('float32')  # (batch_Size, 1)
        states_next = np.asarray([self.ns[i] for i in ids]).astype('float32')  # (batch_size, states_dim)
        dones = np.asarray([self.d[i] for i in ids]).astype('float32')  # (batch_size, 1)

        return states, actions, rewards, states_next, dones


