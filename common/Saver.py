import numpy as np
from copy import deepcopy

class Saver:
    def __init__(self, networks, networks_names, buffer, path):
        self.networks = networks #networks to save
        self.networks_names = networks_names #['network', 'target_network'], same order as networks list
        self.buffer = buffer #buffer to save
        self.path = path#path to save networks and buffer

    def buffer_save(self):
        np.savez_compressed('{}/buffer'.format(self.path), s=self.buffer.s, a=self.buffer.a, r=self.buffer.r, d=self.buffer.d, ns=self.buffer.ns)

    def network_save(self, network, name):
        network.save_weights('{}/{}'.format(self.path, name), overwrite=True, save_format='tf')

    def save(self):
        #self.buffer_save()
        for network, name in zip(self.networks, self.networks_names):
            self.network_save(network, name)

    def buffer_load(self):
        file_path = '{}/buffer'.format(self.path)
        loaded = np.load('{}.npz'.format(file_path))
        self.buffer.s = deepcopy(loaded['s'].tolist())
        self.buffer.a = deepcopy(loaded['a'].tolist())
        self.buffer.r = deepcopy(loaded['r'].tolist())
        self.buffer.ns = deepcopy(loaded['ns'].tolist())
        self.buffer.d = deepcopy(loaded['d'].tolist())

        print("buffer loaded")

    def network_load(self, network, name):
        network.load_weights('{}/{}'.format(self.path, name))
        print("{} loaded".format(name))

    def load(self):
        #self.buffer_load()
        for network, name in zip(self.networks, self.networks_names):
            self.network_load(network, name)










