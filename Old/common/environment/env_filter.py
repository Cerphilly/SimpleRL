import numpy as np

class ENV(object):

    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = np.reshape(next_state,[1,self.state_dim])
        return next_state, reward, done, info

    def reset(self):
        init_obs = np.reshape(self.env.reset(), [1, self.state_dim])
        return init_obs

    def max_step(self):
        return self.env.spec.timestep_limit

    def render(self, close=False):
        if close == False:
            self.env.render()
        else: self.env.close()

    def close(self):
        self.env.close()