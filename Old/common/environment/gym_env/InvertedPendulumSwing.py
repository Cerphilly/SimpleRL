import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumSwingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum_swing.xml', 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= 1.0)
        angle = (1-ob[2])/2
        pos = (1+np.exp(-(ob[0]**2)*np.log(10)/4))/2
        act = (4+np.maximum(1-(a**2),0.0))/5
        vel = (1+np.exp(-(ob[-1]**2)*np.log(10)/25))/2
        reward = angle * pos * act * vel
        done = not notdone
        return ob, reward[0], done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
        ob = np.array([ob[0], np.sin(ob[1]), np.cos(ob[1]), ob[2], ob[3]])
        return ob.ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
