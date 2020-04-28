import numpy as np
from gym import utils
from Tensorflow1.environment.gym_env import mujoco_env


class InvertedDoublePendulumSwingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_swing.xml', 2)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= 1.0)
        pos = (1+np.exp(-(ob[0] ** 2) * np.log(10) / 4))/2
        act = (4+np.maximum(1-(action**2),0.0))/5
        angle1 = (1-ob[3])/2
        angle2 = (1-ob[4])/2
        angle = (angle1+angle2)/2
        vel1 = (1+np.exp(-(ob[6]**2)*np.log(10)/25))/2
        vel2 = (1+np.exp(-(ob[7]**2)*np.log(10)/25))/2
        vel = np.minimum(vel1,vel2)
        r = angle*pos*act*vel
        done = not notdone
        return ob, r[0], done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            [np.sin(self.sim.data.qpos[1])],  # link angles
            [np.sin(self.sim.data.qpos[1]+self.sim.data.qpos[2])],
            [np.cos(self.sim.data.qpos[1])],
            [np.cos(self.sim.data.qpos[1]+self.sim.data.qpos[2])],
            self.sim.data.qvel
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005
