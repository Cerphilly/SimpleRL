import gym
from gym import spaces
from dm_control import suite
import numpy as np
from common.dm2gym import dmstate, dmstep, dmextendstate

'''
>>> from dm_control import suite
>>> env = suite.load("cartpole", "swingup")
>>> env.reset()
TimeStep(step_type=<StepType.FIRST: 0>, reward=None, discount=None, observation=OrderedDict([('position', array([-0.00767611, -0.99987463,  0.01583431])), ('velocity', array([-0.01245492,  0.00296658]))]))
>>> env = suite.load("cartpole", "two_poles")
>>> env.reset()
TimeStep(step_type=<StepType.FIRST: 0>, reward=None, discount=None, observation=OrderedDict([('position', array([-9.68353853e-04, -9.99991623e-01, -4.09323875e-03, -9.97486112e-01,
        7.08622302e-02])), ('velocity', array([ 0.00289493, -0.00048144,  0.00119111]))]))
>>> env = suite.load("cartpole", "three_poles")
>>> env.reset()
TimeStep(step_type=<StepType.FIRST: 0>, reward=None, discount=None, observation=OrderedDict([('position', array([ 0.01566801, -0.99999227, -0.00393219, -0.99901548,  0.04436286,
       -0.99834324,  0.05753937])), ('velocity', array([0.02063189, 0.0115125 , 0.01104127, 0.013586  ]))]))
>>> 
3/2
5/3
7/4
'''

if __name__ == '__main__':

    env = suite.load(domain_name='cartpole', task_name='swingup')
    observation = dmextendstate(env.reset())
    print(observation)
    x = env.step(0)

    ns, r, d = dmstep(x)
    print(ns, r, d)


