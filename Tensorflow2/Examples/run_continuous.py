import tensorflow as tf
import gym

import sys
sys.path.append('../..')

from Tensorflow2.Trainer.Gym_trainer import Offline_Gym_trainer
from Tensorflow2.Algorithms.DDPG import DDPG
from Tensorflow2.Algorithms.TD3 import TD3
from Tensorflow2.Algorithms.SAC_v1 import SAC_v1
from Tensorflow2.Algorithms.SAC_v2 import SAC_v2



def main(cpu_only = False, force_gpu = True):
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    #env = gym.make("Pendulum-v0")
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedTriplePendulumSwing-v2")
    #env = gym.make("InvertedTriplePendulum-v2")
    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps.
    #env = gym.make("InvertedPendulum-v2")


    env = gym.make("Ant-v2")
    #env = gym.make("HalfCheetah-v2")
    #env = gym.make("Hopper-v2")
    #env = gym.make("Humanoid-v2")
    #env = gym.make("HumanoidStandup-v2")
    #env = gym.make("Reacher-v2")
    #env = gym.make("Swimmer-v2")
    #env = gym.make("Walker2d-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("TD3 training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    ddpg = DDPG(state_dim, action_dim, max_action, min_action, False, False)
    td3 = TD3(state_dim, action_dim, max_action, min_action, False, False)
    sac_v1 = SAC_v1(state_dim, action_dim, max_action, min_action, False, False)
    sac_v2 = SAC_v2(state_dim, action_dim, max_action, min_action, False, False)

    trainer = Offline_Gym_trainer(env=env, algorithm=sac_v1, render=True)
    trainer.run()



if __name__ == '__main__':
    main()