
import tensorflow as tf
import gym

from Tensorflow2.Trainer.Gym_trainer import Online_Gym_trainer
from Tensorflow2.Algorithms.DQN import DQN
from Tensorflow2.Algorithms.DDQN import DDQN


def main(cpu_only = False, force_gpu = False):
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    #env = gym.make("CartPole-v0")
    env = gym.make("MountainCar-v0")
    # env = gym.make("Acrobot-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    dqn = DQN(state_dim, action_dim, False, False)
    ddqn = DDQN(state_dim, action_dim, False, False)

    trainer = Online_Gym_trainer(env=env, algorithm=ddqn, render=True)
    trainer.run()



if __name__ == '__main__':
    main(cpu_only=False)