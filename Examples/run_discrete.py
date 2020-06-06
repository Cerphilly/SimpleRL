
import tensorflow as tf
import gym

from Trainer.Gym_trainer import Online_Gym_trainer, Offline_Gym_trainer
from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.REINFORCE import REINFORCE
from Algorithms.VPG import VPG


def main(cpu_only = False, force_gpu = False):
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    #env = gym.make("Acrobot-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    dqn = DQN(state_dim, action_dim)
    ddqn = DDQN(state_dim, action_dim)

    reinforce = REINFORCE(state_dim, action_dim, mode='discrete')
    vpg = VPG(state_dim, action_dim, mode='discrete')

    #trainer = Online_Gym_trainer(env=env, algorithm=dqn, render=True)

    trainer = Offline_Gym_trainer(env=env, algorithm=vpg, render=True, save=False, load=False, log=False, save_period=100)
    trainer.run()



if __name__ == '__main__':
    main(cpu_only=False)