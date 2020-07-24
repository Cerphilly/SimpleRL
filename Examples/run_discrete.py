import tensorflow as tf
import gym

from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.REINFORCE import REINFORCE
from Algorithms.VPG import VPG

class Online_Gym_trainer:
    def __init__(self, env, algorithm, render=True, max_episode = 1e6):

        self.env = env
        self.algorithm = algorithm
        self.render = render
        self.max_episode = max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0


    def run(self):

        while True:
            if self.episode > self.max_episode:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0
            self.losses = 0

            observation = self.env.reset()
            done = False

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    self.env.render()

                action = self.algorithm.get_action(observation)

                if self.local_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()


                next_observation, reward, done, _ = self.env.step(action)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if self.total_step >= self.algorithm.training_start:
                    self.algorithm.train(training_num=1)


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))


class Offline_Gym_trainer:
    def __init__(self, env, algorithm, render, save, load, log, save_period, max_episode=1e6):

        self.env = env
        self.algorithm = algorithm
        #self.saver = Saver('SAC_v1', 'low_damping', log)

        self.render = render
        self.save = save
        self.load = load
        self.log = log

        self.save_period = save_period
        self.max_episode = max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.random = False



    def run(self):

        #if self.load == True:
            #self.saver.load_weights(**self.algorithm.network_list)

        while True:
            if self.episode > self.max_episode:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0
            self.losses = None

            observation = self.env.reset()
            done = False
            self.random = False

            while not done:
                self.local_step += 1
                self.total_step += 1


                if self.render == True:
                    self.env.render()

                action = self.algorithm.get_action(observation)

                if self.load == False and self.total_step <= self.algorithm.training_start:
                    action = self.env.action_space.sample()
                    self.random = True

                next_observation, reward, done, _ = self.env.step(action)
                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward + 10, next_observation, done)
                observation = next_observation

            if self.total_step >= self.algorithm.training_start:
                self.algorithm.train(training_num=self.local_step)


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward,
                                                                                     self.local_step, self.total_step))

            if self.log == True:
                self.saver.log(self.episode, **{"reward": self.episode_reward, "local_step": self.local_step})

            if self.save == True and self.episode % self.save_period == 0:
                self.saver.save_weights(**self.algorithm.network_list)

def main(cpu_only = False, force_gpu = False):
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    #env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    env = gym.make("Acrobot-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    dqn = DQN(state_dim, action_dim)
    ddqn = DDQN(state_dim, action_dim)

    reinforce = REINFORCE(state_dim, action_dim, discrete=True)
    vpg = VPG(state_dim, action_dim, discrete=True)

    #trainer = Online_Gym_trainer(env=env, algorithm=dqn, render=True)

    trainer = Offline_Gym_trainer(env=env, algorithm=vpg, render=True, save=False, load=False, log=False, save_period=100)
    trainer.run()



if __name__ == '__main__':
    main(cpu_only=False)