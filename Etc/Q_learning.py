import numpy as np
import gym


class Q_learning:
    def __init__(self, env):
        self.env = env

        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.alpha = 0.1
        self.gamma = 0.8
        self.eps = 0.1
        self.max_episode = 1000

        self.q = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s,:])

        return action

    def run(self):
        success = 0

        for episode in range(1000):
            observation = env.reset()
            done = False
            episode_reward = 0
            local_step = 0

            while not done:

                action = self.action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                if reward == 0:
                    reward = -0.001
                if done and next_observation != 15:
                    reward = -1

                self.q[observation, action] = self.q[observation, action] + self.alpha*(reward + self.gamma*np.max(self.q[next_observation,:]) - self.q[observation, action])
                observation = next_observation
                episode_reward += reward
                local_step += 1



            print("Episode: {}, Step: {}, Episode_reward: {}".format(episode, local_step, episode_reward))
            if episode_reward >= 0:
                success += 1
        print(self.q)
        print("Success: ", success)

    def eval(self):
        for i in range(1):
            observation = self.env.reset()
            done = False
            local_step = 0

            while not done:
                local_step += 1

                action = np.argmax(self.q[observation,:])
                next_observation, reward, done, _ = self.env.step(action)

                self.env.render()
                observation = next_observation


if __name__ == '__main__':
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    env = gym.make("FrozenLake-v1", is_slippery=False)
    # env = gym.make("FrozenLake8x8-v0", is_slippery=False)

    q_learning = Q_learning(env)
    q_learning.run()
    q_learning.eval()