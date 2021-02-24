import gym
import numpy as np


class MC:
    def __init__(self, env):
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

class SARSA:
    def __init__(self, env):
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.alpha = 0.1
        self.gamma = 0.9
        self.eps = 0.1

        self.q = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s,:])

        return action

    def run(self):
        success = 0

        for episode in range(10000):
            observation = env.reset()
            done = False
            episode_reward = 0
            local_step = 0

            while not done:
                action = self.action(observation)
                next_observation, reward, done, _ = env.step(action)

                if reward == 0:
                    reward = -0.001
                if done and next_observation != 15:
                    reward = -1
                episode_reward += reward

                next_action = self.action(next_observation)

                self.q[observation, action] = self.q[observation, action] + self.alpha * (reward + self.gamma * self.q[next_observation, next_action] - self.q[observation, action])
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
            observation = env.reset()
            done = False
            local_step = 0

            while not done:
                local_step += 1

                action = np.argmax(self.q[observation, :])
                next_observation, reward, done, _ = env.step(action)

                env.render()
                observation = next_observation



class Dyna_Q:
    def __init__(self, env):
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.alpha = 0.1
        self.gamma = 0.8
        self.eps = 0.1
        self.model_training = 10

        self.q = np.zeros([self.state_dim, self.action_dim])
        self.model_r = np.zeros([self.state_dim, self.action_dim])
        self.model_ns = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s,:])

        return action

    def run(self):
        states = []
        actions = []
        success = 0

        for episode in range(10000):
            observation = env.reset()
            done = False
            episode_reward = 0
            local_step = 0


            while not done:
                action = self.action(observation)
                next_observation, reward, done, _ = env.step(action)
                if reward == 0:
                    reward = -0.001
                if done and next_observation != 15:
                    reward = -1
                episode_reward += reward
                local_step += 1
                self.q[observation, action] = self.q[observation, action] + self.alpha*(reward + self.gamma*np.max(self.q[next_observation, :]) - self.q[observation, action])
                #self.q[observation, action] = self.q[observation, action] + self.alpha*(reward + self.gamma*np.max(self.q[next_observation,:]) - self.q[observation, action])

                self.model_r[observation, action] = reward
                self.model_ns[observation, action] = next_observation

                states.append(observation)
                actions.append(action)

                observation = next_observation


            if episode >= 100:
                for _ in range(self.model_training):
                    sample = np.random.randint(low=0, high=len(states) - 1)
                    s = states[sample]
                    a = actions[sample]

                    r= (self.model_r[s, a])
                    ns = int(self.model_ns[s, a])
                    self.q[s, a] = self.q[s,a] + self.alpha*(r + self.gamma*np.max(self.q[ns,:]) - self.q[s,a])

            print("Episode: {}, Step: {}, Episode_reward: {}".format(episode, local_step, episode_reward))
            if episode_reward >=0:
                success += 1
        print("Success: ", success)
        print(self.q)
        print(self.model_r)
        print(self.model_ns)


    def eval(self):
        for i in range(1):
            observation = env.reset()
            done = False
            local_step = 0

            while not done:
                local_step += 1

                action = np.argmax(self.q[observation,:])
                next_observation, reward, done, _ = env.step(action)

                env.render()
                observation = next_observation


class Model:
    def __init__(self, env):
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.alpha = 0.1
        self.gamma = 0.8
        self.eps = 0.1
        self.model_training = 10

        self.q = np.zeros([self.state_dim, self.action_dim])
        self.model_r = np.zeros([self.state_dim, self.action_dim])
        self.model_ns = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s,:])

        return action

    def run(self):
        states = []
        actions = []
        success = 0

        for episode in range(1000):
            observation = env.reset()
            done = False
            episode_reward = 0
            local_step = 0


            while not done:
                action = self.action(observation)
                next_observation, reward, done, _ = env.step(action)
                if reward == 0:
                    reward = -0.001
                if done and next_observation != 15:
                    reward = -1
                episode_reward += reward
                local_step += 1

                self.model_r[observation, action] = reward
                self.model_ns[observation, action] = next_observation

                states.append(observation)
                actions.append(action)

                observation = next_observation


            if episode >= 100:
                for _ in range(self.model_training):
                    sample = np.random.randint(low=0, high=len(states) - 1)
                    s = states[sample]
                    a = actions[sample]

                    r= (self.model_r[s, a])
                    ns = int(self.model_ns[s, a])
                    self.q[s, a] = self.q[s,a] + self.alpha*(r + self.gamma*np.max(self.q[ns,:]) - self.q[s,a])

            print("Episode: {}, Step: {}, Episode_reward: {}".format(episode, local_step, episode_reward))
            if episode_reward >=0:
                success += 1
        print("Success: ", success)
        print(self.q)
        print(self.model_r)
        print(self.model_ns)


    def eval(self):
        for i in range(1):
            observation = env.reset()
            done = False
            local_step = 0

            while not done:
                local_step += 1

                action = np.argmax(self.q[observation,:])
                next_observation, reward, done, _ = env.step(action)

                env.render()
                observation = next_observation

class Q_learing:
    def __init__(self, env):
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
                next_observation, reward, done, _ = env.step(action)
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
            observation = env.reset()
            done = False
            local_step = 0

            while not done:
                local_step += 1

                action = np.argmax(self.q[observation,:])
                next_observation, reward, done, _ = env.step(action)

                env.render()
                observation = next_observation


if __name__ == '__main__':
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    env = gym.make("FrozenLake-v0", is_slippery=False)
    #env = gym.make("FrozenLake8x8-v0", is_slippery=False)
    '''
    q_learning = Q_learing(env)
    q_learning.run()
    q_learning.eval()
    '''
    '''
    dyna_q = Dyna_Q(env)
    dyna_q.run()
    dyna_q.eval()
    '''
    #model = Model(env)
    #model.run()
    #model.eval()
    sarsa = SARSA(env)
    sarsa.run()
    sarsa.eval()




    #dyna_q = Dyna_Q(env=env)
