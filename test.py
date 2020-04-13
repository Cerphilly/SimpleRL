import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("InvertedTriplePendulumSwing-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    while True:
        d = False
        s = env.reset()
        while not d:
            ns, r, d, _ = env.step(env.action_space.sample())
            print(ns, r, d)



