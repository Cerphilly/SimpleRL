import numpy as np
import gym

from SAC import SAC_v1
from SAC import SAC_v2

from common.ReplayBuffer import Buffer
from common.Saver import Saver


def expert_data(env, policy, expert_buffer, expert_saver):
    episode = 0
    total_step = 0
    buffer_start = 0

    while True:
        episode += 1
        episode_reward = 0
        local_step = 0

        done = False
        observation = env.reset()

        while not done:
            local_step += 1
            total_step += 1
            env.render()

            action = np.max(policy.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

            if total_step <= 5 * policy.batch_size:
                action = env.action_space.sample()

            next_observation, reward, done, _ = env.step(policy.max_action * action)
            episode_reward += reward

            policy.buffer.add(observation, action, reward, next_observation, done)
            observation = next_observation

        print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                 episode_reward))

        if total_step >= 5 * policy.batch_size:
            for i in range(local_step):
                s, a, r, ns, d = policy.buffer.sample()
                policy.train(s, a, r, ns, d)

        if local_step == 1000 and episode_reward > 700:
            for i in range(buffer_start+1, buffer_start + local_step):
                expert_buffer.add(policy.buffer.s[i], policy.buffer.a[i], policy.buffer.r[i], policy.buffer.ns[i],
                                       policy.buffer.d[i])

            print("episode {}: saved in expert buffer".format(episode))

            print(buffer_start, buffer_start + local_step, len(expert_buffer.s))

        if len(expert_buffer.s) >= 1000:
            expert_saver.buffer_save()
            print("expert buffer saved")

        buffer_start = buffer_start + local_step

if __name__ == '__main__':
    #env = gym.make("Pendulum-v0")
    # env = gym.make("InvertedDoublePendulumSwing-v2")
    # env = gym.make("InvertedDoublePendulum-v2")
    env = gym.make("InvertedPendulumSwing-v2")
    # env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    policy = SAC_v1.SAC(state_dim, action_dim, max_action, min_action, False, False)
    buffer = Buffer(policy.batch_size)
    saver = Saver([], [], buffer, '/home/cocel/PycharmProjects/SimpleRL/GAIL/expert_Invertedpendulumswing-v2')

    expert_data(env, policy, buffer, saver)
