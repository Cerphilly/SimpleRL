import numpy as np
import gym
import cv2

from SAC import SAC_v1
from SAC import SAC_v1

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstate, dmstep

from dm_control import suite


def data_len(expert_directory):
    file_path = '{}/buffer'.format(expert_directory)
    loaded = np.load('{}.npz'.format(file_path))

    print("Data # of {}: {}".format(expert_directory,len(loaded['s'])))


def expert_data(env, policy, expert_buffer, expert_saver):
    episode = 0
    total_step = 0
    buffer_start = 0

    if policy.load == True:
        policy.saver.load()
        print("network data loaded")

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

        if episode_reward >= 800:
            for i in range(buffer_start, buffer_start + local_step-1):
                expert_buffer.add(policy.buffer.s[i], policy.buffer.a[i], policy.buffer.r[i], policy.buffer.ns[i],
                                       policy.buffer.d[i])

            print("episode {}: saved in expert buffer".format(episode))

            print(buffer_start, buffer_start + local_step, len(expert_buffer.s))

        if len(expert_buffer.s) % 10000 == 0 and len(expert_buffer.s) > 1000:
            expert_saver.buffer_save()
            print("expert buffer saved")

        if episode % 1000 == 0:
            policy.saver.save()
            print("temporary save")

        buffer_start = buffer_start + local_step

def expert_data_dm(env, policy, expert_buffer, expert_saver):
    episode = 0
    total_step = 0
    buffer_start = 0

    height = 480
    width = 640

    video = np.zeros((1001, height, width, 3), dtype=np.uint8)

    while True:
        episode += 1
        episode_reward = 0
        local_step = 0

        done = False
        observation = dmstate(env.reset())

        while not done:
            local_step += 1
            total_step += 1

            x = env.physics.render(height=480, width=640, camera_id=0)
            video[local_step] = x

            action = np.max(policy.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

            if total_step <= 10 * policy.batch_size:
                action = np.random.uniform(min_action, max_action)

            next_observation, reward, done = dmstep(env.step(policy.max_action*action))
            episode_reward += reward

            policy.buffer.add(observation, action, reward, next_observation, done)
            observation = next_observation

            cv2.imshow('result', video[local_step - 1])
            cv2.waitKey(1)
            if local_step == 1000: done = True

        print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                 episode_reward))

        if total_step >= 10 * policy.batch_size:
            for i in range(local_step):
                s, a, r, ns, d = policy.buffer.sample()
                policy.train(s, a, r, ns, d)

        if episode_reward >= 700:
            for i in range(buffer_start, buffer_start + local_step-1):
                expert_buffer.add(policy.buffer.s[i], policy.buffer.a[i], policy.buffer.r[i], policy.buffer.ns[i],
                                       policy.buffer.d[i])

            print("episode {}: saved in expert buffer".format(episode))

            print(buffer_start, buffer_start + local_step, len(expert_buffer.s))

        if len(expert_buffer.s) % 10000 == 0 and len(expert_buffer.s) > 1000:
            expert_saver.buffer_save()
            print("expert buffer saved")

        if episode % 2 == 0:
            policy.saver.save()
            print("temporary save")

        buffer_start = buffer_start + local_step

if __name__ == '__main__':

    #env = gym.make("Pendulum-v0")
    env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    '''
    # env = suite.load(domain_name="cartpole", task_name="three_poles")#300만 스텝 학습: SAC_Test
    # env = suite.load(domain_name="cartpole", task_name="two_poles")
    #env = suite.load(domain_name="acrobot", task_name="swingup")

    env = suite.load(domain_name="cartpole", task_name="swingup")
    state_spec = env.reset()
    action_spec = env.action_spec()
    state_dim = len(dmstate(state_spec))
    print(dmstate(state_spec))
    action_dim = action_spec.shape[0]  # 1
    max_action = action_spec.maximum[0]  # 1.0
    min_action = action_spec.minimum[0]
    '''
    policy = SAC_v1.SAC(state_dim, action_dim, max_action, min_action, True, True)
    buffer = Buffer(policy.batch_size)
    saver = Saver([], [], buffer, '/home/cocel/PycharmProjects/SimpleRL/GAIL/expert_InvertedDoublePendulumSwing-v2')

    expert_data(env, policy, buffer, saver)
