import gym
import tensorflow as tf
import numpy as np
import random
import cv2

from gym.spaces import Box, Discrete
from collections import deque

def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def atari_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import gym
    from gym.wrappers import AtariPreprocessing, FrameStack
    env = gym.make(env_name)
    env = AtariPreprocessing(env, frame_skip=frame_skip, screen_size=image_size, grayscale_newaxis=False)
    env = FrameStack(env, frame_stack)

    env._max_episode_steps = 10000
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env, frame_skip=frame_skip, screen_size=image_size,
                                  grayscale_newaxis=False)
    test_env._max_episode_steps = 10000
    test_env = FrameStack(test_env, frame_stack)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def dmc_env(env_name, random_seed):
    import dmc2gym
    # deepmind control suite
    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)
    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)

    return env, test_env

def dmc_image_env(env_name, image_size, frame_stack, frame_skip, grayscale, random_seed):
    import dmc2gym
    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = FrameStack(env, k=frame_stack, grayscale=grayscale)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=frame_stack, grayscale=grayscale)

    return env, test_env

def dmc_diff_image_env(env_name, image_size, frame_stack, frame_skip, grayscale, random_seed):
    import dmc2gym
    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = DiffFrameStack(env, k=frame_stack, grayscale=grayscale)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = DiffFrameStack(test_env, k=frame_stack, grayscale=grayscale)

    return env, test_env

def procgen_env(env_name, frame_stack):
    import gym
    env_name = "procgen:procgen-{}-v0".format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    env._max_episode_steps = 1000
    env = FrameStack(env, frame_stack, data_format='channels_last')

    test_env = gym.make(env_name, render_mode='rgb_array')
    test_env._max_episode_steps = 1000
    test_env = FrameStack(test_env, frame_stack, data_format='channels_last')

    return env, test_env


def env_info(env):
    if isinstance(env.observation_space, Box):
        state_dim = env.observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
    elif isinstance(env.observation_space, Discrete):
        state_dim = env.observation_space.n

    else:
        raise NotImplementedError

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]

    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
        max_action = 1
        min_action = 1

    else:
        raise NotImplementedError

    return state_dim, action_dim, max_action, min_action

def discrete_env(env):
    if isinstance(env.action_space, Discrete):
        return True
    elif isinstance(env.action_space, Box):
        return False
    else:
        raise NotImplementedError


def render_env(env, env_name, domain_type, algorithm_name):
    if domain_type in {'gym', "atari"}:
        env.render()
    elif domain_type in {'procgen'}:
        cv2.imshow("{}_{}_{}".format(algorithm_name, domain_type, env_name),
                   env.render(mode='rgb_array'))
        cv2.waitKey(1)
    elif domain_type in {'dmc', 'dmc_image'}:
        cv2.imshow("{}_{}_{}".format(algorithm_name, domain_type, env_name),
                   env.render(mode='rgb_array', height=240, width=320))
        cv2.waitKey(1)


def copy_weight(network, target_network):
    variable1 = network.trainable_variables
    variable2 = target_network.trainable_variables

    for v1, v2 in zip(variable1, variable2):
        v2.assign(v1)


def soft_update(network, target_network, tau):
    variable1 = network.trainable_variables
    variable2 = target_network.trainable_variables

    for v1, v2 in zip(variable1, variable2):
        update = (1-tau)*v2 + tau*v1
        v2.assign(update)

def cpu_only():
    cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')
    tf.config.set_visible_devices([], 'GPU')

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed

def remove_argument(parser, args):
    #ref: https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            for group_action in action._group_actions:
                if group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    break

def modify_choices(parser, dest, choices):
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))

def modify_default(parser, dest, default):
    for action in parser._actions:
        if action.dest == dest:
            action.default = default
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == tf.float32
    if bits < 8:
        obs = tf.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + tf.random.uniform(shape=obs.shape) / bins
    obs = obs - 0.5
    return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, data_format='channels_first', grayscale=False):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        self.grayscale = grayscale
        self.data_format = data_format

        if data_format == 'channels_first':
            if grayscale:
                shp = (1,) + shp[1:]
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
            self.channel_first = True
        else:
            if grayscale:
                shp = shp[0:-1] + (1,)

            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(shp[0:-1] + (shp[-1] * k,)),
                dtype=env.observation_space.dtype
            )
            self.channel_first = False

        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        if self.grayscale:
            if self.data_format == 'channels_first':
                obs = cv2.cvtColor(np.transpose(obs, (1,2,0)), cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=0)
            else:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=-1)

        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.grayscale:
            if self.data_format == 'channels_first':
                obs = cv2.cvtColor(np.transpose(obs, (1,2,0)), cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=0)
            else:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=-1)

        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.channel_first == True:
            return np.concatenate(list(self._frames), axis=0)
        elif self.channel_first == False:
            return np.concatenate(list(self._frames), axis=-1)



class DiffFrameStack(gym.Wrapper):
    #env wrapper that provides image difference
    def __init__(self, env, k, data_format='channels_first', grayscale=False):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self._diff_frames = deque([], maxlen=k-1)

        self.data_format = data_format
        self.grayscale = grayscale

        shp = env.observation_space.shape

        if data_format == 'channels_first':
            if grayscale:
                shp = (1,) + shp[1:]

            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((shp[0] * (k - 1),) + shp[1:]),
                dtype=env.observation_space.dtype
            )
            self.channel_first = True
        else:
            if grayscale:
                shp = shp[0:-1] + (1,)

            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(shp[0:-1] + (shp[-1] * (k - 1),)),
                dtype=env.observation_space.dtype
            )
            self.channel_first = False

        self._max_episode_steps = env._max_episode_steps

    def reset(self):

        obs = self.env.reset()
        if self.grayscale:
            if self.data_format == 'channels_first':
                obs = cv2.cvtColor(np.transpose(obs, (1,2,0)), cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=0)
            else:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=-1)

        for _ in range(self._k):
            self._frames.append(obs)
        for i in range(self._k - 1):
            #self._diff_frames.append(self._frames[i+1] - self._frames[i])
            self._diff_frames.append(np.maximum(self._frames[i + 1], self._frames[i]))

        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.grayscale:
            if self.data_format == 'channels_first':
                obs = cv2.cvtColor(np.transpose(obs, (1,2,0)), cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=0)
            else:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=-1)

        #self._diff_frames.append(obs - self._frames[-1])
        self._diff_frames.append(np.maximum(obs, self._frames[-1]))
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.channel_first == True:
            return np.concatenate(list(self._diff_frames), axis=0)
        elif self.channel_first == False:
            return np.concatenate(list(self._diff_frames), axis=-1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    env, test_env = dmc_diff_image_env("cartpole_swingup", 100, 5, 8, 1234)
    print(env.reset().shape)
    for i in range(4):
        state, _, _, _ = env.step(np.array([1.0]))
    plt.imshow(np.transpose(env._frames[-1] - env._frames[-2], (1, 2, 0)))
    plt.show()
    print(env._frames[-1])
    print('---------------')
    print(env._frames[-2])
    print('---------------')
    plt.imshow(np.transpose(state[9:12], (1, 2, 0)))
    plt.show()

    plt.imshow(np.transpose(state[0:3], (1, 2, 0)))
    print(state[0:3])

    plt.show()



