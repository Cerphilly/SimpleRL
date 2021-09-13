import numpy as np
import gym, dmc2gym
import tensorflow as tf
import random

from collections import deque
from skimage.util.shape import view_as_windows

from Network.Basic_Networks import *
from Network.Gaussian_Actor import *
from Network.D2RL_Networks import *



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
    env = AtariPreprocessing(env, frame_skip=frame_skip, screen_size=image_size, grayscale_newaxis=True)
    env = FrameStack(env, frame_stack)

    env._max_episode_steps = 10000
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env, frame_skip=frame_skip, screen_size=image_size,
                                  grayscale_newaxis=True)
    test_env._max_episode_steps = 10000
    test_env = FrameStack(test_env, frame_stack)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def dmc_env(env_name, random_seed):
    import dmc2gym
    # deepmind control suite
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)
    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)

    return env, test_env

def dmc_image_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import dmc2gym
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = FrameStack(env, k=frame_stack)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=frame_stack)

    return env, test_env

def dmcr_env(env_name, image_size, frame_skip, random_seed, mode='classic'):
    assert mode in {'classic', 'generalization', 'sim2real'}

    import dmc_remastered as dmcr

    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    if mode == 'classic':#loads a training and testing environment that have the same visual seed
        env, test_env = dmcr.benchmarks.classic(domain_name, task_name, visual_seed=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'generalization':#creates a training environment that selects a new visual seed from a pre-set range after every reset(), while the testing environment samples from visual seeds 1-1,000,000
        env, test_env = dmcr.benchmarks.visual_generalization(domain_name, task_name, num_levels=100, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'sim2real':#approximates the challenge of transferring control policies from simulation to the real world by measuring how many distinct training levels the agent needs access to before it can succesfully operate in the original DMC visuals that it has never encountered.
        env, test_env = dmcr.benchmarks.visual_sim2real(domain_name, task_name, num_levels=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)

    return env, test_env

def procgen_env(env_name, frame_stack, random_seed):
    import gym
    env_name = "procgen:procgen-{}-v0".format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    env._max_episode_steps = 1000
    env = FrameStack(env, frame_stack, data_format='channels_last')

    test_env = gym.make(env_name, render_mode='rgb_array')
    test_env._max_episode_steps = 1000
    test_env = FrameStack(test_env, frame_stack, data_format='channels_last')

    return env, test_env

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


def random_crop(imgs, output_size, data_format='channels_first'):#random crop for curl
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):#center crop for curl
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, data_format='channels_first'):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        if data_format == 'channels_first':
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
            self.channel_first = True
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(shp[0:-1] + (shp[-1] * k,)),
                dtype=env.observation_space.dtype
            )
            self.channel_first = False

        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.channel_first == True:
            return np.concatenate(list(self._frames), axis=0)
        elif self.channel_first == False:
            return np.concatenate(list(self._frames), axis=-1)

