import numpy as np
import gym
import tensorflow as tf
import random
import cv2
import os
from collections import deque
from skimage.util.shape import view_as_windows
from gym.spaces import Box, Discrete

def copy_weight(network, target_network):
    variable1 = network.trainable_variables
    variable2 = target_network.trainable_variables

    for v1, v2 in zip(variable1, variable2):
        v2.assign(v1)


def soft_update(network, target_network, tau):
    assert 0. < tau < 1.
    variable1 = network.trainable_variables
    variable2 = target_network.trainable_variables

    for v1, v2 in zip(variable1, variable2):
        update = (1 - tau) * v2 + tau * v1
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

def save_weights(algorithm, path):
    for name, network in algorithm.network_list.items():
        network.save_weights(os.path.join(path, name))

def load_weights(algorithm, path):
    for name, network in algorithm.network_list.items():
        network.load_weights(os.path.join(path, name))


#####################################################################
def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed + 1)
    test_env.action_space.seed(random_seed + 1)

    return env, test_env

def atari_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    #channel_first
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
    test_env.seed(random_seed + 1)
    test_env.action_space.seed(random_seed + 1)

    return env, test_env

def dmc_env(env_name, random_seed):
    import dmc2gym
    # deepmind control suite
    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)
    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed + 1)

    return env, test_env

def dmc_image_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    #channel_first env
    import dmc2gym
    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = FrameStack(env, k=frame_stack)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed + 1, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=frame_stack)

    return env, test_env

def dmcr_env(env_name, image_size, frame_stack, frame_skip, random_seed, mode='classic'):
    #https://github.com/jakegrigsby/dmc_remastered
    #channel_first env
    '''
    A version of the DeepMind Control Suite with randomly generated graphics, for measuring visual generalization in continuous control.
    '''
    assert mode in {'classic', 'generalization', 'sim2real'}

    import dmc_remastered as dmcr

    domain_name = env_name.split('_')[0]
    task_name = env_name.split('_')[1]
    if mode == 'classic':#loads a training and testing environment that have the same visual seed
        env, test_env = dmcr.benchmarks.classic(domain_name, task_name, visual_seed=random_seed, frame_stack=frame_stack, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'generalization':#creates a training environment that selects a new visual seed from a pre-set range after every reset(), while the testing environment samples from visual seeds 1-1,000,000
        env, test_env = dmcr.benchmarks.visual_generalization(domain_name, task_name, num_levels=100, frame_stack=frame_stack, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'sim2real':#approximates the challenge of transferring control policies from simulation to the real world by measuring how many distinct training levels the agent needs access to before it can succesfully operate in the original DMC visuals that it has never encountered.
        env, test_env = dmcr.benchmarks.visual_sim2real(domain_name, task_name, num_levels=random_seed, frame_stack=frame_stack, width=image_size, height=image_size, frame_skip=frame_skip)

    return env, test_env

def procgen_env(env_name, frame_stack):
    #channel_last env
    import gym
    env_name = "procgen:procgen-{}-v0".format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    print(env.reset().shape)
    env._max_episode_steps = 1000
    env = FrameStack(env, frame_stack, data_format='channels_last')

    test_env = gym.make(env_name, render_mode='rgb_array')
    test_env._max_episode_steps = 1000
    test_env = FrameStack(test_env, frame_stack, data_format='channels_last')

    return env, test_env


#####################################################################


def preprocess_obs(obs, bits=5):

    """Preprocessing image, see https://arxiv.org/abs/1807.03039.
    Used in SAC-AE"""

    bins = 2**bits
    assert obs.dtype == tf.float32
    if bits < 8:
        obs = tf.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + tf.random.uniform(shape=obs.shape) / bins
    obs = obs - 0.5
    return obs
# def random_crop(imgs, output_size, data_format='channels_first'):#random crop for curl
#     """
#     Vectorized way to do random crop using sliding windows
#     and picking out random ones
#
#     args:
#         imgs, batch images with shape (B,C,H,W)
#     """
#     # batch size
#     n = imgs.shape[0]
#     img_size = imgs.shape[-1]
#     crop_max = img_size - output_size
#     imgs = np.transpose(imgs, (0, 2, 3, 1))
#
#     w1 = np.random.randint(0, crop_max, n)
#     h1 = np.random.randint(0, crop_max, n)
#     # creates all sliding windows combinations of size (output_size)
#     windows = view_as_windows(
#         imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
#     # selects a random window for each batch element
#     cropped_imgs = windows[np.arange(n), w1, h1]
#     return cropped_imgs

# def center_crop_image(image, output_size):#center crop for curl, rad
#     #image type must be channel_first
#     h, w = image.shape[1:]
#     new_h, new_w = output_size, output_size
#
#     top = (h - new_h)//2
#     left = (w - new_w)//2
#
#     image = image[:, top:top + new_h, left:left + new_w]
#     return image
#
# def center_crop_images(image, output_size):
#     #image type must be channel_first
#     h, w = image.shape[2:]
#     new_h, new_w = output_size, output_size
#
#     top = (h - new_h)//2
#     left = (w - new_w)//2
#
#     image = image[:, :, top:top + new_h, left:left + new_w]
#     return image
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

def find_channel(domain_type):
    if domain_type in {'gym', 'dmc'}:
        return None
    elif domain_type in {'dmc_image', 'atari', 'dmcr'}:
        return 'channels_first'
    elif domain_type in {'procgen'}:
        return 'channels_last'
    else:
        raise ValueError

def trim_float(dictionary, r=3):
    for key, value in dictionary.items():
        for key2, value2 in value.items():
            value[key2] = round(value[key2], r)


def render_env(env, env_name, domain_type, algorithm_name):
    if domain_type in {'gym', "atari"}:
        env.render()
    elif domain_type in {'procgen'}:
        cv2.imshow("{}_{}_{}".format(algorithm_name, domain_type, env_name),
                   env.render(mode='rgb_array'))
        cv2.waitKey(1)
    elif domain_type in {'dmc', 'dmc_image', 'dmcr'}:
        cv2.imshow("{}_{}_{}".format(algorithm_name, domain_type, env_name),
                   env.render(mode='rgb_array', height=240, width=320))
        cv2.waitKey(1)



class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_args(args):
    dict_args = vars(args)
    print(color.BOLD + "{:<20}".format("Variable") + color.END, "{:<20}".format("Value"))
    print('=' * 40)
    for k, v in dict_args.items():
        print(color.BOLD + "{:<20}".format(k) + color.END, "{:<20}".format(str(v)))

def print_networks(algorithm):
    for k, v in algorithm.network_list.items():
        v.summary()

def build_networks(algorithm):
    for k, v in algorithm.network_list.items():
        v.build_network()

def print_envs(algorithm, max_action, min_action, args):
    print('=' * 40)
    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    try:
        state_dim = algorithm.state_dim
    except:
        state_dim = algorithm.obs_dim

    print("State dim:", state_dim)
    print("Action dim:", algorithm.action_dim)
    print("Max Action:", max_action)
    print("Min Action:", min_action)
    print('=' * 40)


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
        if self.channel_first:
            return np.concatenate(list(self._frames), axis=0)

        else:
            return np.concatenate(list(self._frames), axis=-1)



if __name__ == '__main__':
    #env, test_env = atari_env('PongNoFrameskip-v4', 100, 3, 4, 1234)
    env, test_env = procgen_env('coinrun',3, 1234)

    print(env.reset().shape)


