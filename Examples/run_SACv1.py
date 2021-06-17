import gym, dmc2gym
import numpy as np
import argparse
import tensorflow as tf
import random

from Algorithms.SAC_v1 import SAC_v1

from Trainer.State_trainer import State_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v1 example')
    #environment
    parser.add_argument('--domain_type', default='dmc', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole_swingup', help='Pendulum-v0, MountainCarContinuous-v0, cartpole_swingup')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=10000, type=int, help='First step to start training')
    parser.add_argument('--max-episode', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval-step', default=200, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='offline, online')
    parser.add_argument('--training-step', default=1, type=int, help='set this to 1 if online training')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--v-lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--console', default=False, type=bool, help='when logged, write in console')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    # random seed setting
    if args.random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = args.random_seed

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    #env setting
    if len(args.env_name.split('_')) == 1:
        #openai gym
        env = gym.make(args.env_name)
        env.seed(random_seed)
        env.action_space.seed(random_seed)
    else:
        #deepmind control suite
        env = dmc2gym.make(domain_name=args.env_name.split('_')[0], task_name=args.env_name.split('_')[1], seed=random_seed)#cartpole 존나 안됨
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    algorithm = SAC_v1(state_dim, action_dim, args)

    print("Training of", env.unwrapped.spec.id)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    trainer = State_trainer(env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

#가능성: mu와 std 레이어 분리
# log_prob랑 action 같이 내놓기(log_prob 삭제)