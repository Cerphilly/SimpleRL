import argparse

from Algorithm.DQNs.DDQN import DDQN
from Common.Utils import cpu_only, set_seed, gym_env, env_info, print_envs, print_args

from Trainer.Basic_trainer import Basic_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Double Deep Q Network(DDQN) example')
    #environment
    parser.add_argument('--domain_type', default='gym', type=str, help='gym')
    parser.add_argument('--env-name', default='CartPole-v0', help='CartPole-v0, MountainCar-v0, Acrobot-v1, and atari games(not yet)')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=100, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=1000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #dqn
    parser.add_argument('--batch-size', default=128, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='Offline, Online')
    parser.add_argument('--training-step', default=100, type=int)
    parser.add_argument('--copy-iter', default=5, type=int, help='Frequency to update target network')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float, help='Exploration probability')
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)
    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)

    algorithm = DDQN(state_dim, action_dim, args)

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

