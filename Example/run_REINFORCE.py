import argparse

from Algorithm.REINFORCE import REINFORCE

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import set_seed, cpu_only, gym_env, dmc_env, env_info, print_envs, print_args, discrete_env
def hyperparameters():
    parser = argparse.ArgumentParser(description='REINFORCE example')
    #environment
    parser.add_argument('--domain_type', default='gym', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='CartPole-v0', help='Pendulum-v0, MountainCarContinuous-v0, CartPole-v0')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=False, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=200, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #reinforce
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline only')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')

    parser.add_argument('--cpu-only', default=True, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)
    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)

    args.discrete = discrete_env(env)

    algorithm = REINFORCE(state_dim, action_dim, args)

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

