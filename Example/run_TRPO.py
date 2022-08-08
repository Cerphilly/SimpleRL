import argparse

from Algorithm.TRPO import TRPO

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, discrete_env, env_info, print_envs, print_args

def hyperparameters():
    parser = argparse.ArgumentParser(description='Trusted Region Policy Optimization(TRPO) example')
    #environment
    parser.add_argument('--domain_type', default='gym', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='CartPole-v0', help='Pendulum-v0, MountainCarContinuous-v0, CartPole-v0')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=0, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=False, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=200, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #ppo
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline')
    parser.add_argument('--backtrack-iter', default=10, type=int)
    parser.add_argument('--backtrack-coeff', default=0.8, type=float)
    parser.add_argument('--delta', default=0.5, type=float)
    parser.add_argument('--training-step', default=5, type=int, help='critic learning epoch')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda-gae', default=0.95, type=float)
    parser.add_argument('--actor-lr', default=0.0003, type=float)
    parser.add_argument('--critic-lr', default=0.0003, type=float)
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

    algorithm = TRPO(state_dim, action_dim, args)

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

