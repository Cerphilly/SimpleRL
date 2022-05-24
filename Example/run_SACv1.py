import argparse
from Algorithm.SAC_v1 import SAC_v1

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, env_info
from Common.Config import basic_config

def sac_v1_configurations(parser):
    parser.set_defaults(domain_type ='gym')
    parser.set_defaults(env_name ='InvertedPendulumSwing-v2')
    parser.set_defaults(render = False)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 10000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 1000)
    parser.set_defaults(batch_size = 256)
    parser.set_defaults(train_mode = 'offline')
    parser.set_defaults(training_step = 200)
    parser.set_defaults(actor_lr=0.001)
    parser.set_defaults(critic_lr=0.001)
    parser.set_defaults(v_lr=0.001)
    parser.set_defaults(tau = 0.005)
    parser.set_defaults(alpha = 0.2)

    return parser

def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    else:
        raise ValueError("only gym and dmc allowed")

    state_dim, action_dim, max_action, min_action = env_info(env)

    algorithm = SAC_v1(state_dim, action_dim, args)

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = basic_config()
    parser = SAC_v1.get_config(parser)
    parser = sac_v1_configurations(parser)
    args = parser.parse_args()
    print(args)

    main(args)
