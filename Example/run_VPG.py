import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.VPG import VPG

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import set_seed, cpu_only, gym_env, dmc_env, env_info, discrete_env
from Common.Config import on_policy_config

def vpg_configurations(parser):
    parser.set_defaults(domain_type ='gym')
    parser.set_defaults(env_name ='InvertedPendulumSwing-v2')
    parser.set_defaults(render = False)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 10000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 0)
    parser.set_defaults(train_mode = 'offline')
    parser.set_defaults(training_step = 1)

    parser.set_defaults(log=False)

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

    args.discrete = discrete_env(env)
    algorithm = VPG(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = on_policy_config()
    parser = VPG.get_config(parser)
    parser = vpg_configurations(parser)
    args = parser.parse_args()
    main(args)

