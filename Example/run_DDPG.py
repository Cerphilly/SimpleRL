import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.DDPG import DDPG
from Common.Utils import env_info, gym_env, dmc_env, cpu_only, set_seed
from Common.Config import basic_config

from Trainer.Basic_trainer import Basic_trainer

def ddpg_configurations(parser):
    parser.set_defaults(domain_type ='gym')
    parser.set_defaults(env_name ='Pendulum-v1')
    parser.set_defaults(render = True)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 1000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 600)
    parser.set_defaults(batch_size = 256)
    parser.set_defaults(train_mode = 'online')
    parser.set_defaults(training_step = 1)
    parser.set_defaults(actor_lr=0.001)
    parser.set_defaults(critic_lr=0.001)
    parser.set_defaults(tau = 0.005)
    parser.set_defaults(noise_scale = 0.1)
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
    algorithm = DDPG(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = basic_config()
    parser = DDPG.get_config(parser)
    parser = ddpg_configurations(parser)
    args = parser.parse_args()
    print(args)

    main(args)


