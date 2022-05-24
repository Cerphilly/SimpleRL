import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common.Utils import gym_env, env_info, cpu_only, set_seed
from Common.Config import basic_config
from Algorithm.DQNs.DQN import DQN

from Trainer.Basic_trainer import Basic_trainer

def dqn_configurations(parser):
    parser.set_defaults(domain_type ='gym')
    parser.set_defaults(env_name ='CartPole-v0')
    parser.set_defaults(render = True)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 1000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 200)
    parser.set_defaults(batch_size = 256)
    parser.set_defaults(train_mode = 'offline')
    parser.set_defaults(training_step = 200)
    parser.set_defaults(copy_iter = 100)
    parser.set_defaults(epsilon = 0.1)
    return parser

def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    else:
        raise ValueError("only gym allowed")


    state_dim, action_dim, max_action, min_action = env_info(env)
    algorithm = DQN(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = basic_config()
    parser = DQN.get_config(parser)
    parser = dqn_configurations(parser)
    args = parser.parse_args()
    print(args)
    main(args)

