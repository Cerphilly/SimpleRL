import datetime
import os
from scipy.io import savemat, loadmat
import tensorflow as tf
import glob, pickle

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

class AverageMeter:
    def __init__(self, name=None):
        self.value = 0
        self.count = 0

        self.name = name

    def __call__(self, value, n=1):
        self.value += value
        self.count += n

    def result(self):
        return round(self.value / max(1, self.count), 4)

    def reset(self):
        self.value = 0
        self.count = 0

class SumMeter:
    def __init__(self, name=None):
        self.value = 0
        self.count = 0

        self.name = name

    def __call__(self, value, n=1):
        self.value += value
        self.count += n

    def result(self):
        return round(self.value, 4)

    def reset(self):
        self.value = 0
        self.count = 0


class ListMeter:
    def __init__(self, name=None):
        self.counts = []

        self.name = name

    def __call__(self, step):
        self.counts.append(step)

    def result(self):
        return round(self.counts[-1], 4)

    def reset(self):
        self.counts = []

class ValueMeter:
    def __init__(self, name=None):
        self.value = 0
        self.name = name

    def __call__(self, value):
        self.value = value

    def result(self):
        return round(self.value, 4)

    def reset(self):
        self.value = 0

METER_DICT = {'step': ValueMeter, 'reward': ValueMeter, 'loss': AverageMeter}

class Logger:
    def __init__(self, env, test_env, algorithm, max_action, min_action, args, path=None):
        self.env = env
        self.test_env = test_env
        self.algorithm = algorithm
        self.args = args
        self.path = path

        print_envs(algorithm, max_action, min_action, args)
        print_args(args)
        print_networks(algorithm)

        self.train_dict = {}
        self.eval_dict = {}

        self.file = args.file
        self.tensorboard = args.tensorboard

        self.hyp_name = None
        self.model_dir = None
        self.buffer_dir = None

        self.model = args.save_model
        self.buffer = args.save_buffer
        self.log = args.log

        if args.log == True:
            if path == None:
                self.current_time = self.log_time()
                self.dir_name = 'D:/cocelRL_Logs/Logs/' + args.domain_type + '-' + args.env_name + '-' +  algorithm.name + '-' +  self.current_time + "/"
                if not os.path.exists(self.dir_name): os.mkdir(self.dir_name)
            else:
                self.dir_name = path

            if self.file:
                self.hyp_name = os.path.join(self.dir_name, "hyperparameter")
                self.train_result_name = os.path.join(self.dir_name, "train")
                self.eval_result_name = os.path.join(self.dir_name, "eval")

                self.log_hyperparameters()

            if self.model:
                self.model_dir = os.path.join(self.dir_name, "models")
                if not os.path.exists(self.model_dir):
                    os.mkdir(self.model_dir)

            if self.buffer:
                self.buffer_dir = os.path.join(self.dir_name, "buffer")
                if not os.path.exists(self.buffer_dir):
                    os.mkdir(self.buffer_dir)

            if self.tensorboard:
                self.tensorboard_dir = os.path.join(self.dir_name, "tensorboard")
                if not os.path.exists(self.tensorboard_dir):
                    os.mkdir(self.tensorboard_dir)
                self.writer = tf.summary.create_file_writer(logdir=self.tensorboard_dir)


    def log_time(self):
        #YYYY-M-D-H-M-S
        current_time = datetime.datetime.now()
        output = str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day) + '-'
        output += str(current_time.hour) + '-' + str(current_time.minute)

        return output

    def log_values(self, log_list, mode='train'):
        if mode == 'train':
            log_dict = self.train_dict
        elif mode == 'eval':
            log_dict = self.eval_dict

        else:
            raise ValueError

        for value in log_list:
            if value[0] not in log_dict.keys():
                if value[0].split('/')[0].lower() in METER_DICT.keys():
                    log_dict[value[0]] = METER_DICT[value[0].split('/')[0].lower()](name=value[0])
                    log_dict[value[0]](value[1])
                else:
                    log_dict[value[0]] = ValueMeter(name=value[0])
                    log_dict[value[0]](value[1])
            else:
                log_dict[value[0]](value[1])

    def results(self, mode):
        if mode == 'train':
            log_dict = self.train_dict
        elif mode == 'eval':
            log_dict = self.eval_dict

        else:
            raise ValueError

        results = []

        for key, value in log_dict.items():
            results.append([key, value.result()])
            value.reset()

        console = self.console_results(results, mode)
        if self.log == True:
            if self.tensorboard == True:
                self.tensorboard_results(results)
            if self.file == True:
                self.save_results(console, mode)

            if mode == 'eval':
                self.save_networks()
                self.save_buffer()

        return results

    def console_results(self, results, mode):

        console = ""
        console += "{:<6}".format(mode.upper()) + "|"
        for result in results:
            console += "{}: {:<8} |".format(result[0], result[1])

        print(console)

        return console

    def tensorboard_results(self, results):
        episode = 0
        with self.writer.as_default():
            for result in results:
                if 'episode' in result[0].lower():
                    episode = result[1]
                    continue
                tf.summary.scalar(result[0], result[1], episode)

    def save_results(self, results, mode):
        if mode == 'train':
            f = open(self.train_result_name + ".txt", "a")
           # f2 = open(self.train_result_name + ".pickle", "wb")
        elif mode == 'eval':
            f = open(self.eval_result_name + ".txt", "a")
            #f2 = open(self.eval_result_name + ".pickle", "wb")

        else:
            raise ValueError

        f.write(results + "\n")
        f.close()

        #pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        #f2.close()


    def log_hyperparameters(self):
        if self.hyp_name == None:
            return
        dict_args = vars(self.args)

        f = open(self.hyp_name + ".txt", "w")

        f.write("{:<20}".format("Variable") + "{:<20}".format("Value") + "\n")
        f.write('=' * 40 + "\n")
        for k, v in dict_args.items():
            f.write("{:<20}".format(k) + "{:<20}".format(str(v)) + "\n")

        f.close()


    def save_networks(self):
        if self.model_dir == None:
            return
        networks = self.algorithm.network_list

        for name, network in networks.items():
            network.save_weights(os.path.join(self.model_dir, "{}".format(name)), save_format='h5')

    def save_buffer(self, total_step=None):
        if self.buffer_dir == None:
            return
        buffer_dict = self.algorithm.buffer.export()
        if total_step != None:
            savemat(os.path.join(self.buffer_dir, "buffer_{}.mat".format(total_step)), buffer_dict)
        else:
            savemat(os.path.join(self.buffer_dir, "buffer.mat"), buffer_dict)

    def load_buffer(self, total_step=None):
        buffers = glob.glob(self.buffer_dir + "*.mat")
        if len(buffers) == 1:
            buffer = loadmat(buffers[0])








if __name__ == '__main__':
    a = AverageMeter()
    a(1)
    a(2)
    print(a.result())


