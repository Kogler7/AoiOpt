import argparse
import os
import copy
from copy import deepcopy
# import pandas as pd


class KeyConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='AOIopt')
        # core options
        self.parser.add_argument('--mode', type=str, default='train_test', choices=['train_test', 'train', 'test'],
                                 help='the mode to run the code: train_test, train, test')
        self.parser.add_argument('--name', type=str, help='the name of the data: n_n', required=True)
        self.parser.add_argument('--env_type', type=str, choices=['synthetic','real'],required=True)
        self.parser.add_argument('--device', type=int, help='-1 is CPU,others is GPU', default=-1, required=False)
        self.parser.add_argument('--model_name', type=str, help='model name', required=True)
        self.parser.add_argument('--tra_eps', type=int, default=1000, help='the training number of episodes')
        self.parser.add_argument('--seed', type=int, default=3047, help='the random seed used')
        self.parser.add_argument('--max_erg', type=int, default=4, help='the max number of traversing')
        self.parser.add_argument('--traj_reward_weight', type=float, help='the weight of trajactory reward', required=False)
        self.parser.add_argument('--road_reward_weight', type=float, help='the weight of road reward', required=False)  #
        self.parser.add_argument('--start_with_road', help='start from road aoi', action='store_true', default=False)
        self.parser.add_argument('--init_param', help='initial parameter of the model', type=str, required=False)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


class MultiConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='MultiAOI')
        self.parser.add_argument('--name', type=str, help='the name of the data: n_n', required=True)
        self.parser.add_argument('--env_type', type=str, choices=['synthetic','real'], required=True)
        self.parser.add_argument('--max_erg', nargs='+', type=int, help='the max number of traversing', required=False)
        self.parser.add_argument('--tra_eps', nargs='+', type=int, help='the training number of episodes', required=False)
        self.parser.add_argument('--seed', nargs='+', type=int, help='the seed used', required=False)
        self.parser.add_argument('--traj_reward_weight', nargs='+', type=float, help='the weight of trajactory reward', required=False)
        self.parser.add_argument('--road_reward_weight', nargs='+', type=float, help='the weight of road reward', required=False)  #

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

    def get_config(self):
        options = self.parser.parse_args()
        bool_name = ['input_norm', 'end_reward', 'fixed_eps', 'matrix_norm']
        single_value = ['name','env_type','datas']
        print(vars(options).values())
        opts = []
        opt = option()
        opt.mode = 'train'
        opt.name = options.name
        opt.env_type = options.env_type

        keys, values = list(vars(options).keys()), list(vars(options).values())
        indexes = [0] * len(keys)
        # get all different parameters.
        multi_index_lst = []
        for i, s in enumerate(keys):
            if values[i] is not None and s not in single_value:
                multi_index_lst.append(i)
        print('multi config index is ', multi_index_lst)
        multi_config_num = len(multi_index_lst)

        if options.datas:
            opt.datas = options.datas
        else:
            opt.datas = [1]

        while indexes[multi_index_lst[0]] < len(values[multi_index_lst[0]]):
            opt.model_name = 'model'
            for i in multi_index_lst:
                opt.model_name += '_' + str(keys[i][0]) + str(values[i][indexes[i]])  #
                if keys[i] in bool_name:
                    value = True if values[i][indexes[i]] == 1 else False
                else:
                    value = values[i][indexes[i]]
                setattr(opt, keys[i], value)
            print(vars(opt).values())
            opts.append(deepcopy(opt))

            cur = multi_config_num - 1
            index = multi_index_lst[cur]
            indexes[index] += 1
            while indexes[index] >= len(values[index]):
                indexes[index] = 0
                cur -= 1
                if cur < 0:
                    return opts
                index = multi_index_lst[cur]
                indexes[index] += 1
            print(indexes)
        return opts


class option:
    def __init__(self):
        pass


if __name__ == '__main__':
    options = MultiConfig()
    opts = options.get_config()

    keys = list(vars(opts[0]).keys())
    datas = {}
    for key in keys:
        datas[key] = []
        for opt in opts:
            value = getattr(opt, key)
            if type(value) == list:
                value = [str(v) for v in value]
                value = ' '.join(value)
            datas[key].append(value)
    df = pd.DataFrame(datas)

    df.to_csv('task.csv', index=False)
