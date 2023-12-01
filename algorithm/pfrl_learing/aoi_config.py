import glob
import time

import torch
import os.path
import numpy as np

from utils.data_creater import *
from utils.read_real import read_real_data
import utils.data_loader as dl


class Config:
    def __init__(self, options):
        self.opt = options
        self.device = torch.device(f'cuda:{options.device}') if hasattr(options, 'device') and \
                                                                options.device != -1 else torch.device("cpu")
        self.seed = options.seed if hasattr(options, 'seed') and options.seed else 3047
        self.load_model = False  # whether the agent load model or not
        self.env_type = options.env_type if hasattr(options, 'env_type') else 'synthetic'  # dataset
        # the path of the model saved
        if self.env_type == 'synthetic':
            self.save_model = os.path.abspath('output/synthetic')
        else:
            self.save_model = os.path.abspath('output/real')

        self.model_name = self.opt.model_name if hasattr(options, 'model_name') else 'model'  # model name
        self.train = True  # train mode
        self.test = False  # test mode

        self.name = 'pfrl_DoubleDQN_' + self.opt.name  # values name
        self.save_dir = os.path.join(self.save_model, self.name)
        self.fig_name = self.model_name  # training fig path
        self.fig_path = os.path.join(self.save_dir, self.fig_name + '.png')
        self.reward_fig = True  # draw the reward figure
        self.log_path = os.path.join(self.save_dir, 'log')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path = os.path.join(self.log_path, self.fig_name)  # training log dir
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.save_model = self.log_path
        self.save_param = os.path.abspath(os.path.join(self.save_model, self.model_name + '.pth'))  # param save path
        self.init_param = self.opt.init_param if hasattr(options,
                                                         'init_param') and options.init_param else self.save_param  # init param path

        self.batch_size = self.opt.batch if hasattr(options, 'batch') else 64  # batch size
        self.tra_eps = self.opt.tra_eps if hasattr(options, 'tra_eps') else 500  # training episode
        self.fixed_eps = True  # The step number is constant in each episode
        self.max_erg = self.opt.max_erg if self.fixed_eps else 7
        self.input_norm = True  # Normalize the input state
        self.play_speed = 0.05  # platform's playing speed. It will affect the training effciency.
        self.signal = None  # platform's signal, it could transfer aoi data to the platform.

        self.agent_type = 'DoubleDQN'
        self.gamma = 0.9  # discount factor
        self.lr = 1e-4  # learning rate

        self.dir_h, self.dir_w = [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0]  # direction array

        self.start_with_road = self.opt.start_with_road if hasattr(options,
                                                                   'start_with_road') else False  # begin from road network
        if self.env_type == 'synthetic':
            # synthetic data
            data_dir = './data/synthetic_data/' + self.opt.name  # data directory
            AOI_path = os.path.abspath(os.path.join(data_dir, 'aoi.csv'))  # initial AOI path
            traj_path = os.path.abspath(os.path.join(data_dir, 'traj.npy'))  # trajectory data path
            matrix_path = os.path.abspath(
                os.path.join(data_dir, f'matrix.npy'))  # matrix data path. it's the adjacency matrix of each grid
            map_path = os.path.abspath(os.path.join(data_dir,
                                                    f'map.npy'))  # map data path. It records the trajectory number of 2 direction(down/right) in each grid.
            road_aoi_path = os.path.abspath(os.path.join(data_dir, f'road_aoi.csv'))  # road-network segmentation path
            self.AOI_path, self.traj_path, self.matrix_path, self.map_path, self.road_aoi_path = AOI_path, traj_path, matrix_path, map_path, road_aoi_path
            grid = GetGrid(file=AOI_path)  # read initial AOI
            traj = GetTrajectory()  # read trajectories
            traj.get_from_file(file=traj_path)
            mmgetter = GetMatrixAndMap(grid.h, grid.w)
            matrix = mmgetter.get_matrix_from_file(matrix_path)  # read matrix
            map = dl.load(map_path)  # read map
            net_map = np.zeros([4, grid.h, grid.w],
                               dtype=np.float32)  # net map. It records the trajectory number of 4 dirction(up/down/left/right)
            for i in range(grid.h):
                for j in range(grid.w):
                    for k in range(4):
                        i1, j1 = i + self.dir_h[k], j + self.dir_w[k]
                        if 0 <= i1 < grid.h and 0 <= j1 < grid.w:
                            net_map[k, i, j] = matrix[i * grid.w + j, i1 * grid.w + j1] + matrix[
                                i1 * grid.w + j1, i * grid.w + j]
            net_map = (net_map - net_map.mean()) / net_map.std()
            self.grid = torch.from_numpy(grid.grid).int()
            road_aoi = pd.read_csv(self.road_aoi_path, header=None).values
            self.h, self.w = grid.h, grid.w
            self.matrix, self.map, self.net_map,self.road_aoi = torch.from_numpy(matrix).to(self.device), torch.from_numpy(map).to(
                self.device), torch.from_numpy(net_map).to(self.device),torch.from_numpy(road_aoi).to(self.device)
        else:
            # real world data
            self.data_dir = './data/real_data/' + self.opt.name
            self.AOI_path = os.path.join(self.data_dir, 'aoi.npy')
            self.matrix_path = os.path.join(self.data_dir, 'matrix.npy')
            self.map_path = os.path.join(self.data_dir, 'map.npy')
            self.road_aoi_path = os.path.abspath(os.path.join(self.data_dir, f'road_aoi.npy'))
            # load datas
            if not os.path.exists(self.matrix_path):
                read_start = time.time()
                [self.h, self.w], matrix, map, parcle_num, parcels = read_real_data(delta_lon=0.0001875,
                                                                                    delta_lat=0.0001739,
                                                                                    lon_range=[121.4910, 121.4938125],
                                                                                    lat_range=[31.1606522, 31.1576956],
                                                                                    )
                read_end = time.time()
                print(f'read used {read_end - read_start:.2f}s.')
                np.save(self.map_path, map)
                if self.start_with_road:
                    state = np.load(self.AOI_path)
                else:
                    state = np.zeros([self.h, self.w], dtype=np.int16) + 1
                road_aoi = np.load(self.road_aoi_path)
                assert road_aoi.shape == state.shape
            else:
                grid = GetGrid(file=self.AOI_path)
                mmgetter = GetMatrixAndMap(grid.h, grid.w)
                matrix = mmgetter.get_matrix_from_file(self.matrix_path)
                map = dl.load(self.map_path)
                if self.start_with_road:
                    state = grid.grid + 1
                else:
                    state = np.arange(grid.h * grid.w).reshape([grid.h, grid.w]) + 1
                self.h, self.w = map.shape[0], map.shape[1]
                road_aoi = np.load(self.road_aoi_path)

            if self.start_with_road:
                print('start from road network aoi.')

            net_map = np.zeros([4, self.h, self.w], dtype=np.float32)
            for i in range(self.h):
                for j in range(self.w):
                    for k in range(4):
                        i1, j1 = i + self.dir_h[k], j + self.dir_w[k]
                        if 0 <= i1 < self.h and 0 <= j1 < self.w:
                            net_map[k, i, j] = matrix[i * self.w + j, i1 * self.w + j1] + matrix[
                                i1 * self.w + j1, i * self.w + j]
            net_map = (net_map - net_map.mean()) / net_map.std()
            self.state = torch.from_numpy(state).int()
            self.matrix, self.map, self.net_map = torch.from_numpy(matrix).to(self.device), torch.from_numpy(map).to(
                self.device), torch.from_numpy(net_map).to(self.device)
            self.road_aoi = np.load(self.road_aoi_path)

        # reward weight
        self.traj_rate = self.opt.traj_reward_weight if hasattr(options, 'traj_reward_weight') else 0.6
        self.road_rate = self.opt.road_reward_weight if hasattr(options, 'road_reward_weight') else 0.4
        print(f'traj_rate:{self.traj_rate}, road_rate:{self.road_rate}')
        self.rates = [self.traj_rate, self.road_rate]

        # debug
        self.debug_main = not True
        self.debug_loss = True

        self.sleep_time = 1  # sleep time after each episode to check the result
        self.figsize = [16, 10]  # reward figure size
        self.output_dim = 5  # output dim
