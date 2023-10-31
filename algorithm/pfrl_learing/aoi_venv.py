import gym
import torch
import random
import numpy as np
from gym import spaces
from queue import Queue

from utils.Scaler import BasicScaler
from utils.metrics import cal_road_aoi_similarity


class AOIVirtualEnv():
    def __init__(self, config):
        self.config = config
        h, w = config.h, config.w
        # state space
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=1, high=h * w, shape=(h, w), dtype=np.int8),
            'target': spaces.Box(low=np.array([0, 0]), high=np.array([h, w]), dtype=np.int8),
            'net_map': spaces.Box(low=-50, high=50, shape=(h, w), dtype=np.float32),
            'road_aoi': spaces.Box(low=1, high=h * w, shape=(h, w), dtype=np.int8)
        })
        # action space
        self.action_space = spaces.Discrete(5)
        self.device = config.device
        self.w, self.h = config.w, config.h
        self.dir_h, self.dir_w = config.dir_h, config.dir_w
        self.signal = config.signal
        self.start_with_road = config.start_with_road
        self.play_speed = config.play_speed
        self.traj_rate, self.road_rate = config.traj_rate, config.road_rate
        self.scaler = BasicScaler(config, config.rates)
        self.fixed_eps = config.fixed_eps
        self.border_index, self.border_lst = 0, []
        if self.fixed_eps:
            self.max_erg = config.max_erg
            self.ergodic_num = 0
        else:
            self.over_flag = True
        self.max_erg = config.max_erg
        self.ergodic_num = 0

        self.map, self.matrix = config.map, config.matrix
        self.net_map = config.net_map
        self.road_aoi = torch.from_numpy(config.road_aoi)

    def reset(self):
        """
        reset the environment
        """
        config = self.config
        self.state = config.state.clone().to(self.device)

        # reset border index
        self.border_index = 0
        self.border_lst = self.find_border(self.state)
        if self.fixed_eps:
            self.ergodic_num = 0
        else:
            self.over_flag = True
        return {'state': self.state,
                'target': torch.from_numpy(np.array([self.border_lst[0][0], self.border_lst[0][1]])).to(self.device),
                'net_map': self.net_map,
                'road_aoi': self.road_aoi
                }

    def step(self, act):
        over = False
        i, j = self.border_lst[self.border_index][0], self.border_lst[self.border_index][1]

        # get reward before agent move.
        reward0_1, reward0_2 = self.cal_reward(i, j)

        # change the state
        if 0 <= act < 4:
            i2, j2 = i + self.dir_h[act], j + self.dir_w[act]
            if 0 <= i2 < self.h and 0 <= j2 < self.w:
                self.state[i, j] = self.state[i2, j2]
        next_state = self.state

        # get the reward after agent move.
        reward1_1, reward1_2 = self.cal_reward(i, j)

        # calculate delta_reward
        rewards = [reward1_1 - reward0_1, reward1_2 - reward0_2]
        reward = self.scaler.cal_reward(rewards)

        # check if the game is over.
        if self.fixed_eps:
            over = (self.border_index >= len(self.border_lst) - 1) & (self.ergodic_num >= self.max_erg - 1)
        else:
            over = over
            self.over_flag = self.over_flag & over
            if self.border_index >= len(self.border_lst) - 1:
                over = over and self.over_flag
                self.over_flag = True

        # get the next stae
        self.border_index += 1
        if self.border_index >= len(self.border_lst):
            self.border_lst = self.find_border(self.state)
            if self.fixed_eps:
                self.ergodic_num += 1
            self.border_index = 0

        if len(self.border_lst) == 0 and not self.fixed_eps:
            i_next, j_next = -1, -1
            over = True
        else:
            i_next, j_next = self.border_lst[self.border_index][0], self.border_lst[self.border_index][1]

        obervation = {'state': next_state, 'target': torch.from_numpy(np.array([i_next, j_next])).to(self.device),
                      'net_map': self.net_map, 'road_aoi': self.road_aoi}
        return obervation, reward, over, rewards

    def cal_reward(self, i, j):
        '''
        calculate reward in the current grid
        '''
        reward1 = 0.
        for k in range(len(self.dir_h)):
            i2, j2 = i + self.dir_h[k], j + self.dir_w[k]
            if 0 <= i2 < self.h and 0 <= j2 < self.w:
                if self.state[i, j] != self.state[i2, j2]:
                    reward1 -= self.matrix[i * self.w + j, i2 * self.w + j2]
        if not isinstance(reward1, float):
            reward1 = reward1.detach().cpu()

        reward2 = cal_road_aoi_similarity(self.road_aoi.to(self.device), self.state).float().cpu()
        return [reward1, reward2]

    def check(self, i, j):
        '''
        check if the coordinate is legel
        '''
        if 0 <= i < self.h and 0 <= j < self.w:
            return True
        return False

    def find_border(self, state):
        '''
        find border grids
        '''
        h, w = state.shape
        state_shift = torch.zeros([h, w, 4], device=self.device)
        state_shift[:h - 1, :, 0], state_shift[1:, :, 1] = state[1:, :], state[:h - 1, :]
        state_shift[:, :w - 1, 2], state_shift[:, 1:, 3] = state[:, 1:], state[:, :w - 1]
        u, d, l, r = torch.ne(state, state_shift[:, :, 0]), torch.ne(state, state_shift[:, :, 1]), \
                     torch.ne(state, state_shift[:, :, 2]), torch.ne(state, state_shift[:, :, 3])
        border_lst = torch.nonzero(u | d | l | r)

        border_lst = sorted(border_lst.tolist(), key=lambda x: int(state[x[0], x[1]]))
        return border_lst
