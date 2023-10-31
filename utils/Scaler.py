import numpy as np
import torch
from collections import deque

class Scaler:
    def __init__(self):
        pass
    def update(self, xs):
        pass
    def cal_reward(self, xs: list, ks: list):
        pass
    def retransform(self, real_xs: np.array, ks: list):
        pass


class MinMaxScaler(Scaler):
    def __init__(self, ratios, mins=None, maxs=None):
        super(MinMaxScaler, self).__init__()
        if maxs is None:
            maxs = [1, 1]
        if mins is None:
            mins = [0, 0]

        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.rates = [0.6, 0.4]

    def update(self, rewardses):
        rewardses = np.array(rewardses)
        self.mins = 0.95 * self.mins + 0.05 * np.min(rewardses, axis=0)
        self.maxs = 0.95 * self.maxs + 0.05 * np.max(rewardses, axis=0)

    def cal_reward(self, rewards: list):
        ans = 0
        for i in range(len(rewards)):
            if self.maxs[i] > self.mins[i]:
                ans += self.rates[i] * (rewards[i] - self.mins[i]) / (self.maxs[i] - self.mins[i])
            else:
                continue
        return ans

    def retransform(self, each_rewards, ks: list):
        each_rewards = np.array(each_rewards)
        if each_rewards.ndim == 2:
            ans = np.zeros([each_rewards.shape[0]])
            for i in range(len(ks)):
                ans[:] += each_rewards[:, i] * ks[i]
        elif each_rewards.ndim == 1:
            ans = 0
            for i in range(len(ks)):
                ans += each_rewards[i] * ks[i]
        return ans


class BasicScaler(Scaler):
    def __init__(self, config, rates=None):
        super(BasicScaler, self).__init__()
        if rates is None:
            rates = [0.6,0.4]
        self.rates = rates
        self.h,self.w = config.h, config.w
        self.dir_h,self.dir_w = config.dir_h,config.dir_w
        self.mins = np.zeros(2)
        self.maxs = np.ones_like(self.mins)

    def update(self):
        pass

    def cal_reward(self, rewards: list):
        ans = 0
        for i in range(len(rewards)):
            if self.maxs[i] > self.mins[i]:
                ans += self.rates[i] * (rewards[i] - self.mins[i]) / (self.maxs[i] - self.mins[i])

        return ans

    def retransform(self, each_rewards, ks: list):
        each_rewards = np.array(each_rewards)
        if each_rewards.ndim == 2:
            ans = np.zeros([each_rewards.shape[0]])
            ans[:] += each_rewards[:,0] *ks[0]
            for i in range(1,len(ks)):
                if self.mins[i] < self.maxs[i]:
                    ans[:] += each_rewards[:, i] * ks[i]
        elif each_rewards.ndim == 1:
            ans = 0
            ans += each_rewards[0] * self.matrix_max * ks[0]
            for i in range(1,len(ks)):
                if self.mins[i] < self.maxs[i]:
                    ans += each_rewards[i] * ks[i]
        return ans


class Distribution:
    def __init__(self, min_value, max_value, num):
        self.min_value, self.max_value = min_value, max_value
        self.num = num
        self.step = (self.max_value - self.min_value) / (self.num + 1)
        # self.distribution = np.zeros(2+self.num+1)
        self.gaps = []
        temp = self.min_value
        while temp <= self.max_value + 0.01:
            self.gaps.append(temp)
            temp += self.step
        self.gaps.append(np.inf)
        self.gaps = torch.tensor(np.array(self.gaps).reshape([1, -1]))
        self.F = torch.zeros(3 + self.num + 1)

    def cumulate_ratio(self, xs):
        vert = ((xs - self.min_value) / self.step).floor().long() + 1
        vert = torch.max(input=vert, other=torch.zeros_like(vert))
        vert = torch.min(input=vert, other=torch.zeros_like(vert) + 2 + self.num)
        if self.F[-1]:
            ans = self.F.index_select(0, vert + 1) / self.F[-1]
        else:
            ans = torch.ones_like(xs)
        return ans

    def update(self, xs):
        gaps = self.gaps.repeat(xs.shape[0], 1)
        xs1 = np.repeat(xs.reshape([-1, 1]), self.gaps.shape[1], axis=1)
        verts = gaps > xs1
        self.F[1:] += verts.sum(dim=0)
