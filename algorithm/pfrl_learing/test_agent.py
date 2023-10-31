from random import random,randint
from collections import deque


class RandomAgent:
    def __init__(self,config):
        self.config = config
        self.dir_h, self.dir_w = [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0]
        self.h, self.w = config.h, config.w
    def choose_action(self,obs):
        act = randint(0,self.config.output_dim-1)
        return act
    def observe(self, obs, reward, over, reset=False):
        pass
    def save(self,path):
        pass
    def load(self,path):
        pass

class GreedyAgent:
    def __init__(self,config):
        self.config = config

        self.dir_h, self.dir_w = [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0]
        self.h, self.w = config.h, config.w
    def choose_action(self,obs):
        i, j,net_map = obs['target'][0], obs['target'][1], obs['net_map']
        return net_map[:,i,j].argmax()
    def observe(self, obs, reward, over, reset=False):
        pass
    def save(self,path):
        pass
    def load(self,path):
        pass