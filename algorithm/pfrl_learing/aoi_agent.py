import math
import os
import numpy as np
from collections import deque

import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        hidden_dim = 32
        self.conv1 = nn.Conv2d(7, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.score_layer = nn.Linear(hidden_dim, config.output_dim)

        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=5, stride=1, padding=2)
        self.policy_layer = nn.Linear(config.h * config.w, config.output_dim)

        self.type = config.agent_type
        self.device = config.device

    def forward(self, x):
        # x C*H*W
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.conv2(x))
        x2 = x1.reshape([x1.shape[0], -1])
        ans = self.policy_layer(x2)
        return pfrl.action_value.DiscreteActionValue(ans)

class AOIAgent:
    def __init__(self, env, config):
        self.config = config
        self.net = MLP(config).to(config.device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=config.lr)

        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.9, end_epsilon=0.1,
                                                           decay_steps=5 * 10 ** 4,
                                                           random_action_func=env.action_space.sample)
        update_interval = 10**4 if config.env_type=='real' else 1000
        self.agent = pfrl.agents.DoubleDQN(self.net, self.optimizer,
                                           replay_buffer, config.gamma, explorer,
                                           phi=self.feature_extract,
                                           gpu=config.device.index,
                                           replay_start_size=update_interval,
                                           target_update_interval=update_interval)

        self.dir_h, self.dir_w = [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0]
        self.h, self.w = config.h, config.w
        self.device = config.device
        self.input_replay_buffer = deque(maxlen=10 ** 4)  # normalize the input of state
        self.input_mean, self.input_std = 0, 1
        self.training = True

    def feature_extract(self, obs):
        # MLP input: AOI, target grid, trajectory, road segmentation
        state, i, j, net_map, road_aoi = obs['state'], obs['target'][0], obs['target'][1], obs['net_map'], obs['road_aoi']
        target = torch.zeros([self.h, self.w], device=self.device)  # C*H*W
        target[i, j] = 1
        if self.config.input_norm:
            state = ((state - self.input_mean) / self.input_std)
        x = torch.cat([target.unsqueeze(0), state.unsqueeze(0), road_aoi.unsqueeze(0).to(self.device), net_map.to(self.device)], dim=0)
        return x.float()

    def choose_action(self, obs):
        # state -> action
        state, i, j = obs['state'], obs['target'][0], obs['target'][1]
        act = self.agent.act(obs)
        if self.config.input_norm:
            self.input_replay_buffer.append(state.float())
        return int(act)

    def observe(self, obs, reward, over, reset=False):
        self.agent.observe(obs, reward, over, reset)

    def save(self, mean_std, path):
        torch.save(self.net.state_dict(), path)
        mean_std_file = path.replace('.pth', '.npy')
        np.save(mean_std_file, mean_std)

    def load(self, path):
        net_weights = torch.load(path, map_location=self.device)
        self.net.load_state_dict(net_weights)

        mean_std_file = path.replace('.pth', '.npy')
        if not os.path.exists(mean_std_file):
            self.agent.input_mean, self.agent.input_std = 0, 1
        else:
            mean_std = np.load(mean_std_file)
            mean, std = mean_std[0, :, :], mean_std[1, :, :]
            self.agent.input_mean, self.agent.input_std = torch.from_numpy(mean).float(), \
                                                          torch.from_numpy(std).float()
        print('load {}'.format(path))
