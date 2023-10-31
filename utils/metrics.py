import os
from queue import Queue
import torch
import numpy as np
from sklearn import metrics


def cal_CR(y_true, y_pred):
    # CR
    y_true = torch.tensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred = torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    assert y_true.size() == y_pred.size(), "y_true and y_pred must have the same length."

    h, w = y_true.shape[:2]

    pred_flat, true_flat = y_pred.flatten(), y_true.flatten()
    pred_1 = torch.repeat_interleave(pred_flat.unsqueeze(0), repeats=h * w, dim=0)
    pred_cmm = torch.eq(pred_1, pred_1.t())  # check grid pairs in pred.
    true_1 = torch.repeat_interleave(true_flat.unsqueeze(0), repeats=h * w, dim=0)
    true_cmm = torch.eq(true_1, true_1.t())  # check grid pairs in true.
    pred_cmm_sum = (pred_cmm & true_cmm).sum()
    true_cmm_sum = true_cmm.sum()

    cmm = pred_cmm_sum / true_cmm_sum
    return cmm.cpu()

def cal_FMI(y_true,y_pred):
    # Fowlkes-Mallows Score
    true_label, pred_label = y_true.reshape([-1]), y_pred.reshape([-1])
    return metrics.fowlkes_mallows_score(true_label,pred_label)

def cal_state_reward(state, road_aoi, config):
    # calculate the state reward
    h, w = state.shape

    reward1, reward2 = 0, 0
    dir_h, dir_w = [-1, 1, 0, 0], [0, 0, -1, 1]

    map, matrix = config.map,config.matrix

    if hasattr(config, 'map'):
        d,r = torch.zeros_like(state),torch.zeros_like(state)
        d1, r1 = torch.ne(state[:-1, :], state[1:, :]), torch.ne(state[:, :-1], state[:, 1:])
        # d:h-1*w  r:h*w-1
        d[:-1,:],r[:,:-1] = d1,r1
        not_equal = torch.stack([d, r], dim=-1)
        reward1 -= (map * not_equal).sum().detach().cpu()
    else:
        for i in range(h):
            for j in range(w):
                for k in range(len(dir_h)):
                    i2, j2 = i + dir_h[k], j + dir_w[k]
                    if 0 <= i2 < h and 0 <= j2 < w and state[i, j] != state[i2, j2]:
                        reward1 -= matrix[i * w + j, i2 * w + j2]

    reward2 = cal_road_aoi_similarity(road_aoi.to(state.device), state).cpu()

    state_reward = [reward1, reward2]
    return state_reward


def cal_road_aoi_similarity(true, pred):
    return cal_CR(true, pred)
