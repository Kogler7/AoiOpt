import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from algorithm.pfrl_learing.aoi_config import Config
from algorithm.pfrl_learing.aoi_agent import AOIAgent
from algorithm.pfrl_learing.aoi_learning import PolicyGradientPFRL


def show(net, path, name):
    conv_weight = net.conv1.weight.detach().numpy()  # 16*6*5*5
    score_weight = net.score_layer.weight  # 6*16
    score_bias = net.score_layer.bias.unsqueeze(1)  # 6
    score = torch.cat([score_weight, score_bias], dim=1).detach().numpy()  # 6*17

    out_channels, in_channels, _, _ = conv_weight.shape
    fig = plt.figure(figsize=[out_channels * 5, in_channels * 5])

    for i in range(in_channels):
        for j in range(out_channels):
            ax = plt.subplot(in_channels, out_channels, i * out_channels + j + 1)
            ax.set_title(f'{j}')
            ax.matshow(conv_weight[j, i, :, :])
    '''fig.show()
    plt.close()'''
    fig.savefig(os.path.join(path, f'conv-{name}.png'))

    fig = plt.figure(figsize=[5, 8])
    plt.matshow(score)
    fig.savefig(os.path.join(path, f'MLP-{name}.png'))

    plt.show()
    plt.close()


if __name__ == '__main__':
    os.chdir('../')
    aoi_learning = PolicyGradientPFRL()
    config = aoi_learning.config
    aoi_learning.agent.load(config.init_param)
    agent = aoi_learning.agent
    show(agent.net, config.save_model, config.model_name)
