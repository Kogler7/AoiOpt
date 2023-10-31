from utils.util import ws
from utils.metrics import cal_CR,cal_FMI
from utils.metrics import cal_state_reward

import pandas as pd
import os
import torch
import numpy as np

def cal_reward(pred, matrix, road_aoi, rates=[0.6,0.4]):
    class Config:
        def __init__(self):
            pass

    config = Config()
    config.env_type = 'synthetic'
    config.matrix = matrix
    pred = torch.from_numpy(pred)
    road_aoi = torch.from_numpy(road_aoi)
    state_reward = cal_state_reward(pred, road_aoi, config)
    reward = 0.
    for rates,sr in zip(rates,state_reward):
        reward += rates*sr
    return reward,state_reward

def metric_test(h,w,pred,matrix,road_aoi,config):
    reward, state_rewards = cal_reward(pred, matrix, road_aoi)

    ans = {'reward':reward,'traj reward':state_rewards[0],'road reward':state_rewards[4]}

    if config.env_type=='synthetic':
        true = pd.read_csv(f'data/synthetic_data/{h}_{w}/aoi.csv', header=None)
        true = true.values
        CR = cal_CR(true,pred)
        FMI = cal_FMI(true,pred)
        ans['CR'], ans['FMI'] = CR,FMI
    return ans

if __name__ == '__main__':
    h, w =  10,10 # 6,6 5,5
    true = pd.read_csv(f'../data/synthetic_data/{h}_{w}/aoi.csv', header=None)
    true = true.values
    pred_lst = []

    # Baseline
    # pred = np.load(f'../result/baseline/RoadNetwork_{h}_{w}.npy') # Road
    # pred_lst.append(('RoadNetwork',pred))
    # pred = np.load(f'../result/baseline/Louvain_{h}_{w}.npy')  # Louvain
    # pred_lst.append(('Louvain',pred))
    # pred = np.load(f'../result/baseline/DBSCAN_{h}_{w}.npy') # DBSCAN
    # pred_lst.append(('DBSCAN',pred))
    # pred = np.load(f'../result/baseline/CKMeans_{h}_{w}.npy') # CKMeans
    # pred_lst.append(('CKMeans',pred))
    # pred = np.load(f'../result/baseline/GCLP_{h}_{w}.npy') # GCLP
    # pred_lst.append(('GCLP',pred))
    # pred = np.load(f'../result/baseline/GreedySeg_{h}_{w}.npy') # GCLP
    # pred_lst.append(('GreedySeg',pred))
    # pred = np.load(f'') #_RL Ours!
    # pred_lst.append(('TrajRL4AOI',pred))

    # traversal number
    # for max_erg in range(1,7):
    #     pred_name = f'TrajRL4AOI_m{max_erg}_Env.npy'
    #     pred = np.load(os.path.join(ws,'result/traversal_number',pred_name))
    #     pred_lst.append((pred_name,pred))

    # post process
    # rlfile = f'TrajRL4AOI_RL_Env.npy'
    # pred = np.load(os.path.join(ws, 'result/post_process', rlfile))
    # pred_lst.append((rlfile, pred))
    # file = f'TrajRL4AOI _Env.npy'
    # pred = np.load(os.path.join(ws, 'result/post_process', file))
    # pred_lst.append((file, pred))


    # reward ablation
    # abnation_t_name = f'10_10_m10_t500_t0.6_r0.0_Env0.npy'
    # pred = np.load(os.path.join(ws,'output/paper/ablation_reward',abnation_t_name))
    # pred_lst.append((abnation_t_name,pred))
    # abnation_r_name = f'10_10_m10_t500_t0.0_r0.4_Env0.npy'
    # pred = np.load(os.path.join(ws, 'output/paper/ablation_reward', abnation_r_name))
    # pred_lst.append((abnation_r_name, pred))

    # ablation_t_name = f'k1=0.6_k2=0.0_RL_Env.npy'
    # pred = np.load(os.path.join(ws,'result/reward_ablation',ablation_t_name))
    # pred_lst.append((ablation_t_name,pred))
    # ablation_r_name = f'k1=0.0_k2=0.4_RL_Env.npy'
    # pred = np.load(os.path.join(ws, 'result/reward_ablation', ablation_r_name))
    # pred_lst.append((ablation_r_name, pred))

    matrix = np.load(f'../data/manu_data/{h}_{w}/matrix.npy')
    matrix = matrix / matrix.max()
    road_aoi = pd.read_csv(f'../data/synthetic_data/{h}_{w}/road_aoi.csv',header=None).values

    for name,pred in pred_lst:
        print(f'method:{name}:',end=' ')
        reward,state_rewards = cal_reward(pred, matrix, road_aoi)
        CR = cal_CR(true, pred)
        FMI = cal_FMI(true,pred)

        print(f'reward:{reward}\n state rewards:{state_rewards}\n FMI:{FMI}\n CR:{CR}\n')
