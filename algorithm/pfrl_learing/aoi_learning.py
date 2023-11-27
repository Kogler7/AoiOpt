import os
import time
import pandas as pd
import torch
import torch.nn
import numpy as np
from tqdm import tqdm
from itertools import count
from pfrl.utils.random_seed import set_random_seed
import community
import networkx as nx

from .aoi_venv import AOIVirtualEnv
from .aoi_agent import AOIAgent
from .test_agent import GreedyAgent
from utils.plot_fig import plot_rewards
from utils.metrics import cal_FMI
from utils.metrics import cal_CR
from utils.metrics import cal_state_reward
from utils.ans_performance import metric_test

try:
    from visual_plat.proxies.update_proxy import UpdateProxy
except Exception:
    class UpdateProxy:
        @staticmethod
        def batched_reload(tags, data):
            pass


def save_datas(datas: list, names: list, file):
    data_dict = {}
    for data, name in zip(datas, names):
        data_dict[name] = np.array(data)
    df = pd.DataFrame(data_dict)
    df.to_csv(file)


class PolicyGradientPFRL:
    def __init__(self, config):
        self.config = config
        self.config.multi = False # multi main or not
        # set random seed
        set_random_seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.device == 'cuda':
            torch.cuda.manual_seed(self.config.seed)

        # self.signal = self.config.signal
        self.env = AOIVirtualEnv(self.config)
        self.agent = AOIAgent(self.env, self.config)
        self.config.agent_type=type(self.agent)
        # self.agent = GreedyAgent(self.config)
        self.config.w = self.env.w
        self.config.h = self.env.h

        if not os.path.exists(self.config.save_model):
            os.makedirs(self.config.save_model)

    def execute(self, signal=None):
        """
        entry point
        """
        # self.signal = signal
        # self.env.signal = signal
        print('using ', self.config.device)

        if self.config.load_model:
            self.agent.load(self.config.init_param)
            print('load param: ',  self.config.init_param)

        if self.config.train:
            self.train()

        if self.config.test:
            self.test()

    def train(self):
        env = self.env
        env.training = True
        agent = self.agent
        agent.training = True
        config = self.config
        config.env = env

        lam = lambda f: 1 - f / config.tra_eps
        scheduler = torch.optim.lr_scheduler.LambdaLR(agent.optimizer, lr_lambda=lam)

        # log file
        file_path = os.path.join(config.log_path, 'log.txt')
        file = open(file_path, mode='w')
        message = 'episode   |  state reward  |  ma_reward  |  similarity: FMI   CR'
        file.write(message)
        file.flush()

        if config.debug_loss:
            print('episode    | reward  sum  mean  best | real_reward  sum  mean | time')
        state_pool, action_pool, prob_pool, reward_pool, over_pool, loss_pool = [], [], [], [], [], []
        rewards, ma_rewards = [], []
        state_rewards, ma_state_rewards, FMIs, CRs = [], [], [], []
        avg_qs, avg_losses = [], []
        best_q, best_episode,best_ep_reward = 0, 0,-10000

        sleep_time = config.sleep_time if not config.debug_main else 0
        env.play_speed = config.play_speed if not config.debug_main else 0

        # begin training
        total_eps = config.tra_eps+1
        for episode in tqdm(range(1, total_eps)): # the agent will be trained for total_eps episodes.
            pid = os.getpid()
            print(f"[{pid}] episode {episode}/{config.tra_eps};")
            start_time = time.time()
            obs = env.reset()
            if config.input_norm and episode > 1:  # input normalization
                input_buffer = torch.stack(list(agent.input_replay_buffer), dim=0)
                agent.input_mean, agent.input_std = input_buffer.mean(dim=0), input_buffer.std(dim=0) + 1e-5
            ep_reward, best_reward, mean_reward = 0, -10000, []

            for iter in count(1, 1): # The agent tack actions util this episode over.
                action = agent.choose_action(obs)
                i, j = obs['target'][0], obs['target'][1]
                obs, reward, over, rewards = env.step(action)  # action
                agent.observe(obs, reward, over, reset=False)  # store the state

                # communicate with the platform
                if not self.config.debug_main:
                    act = action
                    aoi_data = env.state.cpu().numpy()
                    info_data = (
                        (int(j), int(i), act),
                        (reward, rewards[0], rewards[1]),
                        (0, 0)
                    )
                    UpdateProxy.batched_reload(["aoi", "tip"], [aoi_data, info_data])
                    time.sleep(self.config.play_speed)

                reward_pool.append(reward)
                ep_reward += reward
                mean_reward.append(reward)
                best_reward = max(best_reward, reward)

                if over:
                    break

            scheduler.step()  # update the learning rate

            # draw moving avearage reward
            mean_reward = np.mean(mean_reward)
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            # state reward
            state_every_reward = cal_state_reward(env.state, env.road_aoi, config)
            state_reward = 0.
            for ratio, r in zip(config.rates, state_every_reward):
                state_reward += ratio * r
            state_reward = state_reward.cpu()
            state_rewards.append(state_reward)
            if ma_state_rewards:
                ma_state_reward = 0.9 * ma_state_rewards[-1] + 0.1 * state_reward
            else:
                ma_state_reward = state_reward
            ma_state_rewards.append(ma_state_reward)
            # calculate the metrics
            if config.env_type == 'synthetic':
                CR = cal_CR(config.grid.to(config.device),env.state)
                FMI = cal_FMI(config.grid.to(config.device),env.state)
            else:
                CR, FMI = 0, 0  # there is no true segmentation in real-world dataset.
            CRs.append(CR), FMIs.append(FMI)

            message = f'\n{episode}  |  {state_reward:<7.2f}  |  {ma_state_reward:<7.2f}  |  {FMI:<1.4f}  {CR:<1.4f}  '
            file.write(message)
            file.flush()

            msg = agent.agent.get_statistics()
            avg_q, avg_loss = msg[0][1], msg[1][1]
            print('-----avg_q : {:.3f} | avg_loss : {:.3f} | ep_reward:{:.2f} | ma_ep_reward:{:.2f}-----'.format(avg_q, avg_loss,ep_reward,ma_rewards[-1]))

            msg = agent.agent.get_statistics()
            avg_q, avg_loss = msg[0][1], msg[1][1]
            avg_qs.append(avg_q), avg_losses.append(avg_loss)
            if ep_reward > best_ep_reward:
                best_ep_reward, best_episode = ep_reward, episode
                print('[Save] best sum reward now ', ep_reward)
                if best_episode > 1:
                    agent.save(config.save_param)
            '''if ep_reward > best_ep_reward:
                print('best sum reward now ',ep_reward)
                best_ep_reward = ep_reward
                best_ep_reward_model = self.agent.net.state_dict()'''

            end_time = time.time()

            if episode % 10 == 0:
                print(f'{episode} | {state_reward:<7.2f} | {ma_state_reward:<7.2f} | {FMI:<1.4f} {CR:<1.4f}')
                print('{}\t{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}s'.format(episode, ep_reward, mean_reward,best_reward, end_time - start_time))

            time.sleep(sleep_time)

        print("training over!")
        file.close()
        if self.config.reward_fig:
            plot_datas = [rewards, ma_rewards, avg_qs, avg_losses]
            plot_names = ['reward', 'ma_reward', 'avg_q', 'avg_loss']
            plot_rewards(plot_datas, plot_names, config.fig_path)
            save_datas(plot_datas, plot_names, config.fig_path.replace('.png', f'_s{config.seed}.csv'))

            performence_fig_path = os.path.join(config.log_path, config.fig_name + '_performence.png')
            plot_performance_datas = [state_rewards, ma_state_rewards, FMIs, CRs]
            plot_performance_names = ['state reward', 'moving average state reward', 'FMI', 'CR']
            plot_rewards(plot_performance_datas, plot_performance_names, performence_fig_path)
            save_datas(plot_performance_datas, plot_performance_names,performence_fig_path.replace('.png', f'_s{config.seed}.csv'))

    def test(self):
        print('testing...')
        env = self.env
        agent = self.agent
        agent.training = False
        config = self.config

        sleep_time = config.sleep_time
        if not self.config.multi:
            print('iter    | reward  ep_reward')
        with agent.agent.eval_mode():
            env.play_speed = config.play_speed
            env.training = False

            obs = env.reset()
            ep_reward, best_reward = 0, -10000
            for iter in count(1, 1):
                action = agent.choose_action(obs)
                i, j = obs['target'][0], obs['target'][1]

                obs, reward, over, rewards = env.step(action)
                # agent.observe(obs, reward, over, reset=False)
                aoi_data = env.state.cpu().numpy()

                if not config.debug_main:
                    act = action
                    info_data = (
                        (int(j), int(i), act),
                        (reward, rewards[0], rewards[1]),
                        (0, 0)
                    )
                    UpdateProxy.batched_reload(
                        ["aoi", "tip"], [aoi_data, info_data])
                    time.sleep(self.config.play_speed)

                ep_reward += reward

                if not config.multi:
                    print('[{}]: {}\t{:.1f}\t{:.1f}'.format(os.getpid(), iter, reward, ep_reward))

                if over:
                    rl_aoi = obs['state'].detach().cpu().numpy()

                    aoi_data = Louvain_process(rl_aoi, env.matrix, config)

                    if not config.debug_main:
                        info_data = (
                            (int(env.h + 1), int(env.w + 1), 4),
                            (0., 0., 0., 0.),
                            (0, 0)
                        )
                        UpdateProxy.batched_reload(["aoi", "tip"], [aoi_data, info_data])
                    break

            if not config.debug_main:
                time.sleep(sleep_time)

            print(f'[{os.getpid()}] Env test is over.')

            if isinstance(agent, GreedyAgent):
                ans_path = config.fig_path.replace('.png', f'_Greedy_Env.npy')
            else:
                ans_path = config.fig_path.replace('.png', f'_Env.npy')
            np.save(ans_path, aoi_data)

            if not config.debug_main:
                time.sleep(sleep_time)

            metrics = metric_test(config.h, config.w, aoi_data, config.matrix, config.road_aoi,config)
            print(f'[{os.getpid()}] read param:{config.init_param} \n metrics:{metrics}')

            test_ans = aoi_data
        return test_ans

def Louvain_single_process(state, matrix):
    h, w = state.shape[:2]
    nodes = np.unique(state.reshape([-1])).tolist()
    n = len(nodes)
    node_dict = dict(zip(nodes, np.arange(n)))
    nodes_edges = np.zeros([n, n], dtype=np.float32)
    edges = []
    for i in range(h * w):
        ih, iw = i // w, i % w
        for j in range(i, h * w):
            if matrix[i, j]:
                jh, jw = j // w, j % w
                temp1, temp2 = node_dict[state[ih, iw]], node_dict[state[jh, jw]]
                nodes_edges[temp1, temp2] += matrix[i, j]
    nodes_edges = nodes_edges + nodes_edges.T
    xishu = np.ones([n, n], dtype=np.float32) - np.diag(np.ones(n, dtype=np.float32)) / 2
    nodes_edges *= xishu
    nodes_edges = nodes_edges.astype(np.int32)

    for i, n1 in enumerate(nodes):
        for j in range(i, n):
            n2 = nodes[j]
            edges.append([n1, n2, nodes_edges[i, j]])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    partition = community.best_partition(G, resolution=0.7, weight='weight')
    keys = partition.keys()
    values = partition.values()
    print(len(keys), len(values))
    print(len(set(keys)), len(set(values)))

    grid = np.zeros([h, w], dtype=np.int32)
    for i in range(h):
        for j in range(w):
            grid[i, j] = partition[state[i, j]]

    return grid


def Louvain_process(state, matrix,config):
    if config.agent_type == 'AOIAgent':
        if config.env_type == 'synthetic':
            state1 = Louvain_single_process(state, matrix)
        else:
            state0 = state
            flag = True
            while flag:
                state1 = Louvain_single_process(state0, matrix)
                if (state1==state0).all():
                    flag = False
                else:
                    state0 = state1
    return state1