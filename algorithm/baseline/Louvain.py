import os
from time import time
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
import community
from louvain import PyLouvain

from utils.util import ws
from utils.read_data import DataHandler

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
    xishu = np.ones([n,n],dtype=np.float32) - np.diag(np.ones(n,dtype=np.float32))/2
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

def Louvain_process(state, matrix):
    state0 = state
    flag = True
    while flag:
        state1 = Louvain_single_process(state0, matrix)
        if (state1 == state0).all():
            flag = False
        else:
            state0 = state1
    return state1

if __name__=='__main__':
    # read matrix
    h, w = 10, 10
    matrix = np.load(f'../../data/synthetic_data/{h}_{w}/matrix.npy')

    # Louvain
    state = (np.arange(h * w) + 1).reshape([h, w])
    ans = Louvain_process(state,matrix)
    np.save(f'Louvain_{h}_{w}.npy', ans.reshape([h, w]).astype(np.int16))