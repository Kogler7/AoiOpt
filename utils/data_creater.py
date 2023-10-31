import imp
import os
from sys import implementation
import torch
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from turtle import distance
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import utils.data_loader as dl


class GetGrid:
    def __init__(self, file):
        # file should be .csv
        if file[-3:] == 'csv':
            self.grid = pd.read_csv(file, names=None, header=None).to_numpy()  #
        elif file[-3:] == 'npy':
            self.grid = np.load(file)
        else:
            assert file[-3:] == "csv" or file[-3:] == 'npy', "wrong input format, aoi file shoud be .csv or .npy"
        self.h, self.w = self.grid.shape[-2], self.grid.shape[-1]

    def save(self, file):
        np.save(os.path.splitext(os.path.abspath(file))[0] + '.npy', self.grid)

    def get_dir(self):  # dir (h*w)*(h*w)
        try:
            return self.dir
        except:
            self.dijkstra()
            return self.dir

    def get_dis(self):  # dis (h*w)*(h*w)
        try:
            return self.dis
        except:
            self.dijkstra()
            return self.dis

    def dijkstra(self):
        from queue import PriorityQueue as queue
        dis = np.zeros([self.h, self.w, self.h, self.w], dtype='int8') + np.inf
        dir = np.zeros([self.h, self.w, self.h, self.w], dtype='int8')

        dhs, dws = [0, -1, 1, 0, 0], [0, 0, 0, -1, 1]  # direction array
        dir_change = [0, 2, 1, 4, 3]
        for ih in range(self.h):
            for iw in range(self.w):
                dis[ih, iw, ih, iw] = 0
                vis = np.zeros([self.h, self.w], dtype='int8')

                q = queue()
                q.put((0, ih, iw))
                num = 1
                while num < self.h * self.w and not q.empty():
                    _, last_h, last_w = q.get()
                    if vis[last_h, last_w]:
                        continue
                    num += 1
                    for k in range(1, 5):
                        dh, dw = dhs[k], dws[k]
                        now_h = last_h + dh
                        now_w = last_w + dw
                        if now_h >= 0 and now_h < self.h and now_w >= 0 and now_w < self.w:
                            l = 1 if self.grid[last_h, last_w] == self.grid[now_h, now_w] else 1000
                            if not vis[now_h, now_w] and dis[ih, iw, now_h, now_w] > dis[ih, iw, last_h, last_w] + l:
                                dis[ih, iw, now_h, now_w] = dis[ih, iw, last_h, last_w] + l
                                dir[ih, iw, now_h, now_w] = dir_change[k]
                                q.put((dis[ih, iw, now_h, now_w], now_h, now_w))
                    vis[last_h, last_w] = 1
        self.dis = dis
        self.dir = dir


class GetParcels():
    def __init__(self, grid):
        self.grid = grid.grid
        self.h, self.w = grid.h, grid.w

    def get_from_file(self, file):
        self.parcel_lst = dl.load(os.path.abspath(file))

    def generate(self, couriers=None, AOI_num=2, parcel_num=4, times=1):
        self.AOI_num = AOI_num
        self.parcel_num = parcel_num

        if couriers == None:
            self.couriers = []
            couriers_1 = [[i] for i in range(1, self.AOI_num + 1)]
            self.couriers += couriers_1

        self.courier_grid_lst = []
        for courier in self.couriers:
            ans_lst = []
            aoi_lst = []
            for i in courier:
                ans_lst.append(np.argwhere(self.grid == i))
                aoi_lst.append(i)
            ans_lst = np.concatenate(ans_lst, axis=0)
            self.courier_grid_lst.append(ans_lst)

        self.border_dis = self.cal_border_dis([]) # calculate distance to border

        self.parcel_lst = self.init_pacels(times=times) # generate parcels

    def get_xy(self, begin: list, step: list):
        parcel_lst = np.zeros_like(self.parcel_lst)
        k, n, _ = self.parcel_lst.shape
        parcel_lst[:, :, 0] = step[0] * self.parcel_lst[:, :, 0] + begin[0] + np.random.normal(loc=0, scale=step[0] / 2,
                                                                                               size=[k, n])
        parcel_lst[:, :, 1] = step[1] * self.parcel_lst[:, :, 1] + begin[1] + np.random.normal(loc=0, scale=step[1] / 2,
                                                                                               size=[k, n])
        return parcel_lst

    def save(self, file):
        try:
            np.save(os.path.splitext(file)[0] + '.npy', self.parcel_lst)
        except NameError:
            print('parcel_lst was saved before definition.')

    def init_pacels(self, times):
        parcel_lst = []
        for t in range(times):
            for i in range(len(self.courier_grid_lst)):
                courier_grid_xy = self.courier_grid_lst[i]
                # Courier has a higher probility in the border grids
                aois = []
                if i < len(self.couriers):
                    aois = self.couriers[i]
                    border_dis = self.border_dis
                else:
                    border_dis = self.cal_border_dis(aois=aois)
                parcels = self.init_parcel(num=self.parcel_num, courier_grid_xy=courier_grid_xy,
                                           border_dis=border_dis)

                parcel_lst.append(np.expand_dims(parcels, axis=0))
        parcel_lst = np.concatenate(parcel_lst, axis=0)
        print('init parcel over! the shape is ', parcel_lst.shape)
        return parcel_lst

    def init_parcel(self, num, courier_grid_xy, border_dis):
        assert (courier_grid_xy.size != 0)
        courier_grid = courier_grid_xy[:, 0] * self.w + courier_grid_xy[:, 1]
        courier_grid_p = border_dis[courier_grid_xy[:, 0], courier_grid_xy[:, 1]]
        courier_grid_p = np.max(courier_grid_p) - courier_grid_p + 1
        courier_grid_p = courier_grid_p / np.sum(courier_grid_p)
        parcels = np.random.choice(courier_grid, size=num, replace=False, p=courier_grid_p)

        parcels_xy = np.zeros([parcels.shape[0], 2])
        parcels_xy[:, 0] = parcels // self.w
        parcels_xy[:, 1] = parcels % self.w
        return parcels_xy.astype(int)

    def cal_border_dis(self, aois):
        border_dis = np.zeros_like(self.grid)
        for i in range(self.h):
            for j in range(self.w):
                border_dis[i, j] = self.find_border_dis(border_dis, i, j, aois)
        return border_dis

    def find_border_dis(self, border_dis, i, j, aois):
        dirs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        l = 1
        while not border_dis[i, j] and l < max(self.h, self.w):
            for dir in dirs:
                if i + l * dir[0] >= 0 and i + l * dir[0] < self.h and j + l * dir[1] >= 0 and j + l * dir[1] < self.w:
                    if aois:
                        if self.grid[i + l * dir[0], j + l * dir[1]] not in aois:
                            return l
                    else:
                        if self.grid[i, j] != self.grid[i + l * dir[0], j + l * dir[1]]:
                            return l
            l += 1
        return l


def check_parcels(env, parcel_lst, AOIs):
    # k*n*2
    for i in range(parcel_lst.shape[0]):
        AOI_nums = env.grid[parcel_lst[i, :, 0], parcel_lst[i, :, 1]]
        d = Counter(AOI_nums)
        d_s = list(d.keys())
        print(d_s, AOIs[i])
        assert set(d_s) <= set(AOIs[i])


class GetTrajectory():
    def __init__(self):
        pass

    def get_from_file(self, file):
        try:
            del self.h, self.w, self.route, self.center_w, self.center_h, self.n, self.__t, self.direction, self.distance
        except:
            pass
        finally:
            trajectory = dl.load(os.path.abspath(file))
            if len(trajectory[-1]) == 0:
                trajectory.pop()
            self.trajectory = trajectory
        return self.trajectory

    def generate(self, h, w, parcels, dir, dis, plot=None):
        parcels = parcels.parcel_lst
        parcel_num = parcels.shape[1]  # k*n*2
        self.h, self.w = h, w
        self.center_h, self.center_w = h // 2, w // 2
        self.direction = dir
        self.distance = dis
        self.__t = 0
        ans = []

        parcels = np.insert(parcels, 0, values=(self.center_h, self.center_w), axis=1)  # insert a column (center_h,center_w), because VRP ask for a closed trajectory.
        for j in range(len(parcels)):
            location = parcels[j, :, :]
            self.location = location
            self.n = len(location)
            manager = pywrapcp.RoutingIndexManager(self.n, 1, 0)
            routing = pywrapcp.RoutingModel(manager)
            self.__t = 0

            transit_callback_index = routing.RegisterTransitCallback(
                self.distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(
                transit_callback_index)
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.FromMilliseconds(100)
            search_parameters.log_search = False

            solution = routing.SolveWithParameters(search_parameters)
            index = routing.Start(0)
            self.route = []
            self.route.append(location[0].tolist())
            while not routing.IsEnd(index):
                old = index
                index = solution.Value(routing.NextVar(index))
                new = index if index != parcel_num + 1 else 0
                self.toto(location, old, new)
            parcel_lst = location[1:, :].tolist()
            self.clear_head_tail(location, parcel_lst)
            self.route = np.array(self.route, dtype='int8')
            ans.append(self.route)

            if plot:
                if (j % 100 == 0):
                    plt.figure(figsize=(8, 6))
                    depot_coor = self.route[0]
                    plt.plot(depot_coor[0], depot_coor[1], 'r*', markersize=11)
                    for i in range(len(self.route) - 1):
                        start_coor = self.route[i]
                        end_coor = self.route[i + 1]
                        plt.arrow(start_coor[0], start_coor[1], end_coor[0] -
                                  start_coor[0], end_coor[1] - start_coor[1])
                    plt.xlabel("X coordinate", fontsize=14)
                    plt.ylabel("Y coordinate", fontsize=14)
                    plt.title("TSP path for " + str(j), fontsize=16)
                    plt.savefig(os.path.abspath(os.path.join(
                        plot, "TSP path for " + str(j) + ".png")))

        self.trajectory = ans
        return self.trajectory

    def save(self, file):
        try:
            np.save(os.path.splitext(file)[0] + '.npy', self.trajectory)
        except NameError:
            print('trajectroy was saved before definition.')

    def toto(self, location, old, new):
        xo, yo = location[old]
        xn, yn = location[new]
        while (xo != xn or yo != yn):
            tmp = self.direction[self.w * xn + yn][self.w * xo + yo]
            self.__t += 1
            if (tmp == 1):
                xo -= 1
            elif (tmp == 2):
                xo += 1
            elif (tmp == 3):
                yo -= 1
            elif (tmp == 4):
                yo += 1
            self.route.append([xo, yo])
        return self.route

    def clear_head_tail(self, location, parcel_lst):
        # drop some point because they ara specified before calculation.
        if (self.route[0] == [self.center_h, self.center_w]) and ([self.center_h, self.center_w] not in parcel_lst):
            while (self.route[0] not in parcel_lst):
                self.route.pop(0)
        if (self.route[-1] == [self.center_h, self.center_w]) and ([self.center_h, self.center_w] not in parcel_lst):
            while (self.route[-1] not in parcel_lst):
                self.route.pop()

    def distance_callback(self, from_index, to_index):
        from_index %= self.n
        to_index %= self.n
        x1, y1 = self.location[from_index]
        x2, y2 = self.location[to_index]
        return int(self.distance[self.w * x1 + y1][self.w * x2 + y2])


class GetMatrixAndMap():
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.dir_h, self.dir_w = [-1, 1, 0, 0], [0, 0, -1, 1]

    def get_matrix(self):
        return torch.from_numpy(self.matrix)

    def get_map(self):
        return torch.from_numpy(self.map)

    def get_matrix_from_tra(self, trajectory):
        trace_lst = []
        raw = copy.deepcopy(trajectory)
        for i in range(len(raw)):  #
            data = raw[i]
            if type(data) != np.ndarray and len(data) != 0:
                for j in range(len(data)):
                    data[j] = np.expand_dims(np.array(data[j]), axis=0)
                data = np.concatenate(data, axis=0)
                trace_lst.append(data)
            elif type(data) != np.ndarray and len(data) == 0:
                continue
            else:
                trace_lst.append(data)

        matrix = np.zeros([self.h * self.w, self.h * self.w], dtype='int32')
        print("reading trace...")
        for i in tqdm(range(len(trace_lst))):
            trace = trace_lst[i]
            for j in range(trace.shape[0] - 1):
                v1, v2 = trace[j, 0] * self.w + trace[j, 1], trace[j + 1, 0] * self.w + trace[j + 1, 1]
                matrix[v1, v2] += 1

        matrix_1 = matrix.copy().T
        matrix += matrix_1
        self.matrix = matrix
        return torch.from_numpy(matrix)

    def get_map_from_tra(self, trajectory):
        map = np.zeros([self.h, self.w, 2])
        for i in range(len(trajectory)):
            for j in range(len(trajectory[i]) - 1):
                x, y = 0, 1
                if trajectory[i][j][x] - trajectory[i][j + 1][x] != 0:
                    map[min(trajectory[i][j][x], trajectory[i][j + 1][x])
                    ][trajectory[i][j][y]][0] += 1
                else:
                    map[trajectory[i][j][x]][min(
                        trajectory[i][j][y], trajectory[i][j + 1][y])][1] += 1
        self.map = map
        return torch.from_numpy(map)

    def get_map_from_matrix(self, matrix):
        map = np.zeros([self.h, self.w, 2])
        for i in range(self.h):
            for j in range(self.w):
                if j + 1 < self.h:
                    map[i, j, 1] = matrix[i * self.w + j, i * self.w + j + 1]
                if i + 1 < self.w:
                    map[i, j, 0] = matrix[i * self.w + j, i * self.w + self.w + j]
        return map

    def get_matrix_from_map(self, map):
        matrix = np.zeros([self.h * self.w, self.h * self.w])
        for i in range(self.h):
            for j in range(self.w):
                if j + 1 < self.h:
                    matrix[i * self.w + j, i * self.w + j + 1] = map[i, j, 1]
                if i + 1 < self.w:
                    matrix[i * self.w + j, i * self.w +
                           self.w + j] = map[i, j, 0]
        return matrix

    def get_matrix_from_file(self, file):
        matrix = dl.load(os.path.abspath(file))
        self.matrix = matrix
        return matrix

    def get_map_from_file(self, file):
        map = dl.load(os.path.abspath(file))
        self.map = map
        return map

    def save_matrix(self, file):
        try:
            np.save(os.path.splitext(os.path.abspath(file))[0] + '.npy', self.matrix)
        except NameError:
            print('matrix was saved before definition.')

    def save_map(self, file):
        try:
            np.save(os.path.splitext(os.path.abspath(file))[0] + '.npy', self.map)
        except NameError:
            print('map was saved before definition.')


if __name__ == '__main__':
    from utils.util import ws

    path = './data/synthetic_data/5_5'
    grid = GetGrid(os.path.join(path, 'aoi.csv'))
    parcels = GetParcels(grid)
    parcels.generate(AOI_num=2, parcel_num=3, times=10)
    parcels.save(os.path.join(path, f'parcels.npy'))
    traj = GetTrajectory()
    dis = grid.get_dis()
    dir = grid.get_dir()
    dis = dis.reshape([grid.h * grid.w, grid.h * grid.w])
    dir = dir.reshape([grid.h * grid.w, grid.h * grid.w])
    traj.generate(grid.h, grid.w, parcels, dir, dis)
    traj.save(os.path.join(path, f'traj.npy'))
    mmgetter = GetMatrixAndMap(grid.h, grid.w)
    matrix = mmgetter.get_matrix_from_tra(traj.trajectory)
    map = mmgetter.get_map_from_tra(traj.trajectory)
    mmgetter.save_matrix(os.path.join(path, f'matrix.npy'))
    mmgetter.save_map(os.path.join(path, f'map.npy'))
