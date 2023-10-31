import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import os
from tqdm import trange

replace_suffix = ["", "txt", "csv", "xlsx", "xls", "jpg", "png", "_"]


def load(path: str = None):
    """load data"""
    if path.find(":") < 0:
        path = os.path.join(os.getcwd(), path)
    if (path.find('map') >= 0):
        return DataLoader.read_map(path)
    elif (path.find('aoi') >= 0):
        return DataLoader.read_aoi(path)
    elif (path.find('trace') >= 0 or path.find('traj') >= 0):
        return DataLoader.read_traces(path)
    elif (path.find('matrix') >= 0):
        return DataLoader.read_matrix(path)
    elif (path.find('parcel') >= 0):
        return DataLoader.read_parcels(path)


class DataLoader:
    @staticmethod
    def pre_read(path):
        '''check if the file exists and its type'''
        if os.path.exists(path):
            exists = True
            portion = os.path.splitext(path)
            if portion[1] != '.npy':
                newpath = portion[0] + ".npy"
            else:
                newpath = path
            return exists, newpath, portion[1]
        else:
            print("\033[0;31;40m","Dont have this file=>", path, "\033[0m")
            exists = False
            newpath = path.replace("matrix", "trace")
            portion = os.path.splitext(newpath)
            if portion[1] != '.npy':
                newpath = portion[0] + ".npy"
            return exists, newpath, None

    @staticmethod
    def read_grouped_excel(path):
        dat = pd.read_excel(path, header=None)
        idx = 0
        res = [[]]
        for i in trange(len(dat)):
            if pd.isnull(dat.iat[i, 0]):
                idx += 1
                res.append([])
            else:
                res[idx].append((int(dat.iat[i, 0]), int(dat.iat[i, 1])))
        return res

    @staticmethod
    def read_grouped_points(path, mode):
        """
        read list form txt
        """
        idx = 0
        res = [[]]
        with open(path) as f:
            for line in f:
                if mode == ".txt":
                    cp = line.split(' ')
                    if len(cp) > 1:
                        res[idx].append((int(cp[0]), int(cp[1])))
                    else:
                        idx += 1
                        res.append([])
                elif mode == ".csv":
                    cp = line.split(',')
                    if cp[0] != '':
                        res[idx].append((int(cp[0]), int(cp[1])))
                    else:
                        idx += 1
                        res.append([])
        return res

    @staticmethod
    def read_aoi(path: str = None):
        """Read AOI"""
        newaoi = None
        if os.path.exists(path):
            newaoi = np.load(path)
        else:
            for i in range(1, 8):
                try_path = path.replace("npy", replace_suffix[i])
                if (os.path.exists(try_path)):
                    break
            if i == 1:
                newaoi = np.loadtxt(try_path)
            elif i == 2:
                newaoi = np.loadtxt(try_path, delimiter=',',
                                    encoding="utf-8-sig", dtype=np.int)
            elif i == 3 or i == 4:
                newaoi = pd.read_excel(try_path, header=None).values
            elif i == 5 or i == 6:
                newaoi = mpimg.imread(try_path)
            else:
                print("\033[0;31;40m", "Don't have any type of this file=>", path.replace(
                    "npy", "..."), "\033[0m")
                return None
            np.save(path, newaoi)
            print("save as npy file=>", path)
        print("successful read data=>", path)
        return newaoi

    @staticmethod
    def read_parcels(path: str = None):
        """read parcels"""
        parcel = None
        if os.path.exists(path):
            parcel = np.load(path, allow_pickle=True)
        else:
            for i in range(1, 8):
                try_path = path.replace("npy", replace_suffix[i])
                if (os.path.exists(try_path)):
                    break
            if i == 1:
                parcel = DataLoader.read_grouped_points(try_path, ".txt")
            elif i == 2:
                parcel = DataLoader.read_grouped_points(try_path, ".csv")
            elif i == 3 or i == 4:
                parcel = DataLoader.read_grouped_excel(try_path)
            else:
                print("\033[0;31;40m", "Don't have any type of this file=>", try_path.replace(
                    "npy", "..."), "\033[0m")
                return None
            np.save(path, parcel)
            print("save as npy file=>", path)
        print("successful read data=>", path)
        return parcel

    @staticmethod
    def read_traces(path: str = None):
        """read trajectories"""
        return DataLoader.read_parcels(path)

    @staticmethod
    def read_matrix(path: str = None):
        """read adjacency matrix"""
        matrix = None
        if os.path.exists(path):
            matrix = np.load(path, allow_pickle=True)
        else:
            for i in range(1, 8):
                try_path = path.replace("npy", replace_suffix[i])
                if (os.path.exists(try_path)):
                    break
            if i <= 4:
                matrix = DataLoader.read_aoi(path)
            else:
                print("\033[0;31;40m", "Don't have any type of this file=>", path.replace(
                            "npy", "..."), "\033[0m")
                return None
        print("successful read data=>", path)
        return matrix

    def read_map(path: str = None):
        """read map"""
        if os.path.exists(path):
            map = np.load(path)
        else:
            print("\033[0;31;40m", "Don't have any type of this file=>", path.replace(
                            "npy", "..."), "\033[0m")
            return None
        print("successful read data=>", path)
        return map
