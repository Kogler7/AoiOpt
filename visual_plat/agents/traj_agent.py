import numpy as np
from tqdm import trange
from dataclasses import dataclass
from collections import defaultdict

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor
from visual_plat.shared.static.custom_2d import area_of_points
from visual_plat.proxies.color_proxy import ColorProxy
from visual_plat.shared.static.txt_reader import TxtReader


@dataclass
class TrajInfo:
    area: QRect
    index: int
    color: QColor
    p_lst: list[QPoint]


class TrajAgent:
    def __init__(self):
        self.traj_dict = defaultdict(TrajInfo)

    def clear(self):
        self.traj_dict.clear()

    def add_trace(self, crd_list: list[QPoint], index: int, color: QColor):
        """对轨迹数据进行编制，删去不必要的中间点"""
        if len(crd_list) < 2:
            return
        info = TrajInfo(QRect(), index, color, [])
        ldr = QPoint(0, 0)
        for i in range(1, len(crd_list)):
            ndr = crd_list[i] - crd_list[i - 1]
            if ldr != ndr:
                info.p_lst.append(crd_list[i - 1])
            ldr = ndr
        info.p_lst.append(crd_list[len(crd_list) - 1])
        info.area = area_of_points(info.p_lst)
        self.traj_dict[index] = info

    def auto_read(self, data):
        """读入轨迹数据"""
        if isinstance(data, str):
            path = data
            mode = path[-3:]
            if mode == "npy":
                data = np.load(path, allow_pickle=True)
                self.read_npy(data)
            elif mode == "txt":
                self.read_lst(TxtReader.read_grouped_points(path))
            else:
                print("Data Deputy: Unexpected Read Mode.")
        elif isinstance(data, np.ndarray):
            self.read_npy(data)
        else:
            print("Data Deputy: No Data.")

    def read_npy(self, arr: np.array):
        data = []
        with trange(len(arr)) as t:
            t.set_description_str(f"Reading traces")
            for i in t:
                if len(data) < i + 1:
                    data.append([])
                for p in arr[i]:
                    data[i].append(QPoint(p[1], p[0]))
            self.read_lst(data)

    def read_lst(self, lst: list[list[QPoint]]):
        with trange(len(lst)) as t:
            t.set_description_str(f"Processing traces")
            for i in t:
                self.add_trace(lst[i], i, ColorProxy.idx_color(i, "normal"))

    def get_idx_list(self):
        return self.traj_dict.keys()

    def get_trace(self, indexes: list):
        res = []
        if indexes == [-1]:
            for _, trace in self.traj_dict.items():
                res.append(trace)
        else:
            for idx in indexes:
                if idx in self.traj_dict.keys():
                    res.append(self.traj_dict[idx])
                else:
                    print(f"TraceSteward: Trace Index ({idx}) Not Found.")
        return res
