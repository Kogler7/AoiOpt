import numpy as np
from tqdm import trange
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor
from visual_plat.shared.static.custom_2d import area_of_points, area_of_array
from visual_plat.proxies.color_proxy import ColorProxy
from visual_plat.shared.static.txt_reader import TxtReader


@dataclass
class ParcelInfo:
    area: QRect
    index: int
    color: QColor
    p_lst: list[QPoint]


class ParcelAgent:
    def __init__(self):
        self.parcels_dict: dict[int, ParcelInfo] = {}

    def clear(self):
        self.parcels_dict.clear()

    def add_parcels(self, crd_list: list[QPoint], index: int, color: QColor):
        area = area_of_points(crd_list)
        info = ParcelInfo(area, index, color, [])
        info.p_lst = crd_list
        self.parcels_dict[index] = info

    def auto_read(self, data):
        """读入包裹数据"""
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
        with trange(len(arr)) as t:
            t.set_description_str(f"Reading parcels")
            for i in t:
                area = area_of_array(arr[i])
                info = ParcelInfo(area, i, ColorProxy.idx_color(i), [])
                for p in arr[i]:
                    info.p_lst.append(QPoint(p[1], p[0]))
                self.parcels_dict[i] = info

    def read_lst(self, lst: list[list[QPoint]]):
        for i in range(len(lst)):
            self.add_parcels(lst[i], i, ColorProxy.idx_color(i, "dark"))

    def get_idx_list(self):
        return self.parcels_dict.keys()

    def get_parcels(self, indexes: list):
        res = []
        if indexes == [-1]:
            for _, parcels in self.parcels_dict.items():
                res.append(parcels)
        else:
            for idx in indexes:
                if idx in self.parcels_dict.keys():
                    res.append(self.parcels_dict[idx])
                else:
                    print(f"ParcelSteward: Parcels Index ({idx}) Not Found.")
        return res
