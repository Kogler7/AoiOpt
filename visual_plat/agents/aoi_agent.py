import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QSize
from PySide6.QtGui import QColor, QImage
from visual_plat.proxies.color_proxy import ColorProxy


@dataclass
class AoiInfo:
    crd_set: set
    index: int = 0
    label: str = ""
    color: QColor = QColor()


class AoiInfoSet:
    def __init__(self):
        pass

    def __hash__(self):
        return hash(time.time())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        else:
            return False


class AoiAgent:
    def __init__(self):
        self.crd_count = 0
        self.aoi_size = QSize()
        self.aoi_dict = defaultdict(lambda: AoiInfo(crd_set=set()))
        self.crd_dict = {}
        self.index_map: np.ndarray = None
        self.color_map: QImage = QImage()

    def clear(self):
        self.crd_count = 0
        self.aoi_dict.clear()
        self.crd_dict.clear()

    def get_crd_num(self):
        return self.crd_count

    def get_idx_list(self):
        return self.aoi_dict.keys()

    def get_aoi_area(self, idx: int = -1):
        res = []
        if idx == -1:
            for _, aoi in self.aoi_dict.items():
                res.append(aoi)
        else:
            if idx in self.aoi_dict.keys():
                res.append(self.aoi_dict[idx])
            else:
                print(f"AOISteward: AOI Index ({idx}) Not Found.")
        return res

    def add_aoi(self, index: int, label: str = "", color: QColor = QColor()):
        if index not in self.aoi_dict.keys():
            info = self.aoi_dict[index]
            info.index = index
            info.label = label
            info.color = color

    def del_aoi(self, index: int):
        if index in self.aoi_dict.keys():
            info = self.aoi_dict.pop(index)
            for crd in info.crd_set:
                self.crd_dict.pop(crd)

    def append(self, crd: QPoint, aoi: int):
        if crd not in self.crd_dict.keys():
            if aoi not in self.aoi_dict.keys():
                self.add_aoi(index=aoi, color=ColorProxy.new_color())
            self.aoi_dict[aoi].crd_set.add(crd)
            self.crd_dict[crd] = aoi
            self.crd_count += 1
        else:
            print("请改用 change/n")

    def remove(self, crd: QPoint):
        if crd in self.crd_dict.keys():
            aoi = self.crd_dict.pop(crd)
            self.aoi_dict[aoi].crd_set.remove(crd)
            self.crd_count -= 1

    def change(self, crd: QPoint, tgt: int):
        self.remove(crd)
        self.append(crd, tgt)

    def crd2aoi(self, crd: QPoint):
        if crd in self.crd_dict.keys():
            return self.crd_dict.get(crd)
        return None

    def auto_read(self, data):
        """读入AOI数据"""
        if isinstance(data, str):
            path = data
            mode = path[-3:]
            if mode == "npy":
                self.read_npy(np.load(path))
            elif mode == "jpg" or mode == "png":
                img = QImage()
                if img.load(path):
                    self.read_img(img)
                else:
                    print("Aoi Agent: Image Reading Failed.")
            else:
                print("Aoi Agent: Unexpected Read Mode.")
        elif isinstance(data, np.ndarray):
            self.read_npy(data)
        else:
            print("Aoi Agent: No Data.")

    def read_img(self, img: QImage):
        self.color_map = img
        self.index_map = np.zeros((img.height(), img.width()))
        self.aoi_size = self.color_map.size()

    def read_npy(self, arr: np.array, keep_color=False):
        shape = arr.shape
        self.color_map = QImage(shape[1], shape[0], QImage.Format_RGBA8888)
        self.index_map = arr.copy()
        for y in range(shape[0]):
            for x in range(shape[1]):
                if keep_color:
                    val = int(arr[y][x])
                    self.color_map.setPixelColor(
                        QPoint(int(x), int(y)),
                        QColor(val, val, val)
                    )
                else:
                    self.color_map.setPixelColor(
                        QPoint(int(x), int(y)),
                        ColorProxy.idx_color(int(arr[y][x]), c_type="all")
                    )
        self.aoi_size = self.color_map.size()

    def get_aoi_map(self, idx: int = -1):
        return self.color_map
