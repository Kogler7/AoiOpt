from queue import Queue
from abc import abstractmethod

from PySide6.QtCore import *
from visual_plat.layers.layer_base import *


class AutoRenderLayer(LayerBase):
    def __init__(self):
        super(AutoRenderLayer, self).__init__()
        self.update_step = 3
        self.focus_triggered: bool = False
        self.buff_rect = QRect()


class AutoListLayer(LayerBase):
    pass


class AutoRectLayer(LayerBase):
    def __init__(self):
        super(AutoRectLayer, self).__init__()
        self.max_level = 3
        self.render_queue = Queue()

    @abstractmethod
    def draw_each(self, device: QPixmap, coord: QPoint, scale: int, level: int):
        pass
