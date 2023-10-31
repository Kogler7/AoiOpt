import os

import numpy as np
from PySide6.QtCore import QSize, QPoint, QRect, QMutex

from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import rects_intersection, size2rect, max_2d, rect2list
from visual_plat.agents.aoi_agent import AoiAgent
from visual_plat.proxies.config_proxy import ConfigProxy
from visual_plat.proxies.async_proxy import AsyncProxy


class AoiLayer(LayerBase):
    def __init__(self, canvas):
        super(AoiLayer, self).__init__(canvas)
        """初始化AOI图层"""
        self.agent = AoiAgent()
        self.aoi_sample = QRect()
        aoi_img = self.agent.get_aoi_map()
        self.aoi_map: QPixmap = QPixmap.fromImage(aoi_img)
        self.aoi_size = self.aoi_map.size()
        self.aoi_rect = size2rect(self.aoi_size)
        self.config = ConfigProxy.layer("aoi")

        self.should_auto_back = self.config["auto_back"]
        self.keep_color = self.config["keep_color"]

        self.index = 0
        self.light_idx = 0
        self.light_mutex = QMutex()
        # 图片保存设置
        self.save_config = self.config["save_image"]
        self.should_save = self.save_config["enable"]
        self.sava_label = self.save_config["label"]
        self.save_format = self.save_config["format"]
        self.save_path = self.save_config["path"]
        self.save_size = QSize(self.save_config["width"], self.save_config["height"])
        self.max_count = self.save_config["max_count"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.tooltip = self.render.tooltip_deputy.anchor_tips["btm_lft"]

    def load_npy(self, path):
        arr = np.load(path)
        if len(arr.shape) != 2:
            print("AOI Layer: Invalid npy shape.", arr.shape)
            arr = arr.squeeze()
        self.agent.read_npy(arr, self.keep_color)
        self.reload_img()

    def load_img(self, path):
        """加载图片"""
        img = QImage(path)
        self.agent.read_img(img)
        self.reload_img()

    def save_img(self, img: QPixmap):
        """放大并保存图片"""
        if self.index <= self.max_count:
            new_img = QPixmap(self.save_size)
            new_img.fill(Qt.transparent)
            with QPainter(new_img) as painter:
                painter.setWindow(self.aoi_rect)
                painter.drawPixmap(QPoint(0, 0), img, self.aoi_rect)
            new_img.save(f"{self.save_path}/{self.sava_label}[{self.index}].{self.save_format}")
            self.index += 1

    def auto_back(self):
        if self.should_auto_back and self.aoi_sample == QRect():
            self.canvas.animate2center()

    def set_light(self):
        self.light_mutex.lock()
        print(f"AOI Layer: Set light to {self.light_idx}.")
        img = self.agent.get_aoi_map().copy()
        for x in range(img.width()):
            for y in range(img.height()):
                color = img.pixelColor(x, y)
                cval = color.red()
                val = min(255, int(cval * 2 ** self.light_idx))
                img.setPixelColor(x, y, QColor(val, val, val))
        self.aoi_map = QPixmap.fromImage(img)
        self.light_mutex.unlock()
        self.force_restage()

    def enlight(self):
        self.light_idx += 1
        AsyncProxy.run(self.set_light)

    def delight(self):
        self.light_idx -= 1
        AsyncProxy.run(self.set_light)

    def reload_img(self):
        """设置AOI图层"""
        self.aoi_map = QPixmap.fromImage(self.agent.get_aoi_map())
        self.aoi_size = self.aoi_map.size()
        self.aoi_rect = size2rect(self.aoi_size)
        self.force_restage()
        self.canvas.animate2center()

    def look_at(self):
        """查看指定区域"""
        focus = self.event.focus_rect
        if focus is None:
            return
        focus = focus.intersected(self.aoi_rect)
        aoi_set = set()
        img = self.agent.index_map
        for p in rect2list(focus):
            aoi_set.add(img[p.y(), p.x()])
        self.tooltip.set("AOIs", f"{aoi_set}")

    def look_at_cancel(self):
        self.tooltip.set("AOIs", "")

    def on_reload(self, data=None):
        """更新AOI图层"""
        if data is not None:
            self.data = data
            self.agent.auto_read(data)
        self.aoi_map: QPixmap = QPixmap.fromImage(self.agent.get_aoi_map())
        self.aoi_size = self.aoi_map.size()
        self.aoi_rect = size2rect(self.aoi_size)
        if self.should_save:
            self.save_img(self.aoi_map)
        self.force_restage()
        return True

    def on_stage(self, device: QPixmap):
        """在缓冲图层绘制AOI"""
        with QPainter(device) as painter:
            painter.setWindow(self.layout.window)
            painter.setViewport(self.layout.viewport)
            # 此处不采用win_sample，因为希望得到真正的取样大小
            self.aoi_sample = rects_intersection(self.layout.win_sample, self.aoi_rect)
            # 相应的，需要新的算法来调整绘制位置，尤其注重处理负绘制问题
            draw_bias = max_2d(-self.layout.window_bias, -self.layout.win_bias_dec)
            if self.aoi_sample != QRect():
                # aoi_sample 为0时，Qt默认绘制全部，因而需要避开
                painter.drawPixmap(draw_bias, self.aoi_map, self.aoi_sample)
        return True
