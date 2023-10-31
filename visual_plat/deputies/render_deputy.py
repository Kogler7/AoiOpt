from time import sleep
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import *

from visual_plat.proxies.color_proxy import ColorProxy
from visual_plat.shared.static.custom_2d import *
from visual_plat.shared.utils.xps_checker import XPSChecker
from visual_plat.layers.layer_base import LayerBase
from visual_plat.deputies.tooltip_deputy import TooltipDeputy


class RenderDeputy:
    def __init__(self, canvas: QWidget, layers: list[LayerBase], tooltip: TooltipDeputy):
        self.canvas = canvas

        # Layers
        self.layers: list[LayerBase] = layers

        # 视图缓冲图层
        self.buff_map: QPixmap = QPixmap(self.canvas.size())

        # 是否需要重绘缓冲图层
        self.need_restage = True

        # TooltipDeputy
        self.tooltip_deputy = tooltip

        # 性能测算工具
        self.xps = XPSChecker(self.tooltip_deputy.anchor_tips["top_lft"])

    def render(self):
        """用于全部更新"""
        self.tooltip_deputy.relocate(self.canvas.size())
        self.xps.set_anchor_tip(self.tooltip_deputy.anchor_tips["top_lft"])
        self.xps.start()

        if self.need_restage:
            # 绘制视图缓冲图层
            self.need_restage = False
            self.buff_map = QPixmap(self.canvas.size())
            self.buff_map.fill(ColorProxy.named["Background"])
            self.xps.check("BINI")
            for layer in self.layers:
                rendered = False
                if layer.visible:
                    rendered = layer.on_stage(device=self.buff_map)
                if layer.xps_tag != '':
                    if rendered:
                        self.xps.check(layer.xps_tag)
                    else:
                        self.tooltip_deputy.anchor_tips["top_lft"].set(layer.xps_tag)

        self.xps.set_anchor_tip(self.tooltip_deputy.anchor_tips["top_rgt"])

        # 将缓冲图层绘制到屏幕
        with QPainter(self.canvas) as painter:
            painter.drawPixmap(QPoint(0, 0), self.buff_map)

        # 将视图图层绘制到屏幕
        self.xps.check("DPS")
        for layer in self.layers:
            if layer.visible:
                rendered = layer.on_paint(device=self.canvas)
                if rendered and layer.xps_tag != "":
                    self.xps.check(layer.xps_tag)

        self.xps.check("FPS", dif_from="")

    def mark_need_restage(self):
        """标记需要重绘"""
        self.need_restage = True
