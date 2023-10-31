from collections import defaultdict
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from enum import Enum


class AnchorLocate(Enum):
    top_lft = 0
    top_rgt = 1
    btm_lft = 2
    btm_rgt = 3


class AnchorTip:
    def __init__(
            self,
            anchor: AnchorLocate = AnchorLocate.top_lft,
            anchor_bias: QPoint = QPoint(100, 50),
            device_size: QSize = QSize(1000, 800)
    ):
        self.visible = True
        self.anchor = anchor
        self.anchor_bias = anchor_bias
        self.locate = self.get_locate(anchor_bias, device_size)
        self.tips = defaultdict(str)
        self.text = ""
        self.font = QFont("等线", 12, 3)
        self.back_height = 30
        self.path_pen = QPen(QColor(255, 255, 255))
        self.path_brush = QBrush(QColor(240, 240, 240))
        self.text_pen = QPen(QColor(50, 50, 50))
        self.text_pen.setWidth(3)

    def get_locate(self, anchor_bias: QPoint, device_size: QSize):
        if self.anchor is AnchorLocate.top_lft:
            return anchor_bias
        elif self.anchor is AnchorLocate.top_rgt:
            return QPointF(device_size.width() - anchor_bias.x(), anchor_bias.y())
        elif self.anchor is AnchorLocate.btm_lft:
            return QPointF(anchor_bias.x(), device_size.height() - anchor_bias.y())
        elif self.anchor is AnchorLocate.btm_rgt:
            return QPointF(
                device_size.width() - anchor_bias.x(), device_size.height() - anchor_bias.y()
            )
        else:
            raise

    def relocate(self, device_size: QSize):
        self.locate = self.get_locate(self.anchor_bias, device_size)

    def get_rect(self):
        if self.text == "":
            return QRect()
        metrics = QFontMetrics(self.font)
        width = metrics.horizontalAdvance(self.text) + 10
        height = self.back_height
        if self.anchor is AnchorLocate.top_lft:
            return QRect(int(self.locate.x()), int(self.locate.y()), width, height)
        elif self.anchor is AnchorLocate.top_rgt:
            return QRect(int(self.locate.x()) - width, int(self.locate.y()), width, height)
        elif self.anchor is AnchorLocate.btm_lft:
            return QRect(int(self.locate.x()), int(self.locate.y()) - height, width, height)
        elif self.anchor is AnchorLocate.btm_rgt:
            return QRect(int(self.locate.x()) - width, int(self.locate.y()) - height, width, height)
        else:
            raise

    def draw(self, device: QWidget):
        if self.visible:
            painter = QPainter(device)
            painter.setRenderHint(QPainter.Antialiasing)
            rect = self.get_rect()
            path = QPainterPath()
            path.addRoundedRect(rect, 5, 5)
            painter.setPen(self.path_pen)
            painter.fillPath(path, self.path_brush)
            painter.setPen(self.text_pen)
            painter.setFont(self.font)
            painter.drawText(QPointF(rect.x() + 5, rect.y() + 20), self.text)

    def move(self, tgt=QPoint(0, 0)):
        self.locate = tgt

    def set(self, key="", words=""):
        self.tips[key] = words
        self.text = " | ".join([f"{str(k)}:{v}" for k, v in self.tips.items() if v != ""])
        
    def get(self, key=""):
        if key in self.tips:
            return self.tips[key]

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False
