from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.color_proxy import ColorProxy


class FocusLayer(LayerBase):
    def __init__(self, canvas):
        super(FocusLayer, self).__init__(canvas)
        self.focus_brush = QBrush(QColor(255, 255, 255, 100))
        self.focus_rect_pen = QPen(ColorProxy.named["LightGrey"])
        self.focus_rect_pen.setWidth(2)

    def on_paint(self, device: QWidget):
        """绘制聚焦框"""
        with QPainter(device) as painter:
            if self.event.focus_rect:
                tl = self.layout.crd2pos(self.event.focus_rect.topLeft())
                br = self.layout.crd2pos(self.event.focus_rect.bottomRight() + QPoint(1, 1))
            else:
                tl = self.layout.crd2pos(self.event.focus_point)
                br = self.layout.crd2pos(self.event.focus_point + QPoint(1, 1))
            rect = QRect(tl, br)
            path = QPainterPath()
            painter.setRenderHint(QPainter.Antialiasing)
            path.addRoundedRect(rect, 5, 5)
            painter.setPen(self.focus_rect_pen)
            painter.drawPath(path)
            painter.fillPath(path, self.focus_brush)
        return True
