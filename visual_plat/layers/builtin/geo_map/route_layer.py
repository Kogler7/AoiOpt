from PySide6.QtCore import QPoint, QRect
from PySide6.QtSvg import QSvgRenderer

from visual_plat.layers.layer_base import *


class RouteLayer(LayerBase):
    def __init__(self, canvas):
        super(RouteLayer, self).__init__(canvas)
        self.svg_renderer = QSvgRenderer()
        self.buff_map = QPixmap(canvas.size())
        self.buff_map.fill(Qt.transparent)

    def on_stage(self, device: QPixmap):
        with QPainter(device) as painter:
            # painter.drawPixmap(QPoint(0, 0), self.buff_map)
            self.layout.set_crd_rect_level(6)
            self.svg_renderer.render(painter, self.layout.get_crd_rect(QPoint(-330, -200)))
            # self.svg_renderer.render(painter)
            # for i in range(10):
            #     for j in range(10):
            #         self.svg_renderer.render(painter, self.layout.get_crd_rect(QPoint(i, j)))
        return True

    def render_svg(self, path: str):
        self.svg_renderer.load(path)
        self.buff_map.fill(Qt.white)
        with QPainter(self.buff_map) as painter:
            self.svg_renderer.render(painter)
        self.force_restage()

    def start_service(self):
        self.show()
        # print("Route service started.")

    def stop_service(self):
        self.hide()
        # print("Route service stopped.")

    def trans_state(self):
        if self.visible:
            self.stop_service()
        else:
            self.start_service()
