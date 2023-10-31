from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.color_proxy import ColorProxy


class GridsLayer(LayerBase):
    def __init__(self, canvas):
        super(GridsLayer, self).__init__(canvas)
        self.show_base_lines = True  # 是否绘制基础栅格
        self.banned_base_lines = False

    def enable_base_lines(self):
        if not self.banned_base_lines:
            self.show_base_lines = True
            self.force_restage()

    def disable_base_lines(self):
        if not self.banned_base_lines:
            self.show_base_lines = False

    def switch_to_base_lines(self):
        self.banned_base_lines = not self.banned_base_lines
        self.show_base_lines = not self.banned_base_lines
        self.force_restage()

    def draw_grids(self, bias, step, size, painter: QPainter):
        layout = self.layout
        loc = bias
        while loc.x() < layout.viewport_size or loc.y() < layout.viewport_size:
            painter.drawLine(QPoint(0, loc.y()), QPoint(size, loc.y()))
            painter.drawLine(QPoint(loc.x(), 0), QPoint(loc.x(), size))
            loc += QPointF(step, step)

    def on_stage(self, device: QWidget):
        """绘制栅格"""
        layout = self.layout

        size = int(layout.viewport_size)
        step = layout.grid_gap

        pen = QPen(ColorProxy.named["LightGrey"])

        with QPainter(device) as painter:
            painter.setPen(pen)

            # 绘制基础栅格
            if self.show_base_lines:  # 优化性能
                bias = -mod_2d(layout.window_bias, layout.grid_lvl_fac) * layout.win2view_factor
                self.draw_grids(bias, step, size, painter)

            # 绘制定位栅格
            pen.setWidth(2)
            painter.setPen(pen)
            step *= layout.grid_cov_fac
            win_bias = -mod_2d(layout.window_bias, layout.grid_lvl_fac * layout.grid_cov_fac)
            bias = win_bias * layout.win2view_factor
            self.draw_grids(bias, step, size, painter)

        return True
