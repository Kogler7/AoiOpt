from visual_plat.layers.auto.async_layer import *


class MarkLayer(LayerBase):
    def __init__(self, canvas):
        super(MarkLayer, self).__init__(canvas)
        self.data = None
        self.aoi_layer = canvas.get_layer("aoi")
        self.framed = False

    def on_reload(self, data):
        self.data = data
        self.force_restage()

    def load_map(self, path):
        self.data = np.load(path)
        self.force_restage()

    def frame_end(self):
        self.framed = True
        self.force_restage()

    def frame_begin(self):
        self.framed = False
        self.force_restage()

    def on_stage(self, device: QPixmap):
        if self.layout.win2view_factor < 30:
            # 太小了看不清不要了
            return False
        with QPainter(device) as painter:
            if self.framed:
                focus = self.event.focus_rect
                if focus is None:
                    return
                focus = focus.intersected(self.aoi_layer.aoi_rect)
                img = self.aoi_layer.agent.index_map
                for p in rect2list(focus):
                    loc = self.layout.get_central(p)
                    painter.drawText(loc, f"{img[p.y(), p.x()]}")
            if self.data is not None:
                self.data = self.data.astype(np.int32)
                factor = self.layout.win2view_factor
                for y in range(self.data.shape[0]):
                    for x in range(self.data.shape[1]):
                        loc = self.layout.crd2pos(QPoint(x, y))
                        rt_loc = loc + QPoint(factor, factor / 2)
                        bt_loc = loc + QPoint(factor / 2 - 5, factor)
                        painter.drawText(rt_loc, str(self.data[y][x][1]))
                        painter.drawText(bt_loc, str(self.data[y][x][0]))
        return True
