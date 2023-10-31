from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.color_proxy import ColorProxy
from visual_plat.agents.traj_agent import TrajAgent
from tqdm import tqdm


class TrajLayer(LayerBase):
    def __init__(self, canvas):
        super(TrajLayer, self).__init__(canvas)
        self.agent = TrajAgent()
        self.data = []  # traj_indexes
        self.trace_maps: dict[int, QPixmap] = {}
        self.trace_area_dict: dict[int, QRect] = {}
        self.enable_dot_line = True  # 是否使用点线绘制轨迹
        self.dot_back_line_pen = QPen(ColorProxy.named["LightGrey"])
        """初始化轨迹图层"""
        self.on_reload()
        self.indexes_list = [0]

    def set_indexes(self, indexes: list[int]):
        self.data = indexes
        self.on_reload()

    def load_traj(self, path):
        data = np.load(path, allow_pickle=True)
        self.agent.read_npy(data)
        self.indexes_list = self.agent.get_idx_list()
        if len(self.indexes_list) > 10:
            self.set_indexes([i for i in range(10)])
        else:
            self.set_indexes([-1])

    def draw_dot_line(self, painter: QPainter, fst: QPoint, sec: QPoint):
        """绘制点线"""
        if self.enable_dot_line:
            pen = painter.pen()
            self.dot_back_line_pen.setWidth(pen.width() + 2)
            painter.setPen(self.dot_back_line_pen)
            painter.drawLine(fst, sec)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([5, 1, 1, 5])
            painter.setPen(pen)
        painter.drawLine(fst, sec)

    def on_reload(self, data=None):
        if data:
            self.data: list[int] = data
        if self.data:
            lst = self.agent.get_trace(self.data)
            with tqdm(total=len(lst)) as t:
                for trace in lst:
                    t.update()
                    t.set_description_str("Painting traces")
                    bias = rect_bias(trace.area)
                    self.trace_area_dict[trace.index] = trace.area
                    pixmap = QPixmap(trace.area.size() * 15)
                    pixmap.fill(QColor(0, 0, 0, 0))
                    pen = QPen(trace.color)
                    with QPainter(pixmap) as painter:
                        painter.setPen(pen)
                        nt = []
                        for p in trace.p_lst:
                            _p = (p - bias) * 15
                            nt.append(_p + QPoint(7, 7))
                        for i in range(1, len(nt)):
                            self.draw_dot_line(painter, nt[i - 1], nt[i])
                    self.trace_maps[trace.index] = pixmap
        self.force_restage()
        return True

    def on_stage(self, device: QPixmap):
        """在缓冲图层绘制轨迹"""
        with QPainter(device) as painter:
            def draw_trace_at(idx: int):
                area = self.trace_area_dict[idx]
                if rect_overlap(area, layout.win_sample):
                    _map = self.trace_maps[idx]
                    _loc = - layout.window_bias + area.topLeft()
                    painter.drawPixmap(_loc * 15, _map)

            layout = self.layout
            painter.setWindow(mul_rect(layout.window, 15))
            painter.setViewport(layout.viewport)
            if self.data != [-1]:
                for i in self.data:
                    if i in self.trace_maps.keys():
                        draw_trace_at(i)
            else:
                for i in self.trace_maps.keys():
                    draw_trace_at(i)
        return True
