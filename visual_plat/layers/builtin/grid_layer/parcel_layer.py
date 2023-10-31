from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.color_proxy import ColorProxy
from visual_plat.agents.parcel_agent import ParcelAgent
from tqdm import tqdm


class ParcelLayer(LayerBase):
    def __init__(self, canvas):
        super(ParcelLayer, self).__init__(canvas)
        self.agent = ParcelAgent()
        self.data = []  # parcels_indexes
        self.parcels_maps: dict[int, QPixmap] = {}
        self.parcels_area_dict: dict[int, QRect] = {}
        """初始化包裹图层"""
        self.on_reload()
        self.indexes_list = [0]

    def set_indexes(self, indexes):
        self.data = indexes
        self.on_reload()

    def load_parcels(self, path):
        data = np.load(path, allow_pickle=True)
        self.agent.read_npy(data)
        self.indexes_list = self.agent.get_idx_list()
        if len(self.indexes_list) > 10:
            self.set_indexes([i for i in range(10)])
        else:
            self.set_indexes([-1])

    def on_reload(self, data=None):
        if data:
            self.data: list[int] = data
        if self.data:
            lst = self.agent.get_parcels(self.data)
            brush_bk = QBrush(ColorProxy.named["LightGrey"])
            step_p = QPoint(1, 1)
            step_s = QSize(-2, -2)
            with tqdm(total=len(lst)) as t:
                for parcels in lst:
                    t.update()
                    t.set_description_str("Painting parcels")
                    bias = rect_bias(parcels.area)
                    self.parcels_area_dict[parcels.index] = parcels.area
                    pixmap = QPixmap(parcels.area.size() * 10)
                    pixmap.fill(QColor(0, 0, 0, 0))
                    color = parcels.color
                    brush_id = QBrush(color)
                    with QPainter(pixmap) as painter:
                        for p in parcels.p_lst:
                            _p = (p - bias) * 10
                            rect = QRect(_p + step_p, QSize(8, 8))
                            painter.fillRect(rect, brush_id)
                            rect = trans_rect(rect, step_p, step_s)
                            painter.fillRect(rect, brush_bk)
                            rect = trans_rect(rect, step_p, step_s)
                            painter.fillRect(rect, brush_id)
                    self.parcels_maps[parcels.index] = pixmap
        self.force_restage()
        return True

    def on_stage(self, device: QPixmap):
        """在缓冲图层绘制包裹"""
        with QPainter(device) as painter:
            def draw_parcels_at(idx: int):
                area = self.parcels_area_dict[idx]
                if rect_overlap(area, layout.win_sample):
                    _map = self.parcels_maps[idx]
                    _loc = - layout.window_bias + area.topLeft()
                    painter.drawPixmap(_loc * 10, _map)

            layout = self.layout
            painter.setWindow(mul_rect(layout.window, 10))
            painter.setViewport(layout.viewport)
            if self.data != [-1]:
                for i in self.data:
                    if i in self.parcels_maps.keys():
                        draw_parcels_at(i)
            else:
                for i in self.parcels_maps.keys():
                    draw_parcels_at(i)
        return True
