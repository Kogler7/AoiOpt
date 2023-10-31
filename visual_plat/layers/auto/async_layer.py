from visual_plat.layers.layer_base import *
from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.async_proxy import *


class AutoAsyncLayer(LayerBase):
    pass

    def async_draw(self):
        pass


class AutoListAsyncLayer(LayerBase):
    pass


class AutoRectAsyncLayer(LayerBase):
    def __init__(self):
        super(AutoRectAsyncLayer, self).__init__()
        self.update_step = 3
        self.focus_triggered: bool = False
        self.buff_rect = QRect()  # 已绘制完毕的rect
        self.buff_scale = self.layout.win2view_factor  # 上次绘制时的网格边长
        self.data_size = QSize()  # 数据的逻辑尺寸
        self.bound_rect = QRect()  # 需要展示的约束rect
        self.draw_rect = QRect()  # 正在绘制的rect
        self.buff_map = QPixmap(self.data_size * self.buff_scale)
        self.fill_list: list[QPoint] = []
        self.filling: bool = False
        self.finished: bool = False
        self.min_scale = 30

    @abstractmethod
    def draw_each(self, device: QPixmap, coord: QPoint, scale: int):
        pass

    def on_stage(self, device: QPixmap):
        should_start = self.bound_diff()
        if should_start:
            AsyncProxy.run(self.async_update)

        if not self.filling:
            with QPainter(device) as painter:
                painter.drawPixmap(
                    -(self.layout.window_bias - self.bound_rect.topLeft())
                    * self.layout.win2view_factor,
                    self.buff_map,
                    mul_rect(self.bound_rect, self.layout.win2view_factor)
                )
            return True

        return False

    def on_paint(self, device: QWidget):
        if self.focus_triggered and self.event.focus_rect:
            should_start = self.bound_diff(self.event.focus_rect)
            if should_start:
                AsyncProxy.run(self.async_update)

        if self.filling:
            with QPainter(device) as painter:
                painter.drawPixmap(
                    -(self.layout.window_bias - self.bound_rect.topLeft())
                    * self.layout.win2view_factor,
                    self.buff_map,
                    mul_rect(self.bound_rect, self.layout.win2view_factor)
                )
            return True
        return False

    def bound_diff(self, bound_rect: QRect = None):
        """判断是否需要启动更新"""
        # 计算约束矩形
        if bound_rect:
            # 传入约束矩形可能会超出数据范围
            data_rect = QRect(QPoint(0, 0), self.data_size)
            bound_rect = rects_intersection(bound_rect, data_rect)
            if not bound_rect:
                return False
        else:
            # 未传入约束矩形时，根据data_size结合偏移量计算约束矩形
            device_size: QSize = self.layout.get_device_logic_size()
            device_rect = QRect(QPoint(0, 0), device_size)
            data_rect = QRect(-self.layout.win_bias_int, self.data_size)
            bound_rect = rects_intersection(device_rect, data_rect)
            if not bound_rect:
                return False
            # 以data为基准
            bound_rect = trans_rect(bound_rect, bias=self.layout.win_bias_int)

        # 约束未发生更新，不发起绘制
        if self.buff_rect == bound_rect:
            return False

        self.bound_rect = bound_rect

        # 更新缓冲矩形，判断是否需要重绘
        if self.layout.win2view_factor < self.min_scale:
            # 间距过小时，清空，不重绘
            self.buff_rect = QRect()
            return False

        if self.buff_scale != self.layout.win2view_factor:
            # 当发生缩放时，整体重绘
            self.buff_scale = self.layout.win2view_factor
            self.buff_map = QPixmap(self.data_size * self.buff_scale)
            self.buff_map.fill(QColor(0, 0, 0, 0))
            self.buff_rect = QRect()

        return True

    def rect_serialize(self):
        # 待绘矩形序列化
        # print("buff", self.buff_rect)
        # print("bound", self.bound_rect)
        intersect = rects_intersection(self.bound_rect, self.buff_rect)
        if intersect:
            # 二者有交集，即有一部分已画出，则以已画出部分开始编制
            # print("inter", intersect)
            return masked_rect2list(self.bound_rect, intersect, intersect)
        else:
            # 二者无交集，则选取待画矩形中心点进行编制
            center = QPoint(
                (self.bound_rect.x() + self.bound_rect.right()) / 2,
                (self.bound_rect.y() + self.bound_rect.bottom()) / 2
            )
            center_rect = QRect(center, QSize(1, 1))
            return masked_rect2list(self.bound_rect, center_rect)

    def async_batch_draw(self, crd_lst: list[QPoint]):
        for i in range(len(crd_lst)):
            self.filling = True
            crd = crd_lst[i]
            self.draw_each(
                self.buff_map,
                crd,
                self.layout.win2view_factor
            )
            if i % self.update_step:
                self.force_repaint()
        self.filling = False
        self.finished = True
        self.buff_rect = self.bound_rect

    def async_update(self):
        """申请新线程发起"""
        print(AsyncProxy.pool.activeThreadCount())
        lst = self.rect_serialize()
        self.async_batch_draw(lst)
