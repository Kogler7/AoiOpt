from visual_plat.shared.static.custom_2d import *
from visual_plat.proxies.config_proxy import ConfigProxy


class LayoutDeputy:
    def __init__(self, size: QSize):
        # 基础参数
        self.size = size  # Device Size
        self.zoom_step = ConfigProxy.layout_config("zoom_step")  # 每级缩放增加的方格数量
        self.precision = ConfigProxy.layout_config("precision")  # 地理坐标精度，即每个方格对应的经纬度变化量

        # 基础偏移量
        self.geo_info = ConfigProxy.canvas_config("geo_info")  # 地理坐标相对逻辑坐标偏移
        self.geo_locate = QPointF(self.geo_info["locate"][0], self.geo_info["locate"][1])
        self.geo_eastern = self.geo_info["eastern"]
        self.geo_southern = self.geo_info["southern"]
        self.pos_bias: QPoint = QPoint()  # 物理坐标相对逻辑坐标偏移

        # 逻辑视窗
        self.window_size = ConfigProxy.layout_config("init_window_size")  # 逻辑视窗所能容纳的最大方格数量
        self.min_window_size = ConfigProxy.layout_config("min_window_size")  # 最小尺寸
        self.window = QRect(0, 0, self.window_size, self.window_size)

        # 逻辑偏移量
        self.window_bias: QPointF = QPointF(0, 0)  # 基坐标相对视窗逻辑坐标的偏移，单位：格
        self.win_bias_dec, self.win_bias_int = dec_int_2d(self.window_bias)

        # 取样视窗（窗口偏移版）
        self.win_sample = trans_rect(self.window, self.win_bias_int, QSize(1, 1))

        # 物理视窗
        self.viewport_size = max(self.size.width(), self.size.height())  # 尺寸（长宽最大值）
        self.viewport = QRect(0, 0, self.viewport_size, self.viewport_size)

        # 映射因数
        self.view2win_factor = self.window_size / self.viewport_size  # 物理坐标转逻辑坐标因数
        self.win2view_factor = self.viewport_size / self.window_size  # 逻辑坐标转物理坐标因数

        # 栅格参数
        self.grid_level = 0  # 栅格缩放等级
        self.grid_lvl_max = ConfigProxy.layout_config("max_grid_level")
        self.grid_cov_fac = ConfigProxy.layout_config("grid_convert_factor")  # 基础栅格和定位栅格的换算因子
        self.grid_lvl_fac = self.grid_cov_fac ** self.grid_level  # 栅格间距的换算因子
        self.grid_gap_min = ConfigProxy.layout_config("min_grid_gap")  # 栅格最小间距
        self.grid_gap_max = self.grid_gap_min * self.grid_cov_fac  # 栅格最大间距
        self.grid_gap = self.win2view_factor * self.grid_lvl_fac  # 基础栅格间距

        # 节约计算
        self.crd_rect_level = 0
        self.crd_rect_fac = self.grid_cov_fac ** self.crd_rect_level

    def calculate_factors(self):
        # 计算映射因数
        self.view2win_factor = self.window_size / self.viewport_size
        self.win2view_factor = self.viewport_size / self.window_size

    def crd2pos(self, crd: QPoint):
        """逻辑坐标转物理坐标"""
        res = (QPointF(crd) - self.window_bias) * self.win2view_factor
        return QPoint(int(res.x()), int(res.y()))

    def crd2pos_f(self, crd: QPointF):
        """逻辑坐标转物理坐标（高精度）"""
        return (crd - self.window_bias) * self.win2view_factor

    def pos2crd(self, pos: QPoint):
        """物理坐标转逻辑坐标"""
        res = QPointF(pos) * self.view2win_factor + self.window_bias
        return QPoint(math.floor(res.x()), math.floor(res.y()))

    def pos2crd_f(self, pos: QPointF):
        """物理坐标转逻辑坐标（高精度）"""
        return pos * self.view2win_factor + self.window_bias

    def crd2geo(self, crd: QPoint):
        """逻辑坐标转地理坐标"""
        _crd = crd.__copy__()
        if not self.geo_eastern:
            _crd.setX(-crd.x())
        if not self.geo_southern:
            _crd.setY(-crd.y())
        return QPointF(_crd) * self.precision + self.geo_locate

    def geo2crd(self, geo: QPointF):
        """地理坐标转逻辑坐标"""
        crd = (geo - self.geo_locate) / self.precision
        if not self.geo_eastern:
            crd.setX(-crd.x())
        if not self.geo_southern:
            crd.setY(-crd.y())
        return crd

    def translate(self, delt: QPointF):
        """平移视窗"""
        self.window_bias += QPointF(delt) * self.view2win_factor
        self.win_bias_dec, self.win_bias_int = dec_int_2d(self.window_bias)
        self.win_sample = trans_rect(self.window, self.win_bias_int, QSize(1, 1))

    def teleport(self, target: QPoint):
        """传送视窗"""
        self.window_bias = target
        self.win_bias_dec, self.win_bias_int = dec_int_2d(self.window_bias)
        self.win_sample = trans_rect(self.window, self.win_bias_int, QSize(1, 1))

    def zoom_at(self, angle: float, point: QPoint):
        """定点缩放"""
        # 乘以level因子以保证在不同level下有相同的缩放速率
        # 应保证delt为整数，否则会出现错位现象，暂时原因不明
        div_fac = 120 if abs(angle) >= 120 else 10 if abs(angle) >= 10 else 3
        delt = -int(angle / div_fac) * self.grid_lvl_fac

        # 限制缩放区间以免影响视觉效果甚至溢出
        if (delt > 0 and self.grid_gap > self.grid_lvl_max) or \
                (delt < 0 and self.window_size > self.min_window_size):
            # 高精度计算缩放量，并更新相关参数
            p_loc = self.pos2crd_f(QPointF(point))
            self.window_size += delt
            self.window = QRect(0, 0, self.window_size, self.window_size)
            self.calculate_factors()

            # 计算缩放后鼠标所在点的偏移量，更新偏移参数以实现定点缩放
            n_loc = self.pos2crd_f(QPointF(point))
            self.window_bias += p_loc - n_loc
            self.win_bias_dec, self.win_bias_int = dec_int_2d(self.window_bias)
            self.win_sample = trans_rect(self.window, self.win_bias_int, QSize(1, 1))

            # 更新栅格参数，level取值范围为 [0, 10]，level 小于0时无意义，而超过10时会溢出
            if self.grid_gap < self.grid_gap_min:
                if self.grid_level < self.grid_lvl_max:
                    self.grid_level += 1
            elif self.grid_gap > self.grid_gap_max:
                if self.grid_level > 0:
                    self.grid_level -= 1
            self.grid_lvl_fac = self.grid_cov_fac ** self.grid_level
            self.grid_gap = self.win2view_factor * self.grid_lvl_fac
            self.set_crd_rect_level(self.grid_level)
            return True  # 缩放成功
        return False  # 缩放失败

    def resize(self, size=QSize()):
        """改变窗口大小"""
        self.size = size

        # 计算窗口（物理）Size该变量，将其映射再向上取整后得到Window_size改变量
        delt = max(self.size.width(), self.size.height()) - self.viewport_size
        # 保证缩放时不出现非整数变量，否则视觉效果会发生抖动
        win_delt = math.ceil(delt * self.view2win_factor)

        # 设置新的Window_size和Viewport_size，注意保持比例不变
        self.window_size += win_delt
        self.viewport_size = self.window_size * self.win2view_factor

        # 计算更新其他受影响的参数
        self.window = QRect(0, 0, self.window_size, self.window_size)
        self.viewport = QRect(0, 0, self.viewport_size, self.viewport_size)
        self.win_sample = trans_rect(self.window, self.win_bias_int, QSize(1, 1))

    def get_central(self, crd=QPoint(0, 0)):
        """获取基准坐标所对应方格的中心屏幕坐标"""
        bias = int(self.win2view_factor / 2)
        return self.crd2pos(crd) + square_point(bias)

    def wrap_event(self, event):
        pos = event.pos()
        event.pos = pos
        event.crd = self.pos2crd(pos)
        return event

    def get_central_bias(self, aoi_size: QSize):
        """返回当AOI区域处于视窗中心时的逻辑偏移量"""
        win_size = size2point_f(self.size) / self.win2view_factor
        bias = (win_size - size2point_f(aoi_size)) / 2
        return bias

    def get_device_logic_size(self):
        """返回物理视窗对应的逻辑大小"""
        h = math.ceil(self.size.height() * self.view2win_factor)
        w = math.ceil(self.size.width() * self.view2win_factor)
        return QSize(w, h)

    def set_crd_rect_level(self, level: int):
        """设置坐标矩形的level"""
        self.crd_rect_level = level
        self.crd_rect_fac = self.grid_cov_fac ** self.crd_rect_level

    def get_crd_rect(self, crd: QPoint):
        """返回指定逻辑坐标的矩形"""
        pos = self.crd2pos(crd)
        size = self.crd_rect_fac * self.win2view_factor
        siz = QSize(size, size)
        return QRect(pos, siz)
