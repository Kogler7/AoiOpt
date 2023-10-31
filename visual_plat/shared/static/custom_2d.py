import math

import numpy as np
from PySide6.QtCore import QPointF, QPoint, QRect, QSize

inf_int = int(10e8)


def int_2d(src: QPointF):
    """对点取整"""
    _x = int(src.x())
    _y = int(src.y())
    return QPoint(_x, _y)


def dec_int_2d(src: QPointF):
    """
    分别取整数和小数部分函数的2D版本
    返回两个点组成的元组
    第一个元素是小数部分组成的点
    第二个元素是整数部分组成的点
    """
    _x = math.modf(src.x())
    _y = math.modf(src.y())
    decimal = QPointF(_x[0], _y[0])
    integer = QPoint(int(_x[1]), int(_y[1]))
    return decimal, integer


def mod_2d(src: QPointF, op: float):
    """对二维点取模"""
    _x = src.x() % op
    _y = src.y() % op
    return QPointF(_x, _y)


def min_2d(p1: QPointF, p2: QPointF):
    """二维点的最小值"""
    return QPointF(min(p1.x(), p2.x()), min(p1.y(), p2.y()))


def max_2d(p1: QPointF, p2: QPointF):
    """二维点的最大值"""
    return QPointF(max(p1.x(), p2.x()), max(p1.y(), p2.y()))


def great_than_2d(op1: QPoint, op2: QPoint):
    """
    点的大小比较
    当且仅当op1>op2时返回True
    """
    return op1.x() > op2.x() and op1.y() > op2.y()


def trans_rect(rect: QRect, bias: QPoint, delt: QSize = None):
    """对Rect做变换"""
    _p = QPoint(rect.x(), rect.y()) + bias
    if delt:
        _s = rect.size() + delt
    else:
        _s = rect.size()
    return QRect(_p, _s)


def mul_rect(rect: QRect, factor: int):
    """对Rect的长宽做乘积"""
    _p = QPoint(rect.x(), rect.y()) * factor
    _s = QSize(rect.width(), rect.height()) * factor
    return QRect(_p, _s)


def rect_bias(rect: QRect):
    """返回Rect的左上顶点"""
    return QPoint(rect.x(), rect.y())


def rects_intersection(r1: QRect, r2: QRect):
    """返回r1与r2的交集"""
    if rect_overlap(r1, r2):
        tl = QPoint(max(r1.x(), r2.x()), max(r1.y(), r2.y()))
        br = QPoint(min(r1.right(), r2.right()), min(r1.bottom(), r2.bottom())) + QPoint(1, 1)
        return QRect(tl, br)
    else:
        # 二者不存在交集时返回默认值
        return QRect()


def chebyshev_dist(p1: QPoint, p2: QPoint):
    """切比雪夫距离"""
    return max(abs(p1.x() - p2.x()), abs(p1.y() - p2.y()))


def ctr_rect_cdist(ctr_r: QRect, src_p: QPoint):
    """基于rect的切比雪夫距离"""
    return max(
        min(abs(src_p.x() - ctr_r.x()), abs(src_p.x() - ctr_r.right())),
        min(abs(src_p.y() - ctr_r.y()), abs(src_p.y() - ctr_r.bottom()))
    )


def point_azimuth(p: QPointF):
    """返回二维点的方位角（弧度）"""
    return math.atan2(-p.y(), p.x()) + 2 * math.pi


def rect2list(rect: QRect):
    """Rect中的每个点加入到List"""
    for x in range(rect.x(), rect.right() + 1):
        for y in range(rect.y(), rect.bottom() + 1):
            yield QPoint(x, y)


def masked_rect2list(src_r: QRect, ctr_r: QRect, mask: QRect = QRect()):
    """
    将矩形中的点按照一定规则进行序列化
    采用切比雪夫距离和结合重心完成排序
    mask标记的矩形不会纳入计算
    """
    res: list[QPoint] = []
    center = (ctr_r.topLeft() + ctr_r.bottomRight()) / 2
    # 旋转点以保证根据方向角排序而生成的点序列依次相邻
    rotation = Rotation2D(-math.pi / 4 + 0.01)
    # 按照一定规则计算各点距离中心点的距离
    bias = src_r.topLeft()
    dist = np.zeros([src_r.height(), src_r.width()])
    for y in range(src_r.height()):
        for x in range(src_r.width()):
            _p = QPoint(x, y) + bias
            if not in_rect(_p, mask):
                theta = point_azimuth(rotation.trans(QPointF(_p) - QPointF(center)))  # 相对角度
                delta = ctr_rect_cdist(ctr_r, _p)  # 相对距离
                dist[y][x] = delta * 10 + theta
                res.append(_p)
    # 根据距离排序
    res.sort(key=lambda point: dist[point.y() - bias.y()][point.x() - bias.x()])
    return res


def size2point(size: QSize):
    """Size转化为Point"""
    return QPoint(size.width(), size.height())


def size2point_f(size: QSize):
    """Size转化为PointF"""
    return QPointF(size.width(), size.height())


def size2rect(size: QSize):
    """Size转化为Rect"""
    return QRect(QPoint(0, 0), size)


def mul_pot(p1: QPointF, p2: QPointF):
    """两点乘积"""
    return QPointF(p1.x() * p2.x(), p1.y() * p2.y())


def square_size(size: int):
    """方形Size"""
    return QSize(size, size)


def square_point(bias: int):
    """方形偏移点"""
    return QPoint(bias, bias)


def in_rect(point: QPoint, rect: QRect):
    """判断点是否在某个Rect中"""
    return rect.x() <= point.x() <= rect.right() and rect.y() <= point.y() <= rect.bottom()


def rect_overlap(r1: QRect, r2: QRect):
    """
    判断两个矩形是否重叠
    有额外裕度，以服务实际应用
    """
    return (r2.right() + 1 - r1.x()) * (r1.right() + 1 - r2.x()) > 0 and \
           (r2.bottom() + 1 - r1.y()) * (r1.bottom() + 1 - r2.y()) > 0


def area_of_points(p_lst: list[QPoint]):
    """返回点集所在的最小矩形区域"""
    lp = QPoint(inf_int, inf_int)
    rp = QPoint(-inf_int, -inf_int)
    for p in p_lst:
        lp.setX(min(lp.x(), p.x()))
        lp.setY(min(lp.y(), p.y()))
        rp.setX(max(rp.x(), p.x()))
        rp.setY(max(rp.y(), p.y()))
    return QRect(lp, rp)


def area_of_array(arr: np.array):
    """功能同上"""
    lp = QPoint(inf_int, inf_int)
    rp = QPoint(-inf_int, -inf_int)
    for p in arr:
        lp.setX(min(lp.x(), p[1]))
        lp.setY(min(lp.y(), p[0]))
        rp.setX(max(rp.x(), p[1]))
        rp.setY(max(rp.y(), p[0]))
    return QRect(lp, rp)


def dist_2d(p1: QPointF, p2: QPointF):
    """计算两点距离"""
    return math.sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)


class Rotation2D:
    """旋转（x+->y+）"""

    def __new__(cls, rad):
        return Matrix2D(
            math.cos(rad),
            math.sin(rad),
            -math.sin(rad),
            math.cos(rad)
        )


class Matrix2D:
    def __init__(self, x11, x12, x21, x22):
        self.x11 = x11
        self.x12 = x12
        self.x21 = x21
        self.x22 = x22

    def trans(self, point: QPointF):
        _x = point.x() * self.x11 + point.y() * self.x21
        _y = point.x() * self.x12 + point.y() * self.x22
        return QPointF(_x, _y)


if __name__ == "__main__":
    from visual_plat.shared.utils.xps_checker import XPSChecker

    rect = QRect(1, 1, 100, 100)
    ctr_r = QRect(2, 2, 4, 4)
    mask = QRect(2, 2, 6, 4)

    xps = XPSChecker()
    xps.start()
    # p = QPointF(1, 0)
    # ro = Rotation2D(-math.pi / 4)
    # for i in range(100000):
    #     p = ro.trans(p)
    # print(p)
    # xps.check("rotate")
    # for i in range(100000):
    #     point_azimuth(p)
    # xps.check("azi")
    # for i in range(100000):
    #     ctr_rect_cdist(rect, p)
    # xps.check("dist")

    lst = masked_rect2list(rect, ctr_r, mask)
    # print(rects_intersection(rect, mask))
    xps.check("over")
    # print(lst)

    p1 = QPoint(0, 0)
    p2 = QPoint(1, 1)
    xps.start()
    for i in range(1000000):
        chebyshev_dist(p1, p2)
    xps.check("chebyshev")
    # test(1000000)
    xps.check("chebyshev_h")
    # test(100000000)
    xps.check("chebyshev_hh")
