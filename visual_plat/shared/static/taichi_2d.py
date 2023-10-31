import taichi as ti

ti.init(arch=ti.gpu)
mat3x3i = ti.types.matrix(3, 3, ti.i32)


@ti.func
def chebyshev_dist_h(x1: ti.int32, y1: ti.int32, x2: ti.int32, y2: ti.int32):
    """切比雪夫距离"""
    return ti.max(ti.abs(x1 - x2), ti.abs(y1 - y2))


@ti.func
def chebyshev_dist_m(x: mat3x3i, y: mat3x3i, ctr_x: ti.int32, ctr_y: ti.int32):
    """切比雪夫距离"""
    return ti.max(ti.abs(x - ctr_x), ti.abs(y - ctr_y))


@ti.func
def point_azimuth_h(x: ti.int32, y: ti.int32):
    """返回二维点的方位角（弧度）"""
    return ti.atan2(-y, x) + 2 * ti.pi


@ti.kernel
def test() -> mat3x3i:
    x = mat3x3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = mat3x3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return chebyshev_dist_m(x, y, 1, 1)


if __name__ == "__main__":
    print(test())
