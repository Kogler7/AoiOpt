import math


class BezierCurves:
    @staticmethod
    def ease_in_out():
        return CubicBezier(0.42, 0.0, 0.58, 1.0)


class CubicBezier:
    """三次贝塞尔曲线"""

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @staticmethod
    def evaluate_cubic(a, b, m):
        return 3 * a * (1 - m) * (1 - m) * m + \
               3 * b * (1 - m) * m * m + \
               m * m * m

    def transform(self, t):
        start = 0.0
        end = 1.0
        while True:
            midpoint = (start + end) / 2
            estimate = self.evaluate_cubic(self.a, self.c, midpoint)
            if math.fabs(t - estimate) < 0.001:
                return self.evaluate_cubic(self.b, self.d, midpoint)
            if estimate < t:
                start = midpoint
            else:
                end = midpoint
