import os
import numpy as np
from queue import Queue


class RLChecker():
    def __init__(self, h=3, w=3, color_num=3, path=None):
        self.h, self.w = h, w
        self.color_num = color_num
        self.grid = np.ones([self.h * self.w], dtype='int8')
        self.check_dir_x, self.check_dir_y = [-1, 1, 0, 0], [0, 0, -1, 1]
        self.path = path
        self.ans = []

    def get_all_aoi(self):
        num = 0
        f = self.generate0()
        try:
            while True:
                grid, legal = next(f)  #
                if legal:
                    num += 1
                    self.ans.append(np.expand_dims(grid, axis=0).copy())

        except:
            print('generated over! total num is {}'.format(num))
            ans = np.concatenate(self.ans, axis=0)
            print(ans.shape)
            if self.path is not None:
                np.save(os.path.join(self.path, '{}-{}-aoi.npy'.format(self.h, self.w)), ans)
                print(os.path.abspath(os.path.join(self.path, '{}-{}-aoi.npy'.format(self.h, self.w))))

    def generate0(self):
        while self.grid[0] <= 1:
            grid = self.grid.view().reshape([self.h, self.w])
            legal = self.check_unicom(grid)
            yield grid, legal

            vert = -1
            self.grid[vert] += 1
            while self.grid[vert] >= self.color_num + 1 and vert > -self.h * self.w:
                self.grid[vert] = 1
                vert -= 1
                self.grid[vert] += 1

    def generate1(self):
        self.grid = self.grid.reshape([self.h, self.w])
        colors = dict(zip(np.arange(0, self.color_num + 1), [0] * (self.color_num + 1)))
        colors[1] = self.h * self.w
        q = []
        q.append((0, 0))
        while self.grid[0, 0] <= 1 and len(q) != 0:
            vert_top, c_top = q[-1]
            h, w = vert_top // self.w, vert_top % self.w
            if h == 0 and w == 0:
                min_color, other_color = 1, 1
                max_color = 1
            elif h > 0 and w == 0:
                min_color, max_color = self.grid[h - 1, w], self.grid[h - 1, w]
                other_color = 0
            elif h == 0 and w > 0:
                min_color, max_color = self.grid[h, w - 1], self.grid[h, w - 1]
                other_color = 0
            else:
                min_color = min(self.grid[h - 1, w], self.grid[h, w - 1])
                other_color = max(self.grid[h - 1, w], self.grid[h, w - 1])
                max_color = max(self.grid[h - 1, w], self.grid[h, w - 1])

            choose_c = np.arange(max_color, self.color_num + 1).tolist()
            choose_c.append(max(min_color, c_top + 1))
            choose_c = list(set(choose_c))
            change_c_flag = False
            for c in choose_c:
                if c <= 0:
                    continue
                else:
                    colors[self.grid[h, w]] -= 1
                    self.grid[h, w] = c
                    colors[c] += 1

                    change_c_flag = True
                    if vert_top == self.w * self.h - 1:
                        legal = self.check_unicom(self.grid)
                        yield self.grid, legal
                    else:
                        q[-1] = (vert_top, c)
                        q.append((vert_top + 1, c - 1))
                        break

            if not change_c_flag:
                q.pop()

    def check_unicom(self, grid):
        q = Queue(self.h * self.w)
        vis = np.zeros([self.h, self.w], dtype=np.bool8)
        aois = []
        for i in range(self.h):
            for j in range(self.w):
                if not vis[i, j]:
                    if grid[i, j] not in aois:
                        color = grid[i, j]
                        aois.append(color)
                        q.put((i, j))
                        vis[i, j] = True
                        while not q.empty():
                            i1, j1 = q.get()
                            for dx, dy in zip(self.check_dir_x, self.check_dir_y):
                                i2, j2 = i1 + dx, j1 + dy
                                if self.check_xy(i2, j2) and grid[i2, j2] == color and not vis[i2, j2]:
                                    q.put((i2, j2))
                                    vis[i2, j2] = True
                    else:
                        return False
        return True

    def check_rotate(self, grid):
        pass

    def check_xy(self, i, j):
        if 0 <= i < self.h and 0 <= j < self.w:
            return True
        return False


if __name__ == '__main__':
    checker = RLChecker(5, 5, 2, path='../data/synthetic_data/5_5')
    checker.get_all_aoi()
