from PySide6.QtCore import QPoint


class TxtReader:
    def __init__(self):
        pass

    @staticmethod
    def read_points(path: str):
        """
        从txt文件中读取二维点列表
        """
        res = []
        with open(path) as f:
            for line in f:
                cp = line.split(' ')
                if len(cp) > 1:
                    res.append(QPoint(int(cp[0]), int(cp[1])))
        return res

    @staticmethod
    def read_grouped_points(path: str):
        """
        从txt文件中读取多组二维点列表
        """
        idx = 0
        res = [[]]
        with open(path) as f:
            for line in f:
                cp = line.split(' ')
                if len(cp) > 1:
                    res[idx].append(QPoint(int(cp[0]), int(cp[1])))
                else:
                    idx += 1
                    res.append([])
        return res
