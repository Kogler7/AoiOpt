import time
from PySide6.QtCore import *
from visual_plat.proxies.async_proxy import AsyncWorker


class CountDownWorker(AsyncWorker):
    finished = Signal()

    def __init__(self, finished, total_time=8, time_gap=0.02):
        super(CountDownWorker, self).__init__()
        self.total_time = total_time
        self.time_gap = time_gap
        self.remaining_time = total_time
        self.finished.connect(finished)

    def ready(self):
        return self.remaining_time == self.total_time

    def reset(self):
        self.remaining_time = self.total_time

    def runner(self):
        while self.remaining_time > 0:
            self.remaining_time -= 1
            time.sleep(self.time_gap)
        self.finished.emit()
