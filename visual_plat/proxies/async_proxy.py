from PySide6.QtCore import *
from abc import abstractmethod


class AsyncWorker(QObject):
    @abstractmethod
    def runner(self):
        pass


class AsyncProxy:
    pool = QThreadPool.globalInstance()

    def __new__(cls, *args, **kwargs):
        cls.pool.setMaxThreadCount(20)
        return cls.pool

    @staticmethod
    def start(worker: AsyncWorker):
        """申请线程执行 Worker"""
        AsyncProxy.pool.start(worker.runner)

    @staticmethod
    def run(runner):
        """获取线程执行函数"""
        AsyncProxy.pool.start(runner)

    def redirect(self):
        pass
