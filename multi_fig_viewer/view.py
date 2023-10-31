import os
import time
import queue
from copy import copy, deepcopy
from PIL import Image
from importlib import import_module

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import warnings


# warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_index(nums: list):
    index = [0] * len(nums)
    indexes = [index.copy()]
    while index[0] < nums[0]:
        cur = len(nums) - 1
        index[cur] += 1
        while index[cur] >= nums[cur]:
            index[cur] = 0
            cur -= 1
            if cur >= 0:
                index[cur] += 1
            else:
                return indexes
        indexes.append(index.copy())
    return indexes


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        w, h = 1000, 600
        self.resize(w, h)
        self.setWindowTitle('Multi Fig Viewer')

        layout = QVBoxLayout()

        h1 = 100
        self.control_widget = QWidget()
        self.control_widget.setGeometry(QRect(0, 0, w, h1))

        self.dirctory_label = QLabel(self.control_widget)
        self.dirctory_label.setGeometry(QRect(10, 10, 200, 30))
        self.dirctory_label.setText('')
        self.param_name_line = QLineEdit(self.control_widget)
        self.param_name_line.setGeometry(QRect(220, 10, 300, 30))
        self.param_name_line.setPlaceholderText("input figures name (param1_param2_param3)")
        self.param_name_line.textEdited.connect(self.param_edited)
        self.dirctory_bt = QPushButton(self.control_widget)  #
        self.dirctory_bt.setGeometry(QRect(520, 10, 200, 30))
        self.dirctory_bt.setText('select dir')
        self.dirctory_bt.clicked.connect(self.get_fig_path)
        self.get_fig_label = QLabel(self.control_widget)
        self.get_fig_label.setGeometry(QRect(720, 10, 200, 30))

        # buttons
        self.param_bts = []
        param_bt_num = 10
        for i in range(param_bt_num):
            param_bt = QPushButton(self.control_widget)
            param_bt.setGeometry(QRect(i * 70, 50, 50, 30))
            param_bt.setText(f'param {i}')
            param_bt.setEnabled(False)
            param_bt.setCheckable(True)
            self.param_bts.append(param_bt)  # deepcopy(param_bt)

        # 确认按钮
        self.ok_bt = QPushButton(self.control_widget)
        self.ok_bt.setGeometry(QRect(param_bt_num * 70, 50, 50, 30))
        self.ok_bt.setText('Yes')
        self.ok_bt.setEnabled(True)
        self.ok_bt.clicked.connect(self.choose_param)

        layout.addWidget(self.control_widget)

        # 滚动条
        self.fig_widget = QWidget()
        scroll_layout = QVBoxLayout()
        self.fig_widget.setLayout(scroll_layout)
        self.scrollArea = QScrollArea()
        scroll_layout.addWidget(self.scrollArea)
        #self.scrollArea.setGeometry((QRect(10, 10, 950, 430)))
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scroll = QWidget()
        self.scrollArea.setWidget(self.scroll)
        self.figs_layout = QGridLayout()
        self.scroll.setLayout(self.figs_layout)

        layout.addWidget(self.fig_widget)
        layout.setStretchFactor(self.control_widget, 1)
        layout.setStretchFactor(self.fig_widget, 4)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def param_edited(self, s):
        param_names = s.split('_')
        self.param_name = param_names
        self.params = {}
        for param_name in param_names:
            self.params[param_name] = set()
        # self.params = dict(zip(param_names, [set()]*len(param_names)))
        self.param_num = len(param_names)
        for i in range(len(param_names)):
            self.param_bts[i].setText(param_names[i])

    def get_fig_path(self):
        directory_path = QFileDialog.getExistingDirectory(
            self.control_widget,
            "select directory saving figures",
            r'C:\Users\user\Desktop'
        )
        self.dirctory_label.setText(directory_path)
        self.fig_paths, self.fig_params = [], []
        directores = queue.Queue()  #
        directores.put(directory_path)
        while not directores.empty():
            dir = directores.get()
            dirs = os.listdir(dir)
            for dir_temp in dirs:
                dir_temp = os.path.join(dir, dir_temp)
                if os.path.isdir(dir_temp):
                    directores.put(dir_temp)
                elif os.path.isfile(dir_temp) and dir_temp.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    self.fig_paths.append(dir_temp)
                    param = os.path.basename(dir_temp).split('_')
                    param[-1] = os.path.splitext(param[-1])[0]
                    if len(param) == len(self.params.keys()):
                        fig = QtGui.QPixmap(dir_temp)
                        fig_param = dict(zip(self.params.keys(), param))
                        for i, key in enumerate(self.params.keys()):
                            self.params[key].add(param[i])
                        self.fig_params.append((fig_param, fig))

        self.get_fig_label.setText('Finded {} figures'.format(len(self.fig_params)))
        for i, key in enumerate(self.params.keys()):
            value = self.params[key]
            self.params[key] = list(value)
            if len(value) > 1:
                self.param_bts[i].setEnabled(True)

    def choose_param(self):
        row_params, col_params = [], []
        for i in range(self.param_num):
            if self.param_bts[i].isChecked():
                row_params.append(self.param_name[i])
            else:
                col_params.append(self.param_name[i])
        rows, cols = self.get_RowAndCol_params(row_params, col_params)
        row_num, col_num = len(rows), len(cols)
        old_labels = self.scrollArea.findChildren(QPushButton)
        del old_labels
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                param = dict(row, **col)
                label = QLabel()
                label.resize(1280, 400)
                for fig_param, fig in self.fig_params:
                    if fig_param == param:
                        label.setPixmap(fig.scaled(label.size(), aspectMode=Qt.KeepAspectRatio))
                        self.figs_layout.addWidget(label,j,i)

    def get_RowAndCol_params(self, row_params, col_params):
        # Row
        row_lens = [len(self.params[row_param]) for row_param in row_params]
        row_indexes = get_index(row_lens)
        rows = []
        for index in row_indexes:
            row_one = {}
            for i, j in enumerate(index):
                row_one[row_params[i]] = self.params[row_params[i]][j]
            rows.append(row_one)
        # Col
        col_lens = [len(self.params[col_param]) for col_param in col_params]
        col_indexes = get_index(col_lens)
        cols = []
        for index in col_indexes:
            col_one = {}
            for i, j in enumerate(index):
                col_one[col_params[i]] = self.params[col_params[i]][j]
            cols.append(col_one)

        return rows, cols


if __name__ == '__main__':
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
