from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from visual_plat.shared.utils.anchor_tip import AnchorTip


class MenuDeputy:
    def __init__(self, canvas: QWidget, tooltip: AnchorTip):
        self.allow_menu = False
        self.canvas = canvas
        self.set_style()
        self.tooltip = tooltip
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.pop_menu)
        self.pop_menu = QMenu(canvas)
        self.pop_menu.setWindowOpacity(0)
        self.pop_menu.addAction(QAction('切换至 AOI >', canvas))
        self.pop_menu.addAction(QAction('选择并高亮 >', canvas))
        self.pop_menu.addSeparator()
        self.pop_menu.addAction(QAction('绘制模式 >', canvas))

    def pop_menu(self, point: QPoint):
        if self.allow_menu:
            self.pop_menu.exec(self.canvas.mapToGlobal(point))

    def set_style(self):
        self.canvas.setStyleSheet(
            "QMenu{background:#F5F5F5;}"  # 选项背景颜色
            "QMenu{border:none;}"  # 设置整个菜单框的边界高亮厚度

            "QMenu::item{padding:0px 5px 0px 5px;}"  # 以文字为标准，右边距文字40像素，左边同理
            "QMenu::item{height:20px;}"  # 显示菜单选项高度
            "QMenu::item{color:#212121;}"  # 选项文字颜色
            "QMenu::item{background:transparent;}"  # 选项背景
            "QMenu::item{margin:3px 6px 3px 6px;}"  # 每个选项四边的边界厚度，上，右，下，左

            "QMenu::item:selected:enabled{background:#BDBDBD;}"
            "QMenu::item:selected:enabled{color:#F5F5F5;}"  # 鼠标在选项上面时，文字的颜色
            # "QMenu::item:selected:!enabled{background:red;}"  # 鼠标在上面时，选项背景为不透明

            "QMenu::separator{height:1px;}"  # 要在两个选项设置self.groupBox_menu.addSeparator()才有用
            "QMenu::separator{width:50px;}"
            "QMenu::separator{background:#757575;}"
            "QMenu::separator{margin:2px 5px 2px 5px;}"

        )
