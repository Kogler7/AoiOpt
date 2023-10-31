from PySide6.QtCore import QPointF, QSize
from PySide6.QtWidgets import QWidget
from visual_plat.shared.utils.anchor_tip import AnchorTip, AnchorLocate
from visual_plat.shared.utils.floating_tip import FloatingTip
from visual_plat.proxies.config_proxy import ConfigProxy


class TooltipDeputy:
    def __init__(self, canvas: QWidget):
        self.canvas = canvas
        self.size: QSize = canvas.size()
        anchor_bias = ConfigProxy.tooltip_config("anchor_bias")
        self.anchor_bias = QPointF(anchor_bias[0], anchor_bias[1])
        self.anchor_tips: dict[str, AnchorTip] = {
            "cursor": AnchorTip(AnchorLocate.btm_lft),
            "top_lft": AnchorTip(AnchorLocate.top_lft, self.anchor_bias, self.size),
            "top_rgt": AnchorTip(AnchorLocate.top_rgt, self.anchor_bias, self.size),
            "btm_lft": AnchorTip(AnchorLocate.btm_lft, self.anchor_bias, self.size),
            "btm_rgt": AnchorTip(AnchorLocate.btm_rgt, self.anchor_bias, self.size)
        }
        self.floating_tips: dict[str, FloatingTip] = {}

    def add_anchor_tip(self, tag: str, tooltip: AnchorTip):
        self.anchor_tips[tag] = tooltip

    def add_floating_tip(self, tooltip: FloatingTip):
        pass

    def relocate(self, size: QSize):
        self.size = size
        for tip in self.anchor_tips.values():
            tip.relocate(size)

    def draw(self):
        for tip in self.anchor_tips.values():
            tip.draw(self.canvas)
