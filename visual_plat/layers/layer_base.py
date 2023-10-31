from PySide6.QtWidgets import *
from PySide6.QtGui import *


class LayerBase:
    def __init__(self, canvas):
        import visual_plat.canvas as vis_canvas
        self.canvas: vis_canvas.VisualCanvas = canvas
        self.event: vis_canvas.EventDeputy = canvas.event_deputy
        self.render: vis_canvas.RenderDeputy = canvas.render_deputy
        self.layout: vis_canvas.LayoutDeputy = canvas.layout_deputy
        self.layers: dict[str, LayerBase] = canvas.layer_dict
        self.level = 0
        self.xps_tag = ""
        self.visible = True
        self.data = None

    def set_level(self, level: int):
        """调整层级关系"""
        self.level = level
        self.canvas.layer_list.sort(key=lambda layer: layer.level, reverse=False)

    """
    由StateDeputy自动调用
    """

    def on_reload(self, data: any):
        """全局更新时调用"""
        return False

    def on_adjust(self, data: any):
        """局部更新时调用"""
        return False

    def on_reverse(self, data: any):
        """反向更新时调用"""
        return False

    """
    由RenderDeputy自动调用
    """

    def on_stage(self, device: QPixmap):
        """重绘buff图层时自动调用"""
        return False

    def on_paint(self, device: QWidget):
        """重绘时自动调用"""
        return False
    
    def get_data(self):
        """保存数据"""
        return self.data

    """
    主动调用以更新状态
    """

    def force_restage(self):
        """用于主动更新"""
        self.render.mark_need_restage()
        self.canvas.update()

    def force_repaint(self):
        """用于主动刷新"""
        self.canvas.update()

    def show(self):
        self.visible = True
        self.force_restage()

    def hide(self):
        self.visible = False
        self.force_restage()
    
    def switch_visibility(self):
        self.visible = not self.visible
        self.force_restage()
