import time
from importlib import import_module

from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import warnings

import visual_plat.platform as platform

from visual_plat.deputies.layout_deputy import LayoutDeputy
from visual_plat.deputies.menu_deputy import MenuDeputy
from visual_plat.deputies.record_deputy import RecordDeputy
from visual_plat.deputies.render_deputy import RenderDeputy
from visual_plat.deputies.event_deputy import EventDeputy
from visual_plat.deputies.tooltip_deputy import TooltipDeputy

from visual_plat.shared.static.custom_2d import *
from visual_plat.shared.static.bezier_curves import *
from visual_plat.shared.utils.status_bar import StatusBar
from visual_plat.shared.utils.notifier.key_notifier import KeyEventNotifier

from visual_plat.proxies.config_proxy import ConfigProxy
from visual_plat.proxies.async_proxy import AsyncProxy

from visual_plat.layers.layer_base import LayerBase
from visual_plat.layers.builtin.grid_layer.aoi_layer import AoiLayer

warnings.filterwarnings("ignore", category=DeprecationWarning)


class VisualCanvas(QWidget):
    def __init__(self, on_init_finished=None):
        super(VisualCanvas, self).__init__()

        # status
        self.status_bar = StatusBar(self)

        # 事件监听
        self.key_notifier = KeyEventNotifier()

        init_size = ConfigProxy.canvas_config("init_size")
        self.resize(init_size[0], init_size[1])

        # Layers
        self.layer_dict: dict[str] = {}
        self.layer_list: list[LayerBase] = []

        # Deputies
        self.tooltip_deputy = TooltipDeputy(self)
        self.state_deputy = RecordDeputy(
            layers=self.layer_dict, status_bar=self.status_bar)
        self.event_deputy = EventDeputy(self)
        self.render_deputy = RenderDeputy(
            self, layers=self.layer_list, tooltip=self.tooltip_deputy)
        self.layout_deputy = LayoutDeputy(size=self.size())
        self.menu_deputy = MenuDeputy(
            self, self.render_deputy.tooltip_deputy.anchor_tips["cursor"])

        # Layers 载入，需要在 Deputy 声明后完成
        layers_config = ConfigProxy.get("layers")
        self.load_layers_by_config(layers_config)
        self.aoi_layer: AoiLayer = self.layer_dict["aoi"]

        # TooltipProxy
        self.tooltip_proxy = self.render_deputy.tooltip_deputy
        self.tooltip_proxy.anchor_tips["btm_rgt"].set("LEVEL", "0")

        # Other
        self.setMouseTracking(True)  # 开启鼠标追踪
        self.setAcceptDrops(True)  # 接受拖拽
        self.show_tooltips = ConfigProxy.tooltip_config("visible")

        # 预处理
        if on_init_finished:
            on_init_finished(self)

        # Teleport
        self.animate2center()

    def mount_layer(self, layer_cls, layer_tag: str, layer_cfg: dict):
        layer_obj = layer_cls(self)
        if layer_tag in self.layer_dict.keys():
            raise Exception("VisualCanvas: Layer tag already exists.")
        if "level" in layer_cfg.keys():
            layer_obj.level = layer_cfg["level"]
        if "xps_tag" in layer_cfg.keys():
            layer_obj.xps_tag = layer_cfg["xps_tag"]
        if "visible" in layer_cfg.keys():
            layer_obj.visible = layer_cfg["visible"]
        if "event" in layer_cfg.keys():
            self.event_deputy.bind_layer_event(layer_obj, layer_cfg["event"])
        self.layer_dict[layer_tag] = layer_obj
        self.layer_list.append(layer_obj)
        self.layer_list.sort(key=lambda layer: layer.level, reverse=False)

    def get_layer(self, tag: str) -> LayerBase:
        return self.layer_dict[tag]

    def load_layers_by_config(self, layers_config: dict):
        """根据配置文件载入图层"""

        def load_layers(pth_base: str, layers: dict):
            mod_back = "_layer"
            nme_back = "Layer"
            for tag, layer_cfg in layers.items():
                ConfigProxy.load_layer(tag, layer_cfg)
                module = pth_base + tag + mod_back
                mod = import_module(module)
                name = tag.capitalize() + nme_back
                layer_cls = getattr(mod, name)
                self.mount_layer(layer_cls, tag, layer_cfg)

        path_base = "visual_plat.layers."
        load_layers(path_base + "builtin.geo_map.",
                    layers_config["builtin"]["geo_map"])
        load_layers(path_base + "builtin.grid_layer.",
                    layers_config["builtin"]["grid_layer"])
        load_layers(path_base + "custom.", layers_config["custom"])

    def unmount_layer(self, tag: str):
        """卸载外部图层"""
        self.layer_list.remove(self.layer_dict[tag])
        self.layer_dict.pop(tag)

    def paintEvent(self, event):
        """窗口刷新时被调用，完全交由 Render Deputy 代理"""
        self.render_deputy.render()
        if self.show_tooltips:
            self.tooltip_deputy.draw()

    def on_sliding(self):
        """中键滑动"""
        while self.event_deputy.on_sliding:
            delt = -(self.event_deputy.start_sliding_pos -
                     self.event_deputy.last_mouse_pos) / 20
            self.layout_deputy.translate(delt)
            self.render_deputy.mark_need_restage()
            self.update()
            time.sleep(0.01)

    def animate_to(self, target: QPointF = QPointF(0, 0)):
        """滑动至（另起线程调用）"""
        self.event_deputy.on_sliding = True
        self.event_deputy.view_notifier.invoke("slide_begin")
        self.event_deputy.try_start_view_update()

        def step():
            curve = BezierCurves.ease_in_out()
            init_bias = self.layout_deputy.window_bias
            distance = target - init_bias
            p = 0.0
            while p < 1.0:
                p += 0.05
                self.layout_deputy.teleport(
                    init_bias + distance * curve.transform(p))
                self.render_deputy.mark_need_restage()
                self.update()
                time.sleep(0.01)
            self.event_deputy.on_sliding = False
            self.event_deputy.view_notifier.invoke("slide_end")
            self.event_deputy.try_end_view_update()

        AsyncProxy.run(step)

    def animate2center(self):
        """滑动至中心"""
        target = self.layout_deputy.get_central_bias(self.aoi_layer.aoi_size)
        self.animate_to(-target)

    def mousePressEvent(self, event):
        """鼠标按下时调用"""
        event = self.layout_deputy.wrap_event(event)
        self.event_deputy.on_mouse_press(event)

        if self.event_deputy.on_sliding:
            self.setCursor(Qt.SizeAllCursor)
            AsyncProxy.run(self.on_sliding)
        elif not self.event_deputy.on_dragging:
            self.setCursor(Qt.ArrowCursor)

        self.tooltip_proxy.anchor_tips["cursor"].show()
        self.tooltip_proxy.anchor_tips["cursor"].move(
            event.pos + QPoint(10, -20))
        self.tooltip_proxy.anchor_tips["cursor"].set(
            "At", f"({event.crd.x()}, {event.crd.y()})")

        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            """滑动回原点"""
            self.setCursor(Qt.ArrowCursor)
            self.event_deputy.on_sliding = False
            self.animate2center()

    def mouseMoveEvent(self, event):
        """鼠标移动时调用"""
        if self.event_deputy.on_dragging:
            self.layout_deputy.translate(
                self.event_deputy.last_mouse_pos - event.pos())
            self.render_deputy.mark_need_restage()

        event = self.layout_deputy.wrap_event(event)
        self.event_deputy.on_mouse_move(event)

        self.render_deputy.tooltip_deputy.anchor_tips["cursor"].hide()
        self.render_deputy.tooltip_deputy.anchor_tips["btm_lft"].set(
            "CRD", f"({event.crd.x()}, {event.crd.y()})")
        self.render_deputy.tooltip_deputy.anchor_tips["btm_lft"].set(
            "POS", f"({event.pos.x()}, {event.pos.y()})")

        self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放时调用"""
        self.event_deputy.on_mouse_release(event)

        self.tooltip_proxy.anchor_tips["btm_lft"].set("POS")
        self.tooltip_proxy.anchor_tips["top_lft"].set("FPS")

        self.update()

    def wheelEvent(self, event):
        """滚动鼠标滚轮时调用"""
        success = self.layout_deputy.zoom_at(
            event.angleDelta().y(), self.event_deputy.last_mouse_pos
        )
        if success:
            self.event_deputy.on_mouse_wheel(event)

            self.tooltip_proxy.anchor_tips["btm_rgt"].set(
                "LEVEL", str(self.layout_deputy.grid_level))

            self.render_deputy.mark_need_restage()
            self.update()

    @staticmethod
    def create_new_canvas():
        """创建新画布"""
        platform.VisualPlatform.new_canvas()

    def snapshot_and_create_new_canvas(self):
        """截图并创建新画布"""
        path, _ = self.state_deputy.snapshot()

        def pre_setter(canvas: VisualCanvas):
            rcd = canvas.state_deputy.load_record(path)
            canvas.state_deputy.apply_snapshot(rcd)

        platform.VisualPlatform.new_canvas(on_init_finished=pre_setter)

    def record(self):
        """记录"""
        if self.state_deputy.recording:
            self.state_deputy.stop_record()
        else:
            self.state_deputy.start_record()

    def keyPressEvent(self, event: QKeyEvent):
        self.key_notifier.invoke(event.modifiers(), event.key())
        self.event_deputy.on_key_pressed(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """检测是否拖拽文件进入窗口"""
        if event.mimeData().hasUrls():
            tp = event.mimeData().urls().pop().toLocalFile().split(".")[-1]
            if tp == "rcd":
                event.accept()
            if self.event_deputy.drag_notifier.has_event(tp):
                event.accept()

    def dropEvent(self, event: QDropEvent):
        """放下文件时触发"""
        url = event.mimeData().urls()
        drp_path = url.pop().toLocalFile()
        drp_full_name = drp_path.split("/")[-1].split(".")
        drp_name = drp_full_name[0]
        drp_type = drp_full_name[-1]
        if drp_type == "rcd":
            self.accept_record(drp_path, drp_name)
            self.status_bar.set(f"File loaded:", f"[{drp_name}] ({drp_type})")
        elif self.event_deputy.drag_notifier.has_event(drp_type):
            success = self.event_deputy.drag_notifier.invoke(
                drp_type, drp_path)
            if success:
                self.status_bar.set(
                    f"File loaded:", f"[{drp_name}] ({drp_type})")

    def accept_record(self, drp_path: str, drp_name: str):
        rcd = self.state_deputy.load_record(drp_path)
        if not hasattr(rcd, 'comp_idx'):
            print(
                "RECORD has no COMPATIBLE INDEX. [It may be derived from ANCIENT versions]")
        elif rcd.comp_idx != ConfigProxy.record_config("compatible_index"):
            print(f"The RECORD is not COMPATIBLE with this VERSION of canvas. "
                  f"[{rcd.comp_idx} != {ConfigProxy.record_config('compatible_index')}]")
        else:
            self.state_deputy.start_replay(rcd, drp_name)
            self.animate2center()

    def resizeEvent(self, event):
        """改变窗口大小时调用"""
        self.layout_deputy.resize(event.size())
        self.render_deputy.mark_need_restage()
        self.update()

    def server_switch(self):
        """启动服务器"""
        # self.server = Server(self)
        # self.server.start()
        print("Server started.")
        pass

    def switch_server_host(self):
        """按下键盘"""
        print("key down")
        pass
