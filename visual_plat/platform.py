try:
    from PySide6.QtWidgets import QApplication
    from visual_plat.canvas import VisualCanvas
    from visual_plat.layers.layer_base import LayerBase
    from visual_plat.proxies.async_proxy import AsyncProxy
    from visual_plat.proxies.config_proxy import ConfigProxy
    from visual_plat.proxies.update_proxy import UpdateProxy

    """
    可视化服务平台
    提供全局代理的初始设置服务
    """

    class VisualPlatform:
        canvas_list: list[VisualCanvas] = []

        @staticmethod
        def launch(
                async_task: callable = None,
                canvas_init: callable = None,
                canvas_config: callable = None,
                plat_path: str = '.'
        ):
            """
            启动可视化平台
            可附带执行一个异步任务
            请不要在重复调用此函数
            也不要在此函数后再执行其他同步任务
            此函数全局仅允许被调用一次
            """
            ConfigProxy.plat_path = plat_path  # 平台路径
            if not ConfigProxy.loaded:
                ConfigProxy.load()
                app = QApplication([])
                released = ConfigProxy.canvas("release")
                version = ConfigProxy.get_version()
                if not released:
                    print("\033[31m[AoiOpt Visual Platform] Warning: "
                          f"Version {version} is under Test and may NOT be "
                          "Stable.\033[0m")
                app_name = f"AoiOpt Visual Platform {version}"
                app.setApplicationName(app_name)
                canvas = VisualPlatform.new_canvas(app_name, canvas_init)
                UpdateProxy.set_canvas(canvas)
                if canvas_config is not None:
                    canvas_config(canvas)
                if async_task is not None:
                    AsyncProxy.run(async_task)
                app.exec()
            else:
                raise RuntimeError("VisualPlatform has been launched.")

        @staticmethod
        def new_canvas(title: str = "New Canvas", on_init_finished: callable = None):
            canvas = VisualCanvas(on_init_finished=on_init_finished)
            VisualPlatform.canvas_list.append(canvas)
            canvas.setWindowTitle(title)
            canvas.status_bar.set_default(title)
            canvas.key_notifier.parse_config(
                ConfigProxy.canvas_event("key"), canvas)
            canvas.show()
            return canvas

        @staticmethod
        def get_canvas(index: int = 0):
            if VisualPlatform.canvas_list and len(VisualPlatform.canvas_list) > index:
                return VisualPlatform.canvas_list[index]
            return None

        @staticmethod
        def mount_layer(layer: LayerBase, tag: str, config: dict = None):
            canvas = VisualPlatform.get_canvas()
            if canvas:
                canvas.mount_layer(layer, tag, config)
            else:
                return lambda canvas: canvas.mount_layer(layer, tag, config)

except ImportError:
    class VisualPlatform:
        @staticmethod
        def launch(*args, **kwargs):
            raise RuntimeError("PySide6 is not installed.")

        @staticmethod
        def new_canvas(*args, **kwargs):
            raise RuntimeError("PySide6 is not installed.")

        @staticmethod
        def get_canvas(*args, **kwargs):
            raise RuntimeError("PySide6 is not installed.")

        @staticmethod
        def mount_layer(*args, **kwargs):
            raise RuntimeError("PySide6 is not installed.")
