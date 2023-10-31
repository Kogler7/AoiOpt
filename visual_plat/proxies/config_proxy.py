import json
import os.path
from glob import glob


class ConfigProxy:
    loaded = False
    plat_path = "."
    config: dict[str, dict] = {}
    layer_config: dict[str, dict] = {}

    @staticmethod
    def load():
        config_path = os.path.abspath(f"{ConfigProxy.plat_path}\\visual_plat\\configs\\*.json")
        path = os.path.join(config_path)
        config_paths = glob(path)
        for pth in config_paths:
            with open(pth) as f:
                tag = pth.split('\\')[-1][:-5]
                ConfigProxy.config[tag] = json.load(f)
        ConfigProxy.loaded = True

    @staticmethod
    def load_layer(tag: str, config: dict):
        """载入图层配置（方便图层访问）"""
        ConfigProxy.layer_config[tag] = config

    @staticmethod
    def layer(tag):
        return ConfigProxy.layer_config[tag]

    @staticmethod
    def get(tag):
        return ConfigProxy.config[tag]

    @staticmethod
    def path(tag):
        return ConfigProxy.config["paths"][tag]

    @staticmethod
    def canvas(tag):
        return ConfigProxy.config["canvas"][tag]

    @staticmethod
    def get_version():
        ver_cfg = ConfigProxy.canvas("version")
        version = f"{ver_cfg['major']}.{ver_cfg['minor']}.{ver_cfg['patch']}"
        if not ConfigProxy.canvas("release"):
            pre_cfg = ConfigProxy.canvas("preview")
            version += f"-pre-{pre_cfg['phase']}.{pre_cfg['patch']}"
        return version

    @staticmethod
    def canvas_config(tag):
        return ConfigProxy.config["canvas"]["config"][tag]

    @staticmethod
    def canvas_setting(tag):
        return ConfigProxy.config["canvas"]["setting"][tag]

    @staticmethod
    def canvas_event(tag):
        return ConfigProxy.config["canvas"]["event"][tag]

    @staticmethod
    def layout_config(tag):
        return ConfigProxy.canvas_config("layout")[tag]

    @staticmethod
    def tooltip_config(tag):
        return ConfigProxy.canvas_config("tooltip")[tag]

    @staticmethod
    def record_config(tag):
        return ConfigProxy.canvas_config("record")[tag]
