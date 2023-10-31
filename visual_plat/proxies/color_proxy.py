from PySide6.QtGui import QColor
from visual_plat.proxies.config_proxy import ConfigProxy


class ColorProxy:
    global_count = 0
    normal_cnt = 0
    dark_cnt = 0
    color_dict = {
        "light": [],
        "normal": [],
        "dark": [],
        "all": []
    }
    named = {
        "Background": QColor(240, 240, 240),
        "LightGrey": QColor(225, 225, 225),
        "LightDark": QColor(100, 100, 100),
        "Indigo": QColor(75, 0, 130),  # 靛青
        "DoderBlue": QColor(30, 144, 255),  # 道奇蓝
        "Gold": QColor(255, 215, 0),  # 金色
    }

    @classmethod
    def init(cls):
        color_dict: dict = ConfigProxy.get("colors")
        for cate, colors in color_dict.items():
            alpha = colors["alpha"]
            for c_str in colors["value"]:
                color = cls.hex_to_rgb(c_str)
                cls.color_dict[cate].append(QColor(color[0], color[1], color[2], alpha))
        cls.color_dict["all"] = cls.color_dict["light"] + cls.color_dict["normal"] + cls.color_dict["dark"]

    @classmethod
    def new_color(cls, c_type: str = "light"):
        """按顺序返回一个颜色"""
        if c_type not in cls.color_dict.keys():
            print(f"ColorProxy: Color Type ({c_type}) Not Found.")
            return QColor(0, 0, 0)
        if not cls.color_dict[c_type]:
            cls.init()
        cls.global_count += 2
        return cls.color_dict[c_type][cls.global_count % len(cls.color_dict[c_type])]

    @classmethod
    def idx_color(cls, idx: int, c_type: str = "light"):
        """根据idx返回对应颜色"""
        step_factor = 7 if c_type == "all" else 3
        if c_type not in cls.color_dict.keys():
            return QColor(0, 0, 0)
        if not cls.color_dict[c_type]:
            cls.init()
        return cls.color_dict[c_type][idx * step_factor % len(cls.color_dict[c_type])]

    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
