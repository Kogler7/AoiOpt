from importlib import import_module


class KeyEventNotifier:
    def __init__(self):
        self.key_map = import_module("PySide6.QtCore").Qt.__dict__
        self.registered_dict = {}

    def parse_key_str(self, key_str: str):
        """将按键字符串解析为按键标记"""
        key_list: list[str] = [key.capitalize() for key in key_str.split("+")]
        for i in range(len(key_list)):
            if i + 1 == len(key_list):
                key_list[i] = "Key_" + key_list[i]
            else:
                key_list[i] = key_list[i] + "Modifier"
        key_list = [self.key_map[key] for key in key_list]
        modifier: int = 0
        for _mod in key_list[:-1]:
            modifier |= int(_mod)
        key = key_list[-1]
        return modifier, key

    def register(self, event: str, callback: callable):
        """主动动态注册事件"""
        modifier, key = self.parse_key_str(event)
        if modifier not in self.registered_dict.keys():
            self.registered_dict[modifier] = {}
        if key not in self.registered_dict[modifier].keys():
            self.registered_dict[modifier][key] = []
        self.registered_dict[modifier][key].append(callback)

    def parse_config(self, config: dict, host):
        """将配置解析为按键标记和对应函数的字典"""
        if host is not None:
            attrs = dir(host)
            for key_str, cbk_str in config.items():
                modifier, key = self.parse_key_str(key_str)
                cbk_list = cbk_str.split(".")
                if cbk_list[0] in attrs:
                    if modifier not in self.registered_dict.keys():
                        self.registered_dict[modifier] = {}
                    if key not in self.registered_dict[modifier].keys():
                        self.registered_dict[modifier][key] = []
                    if len(cbk_list) == 1:
                        self.registered_dict[modifier][key].append(getattr(host, cbk_list[0]))
                    else:
                        _obj = host
                        for attr in cbk_list:
                            _obj = getattr(_obj, attr)
                        self.registered_dict[modifier][key].append(_obj)

    def invoke(self, modifier, key):
        """根据按键标记调用对应函数"""
        modifier = int(modifier)
        if modifier in self.registered_dict.keys():
            if key in self.registered_dict[modifier].keys():
                for callback in self.registered_dict[modifier][key]:
                    callback()
