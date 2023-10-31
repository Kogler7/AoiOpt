class EventNotifier:
    def __init__(self):
        self.registered_dict = {}

    def register(self, event: str, callback: callable):
        """主动动态注册事件"""
        if event not in self.registered_dict.keys():
            self.registered_dict[event] = []
        self.registered_dict[event].append(callback)

    def has_event(self, event: str):
        return event in self.registered_dict.keys()

    def parse_config(self, config: dict, host):
        """解析配置文件中的事件"""
        if host is not None:
            attrs = dir(host)
            for event, cbk_str in config.items():
                cbk_list = cbk_str.split(".")
                if cbk_list[0] in attrs:
                    if event not in self.registered_dict.keys():
                        self.registered_dict[event] = []
                    if len(cbk_list) == 1:
                        self.registered_dict[event].append(getattr(host, cbk_list[0]))
                    else:
                        _obj = host
                        for attr in cbk_list:
                            _obj = getattr(_obj, attr)
                        self.registered_dict[event].append(_obj)

    def invoke(self, event, data: any = None):
        """触发事件"""
        if event in self.registered_dict.keys():
            for callback in self.registered_dict[event]:
                if data is not None:
                    callback(data)
                else:
                    callback()
            return True
        return False
