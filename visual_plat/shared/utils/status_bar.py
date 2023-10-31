from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QObject, Signal


class StatusBar(QObject):
    update_signal: Signal = Signal(str)

    def __init__(self, canvas: QWidget):
        super().__init__()
        self.canvas = canvas
        self.update_signal.connect(canvas.setWindowTitle)
        self.default = canvas.windowTitle()
        self.status: dict[str, str] = {}
        self.text = ""

    def set_default(self, default: str):
        self.default = default

    def set(self, tag: str, state: str = ""):
        self.status[tag] = state
        self.update()

    def reset(self, tag: str):
        if tag in self.status.keys():
            self.status.pop(tag)
            self.update()
        else:
            print(f"Status Bar: Key Error [{tag}]")

    def update(self):
        self.text = " ".join(f"{k} {v}" for k, v in self.status.items())
        title = self.get_text()
        self.update_signal.emit(title)

    def get_text(self):
        if self.text == "":
            return self.default
        else:
            return self.text
