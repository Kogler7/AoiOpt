from visual_plat.shared.utils.anchor_tip import AnchorTip
from collections import defaultdict
import timeit


class XPSChecker:
    def __init__(self, tooltip: AnchorTip = None):
        self.anchor_tip = tooltip
        self.cache = defaultdict(float)
        self.last_head = ""

    def start(self):
        self.cache.clear()
        self.last_head = ""
        self.cache[""] = timeit.default_timer()

    def set_anchor_tip(self, tip: AnchorTip):
        self.anchor_tip = tip

    def check(self, head: str, tail: str = "", dif_from: str = "-1", factor: int = 1, no_tooltip: bool = False):
        self.cache[head] = timeit.default_timer()
        if dif_from == "-1":
            dif_from = self.last_head
        diff = self.cache[head] - self.cache[dif_from]
        xps = "inf"
        if diff > 0:
            xps = str(int(1 / diff * factor))
            if xps == '0':
                xps = f"-{int(diff)}"
        if self.anchor_tip is not None and not no_tooltip:
            self.anchor_tip.set(head, str(xps) + tail)
        else:
            print(f"{head}: {xps}{tail}")
        self.last_head = head