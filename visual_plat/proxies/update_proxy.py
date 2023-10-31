import time

try:
    from visual_plat.deputies.record_deputy import RecordDeputy

    class UpdateProxy:
        canvas = None
        state_deputy: RecordDeputy = None

        @staticmethod
        def set_canvas(canvas):
            UpdateProxy.canvas = canvas
            UpdateProxy.state_deputy = canvas.state_deputy

        @staticmethod
        def reload(layer_tag: str, data=None, new_step=True):
            if UpdateProxy.canvas:
                if not UpdateProxy.state_deputy.suspended:
                    UpdateProxy.state_deputy.reload(
                        layer_tag=layer_tag, data=data, new_step=new_step)
                    while UpdateProxy.state_deputy.blocked:
                        time.sleep(0.2)
            else:
                print("UpdateProxy: Canvas is not yet set.")

        @staticmethod
        def batched_reload(tags: list[str], data: list):
            new_step = True
            for i in range(len(tags)):
                UpdateProxy.reload(tags[i], data[i], new_step=new_step)
                new_step = False

        @staticmethod
        def adjust(layer_tag: str, data=None, new_step=True):
            if UpdateProxy.canvas:
                if not UpdateProxy.state_deputy.suspended:
                    UpdateProxy.state_deputy.adjust(
                        layer_tag=layer_tag, data=data, new_step=new_step)
                    while UpdateProxy.state_deputy.blocked:
                        time.sleep(0.2)
            else:
                print("UpdateProxy: Canvas is not yet set.")

        @staticmethod
        def batched_adjust(tags: list[str], data: list):
            new_step = True
            for i in range(len(tags)):
                UpdateProxy.adjust(tags[i], data[i], new_step=new_step)
                new_step = False

        @staticmethod
        def start_record():
            UpdateProxy.state_deputy.start_record()

        @staticmethod
        def stop_record():
            UpdateProxy.state_deputy.stop_record()

        @staticmethod
        def pause():
            UpdateProxy.state_deputy.pause()

        @staticmethod
        def block():
            UpdateProxy.state_deputy.block()

        @staticmethod
        def snapshot():
            return UpdateProxy.state_deputy.snapshot()

        @staticmethod
        def start_replay(record, name=""):
            UpdateProxy.state_deputy.start_replay(record, name)

except ImportError:
    class UpdateProxy:
        @staticmethod
        def reload(layer_tag: str, data=None, new_step=True):
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def batched_reload(tags: list[str], data: list):
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def adjust(layer_tag: str, data=None, new_step=True):
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def batched_adjust(tags: list[str], data: list):
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def start_record():
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def stop_record():
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def pause():
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def block():
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def snapshot():
            print("UpdateProxy: VisualPlatform is not imported.")

        @staticmethod
        def start_replay(record, name=""):
            print("UpdateProxy: VisualPlatform is not imported.")
