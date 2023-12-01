# coding=UTF-8
import warnings
from algorithm.pfrl_learing.aoi_learning import PolicyGradientPFRL
from algorithm.pfrl_learing.aoi_config import Config
import utils.data_loader as dl
from utils.options import KeyConfig

try:
    from visual_plat.platform import VisualPlatform, VisualCanvas
    from visual_plat.proxies.update_proxy import UpdateProxy


    def canvas_config(vis_canvas: VisualCanvas):
        # vis_canvas.layer_dict["traj"].agent.auto_read(config.trace_path)
        UpdateProxy.reload("weight", dl.load(config.map_path))
except ImportError:
    print("Visual Platform is not installed, please install it first.")


    def canvas_config(vis_canvas):
        pass

warnings.filterwarnings("ignore")
options = KeyConfig()
opts = options.parse()
config = Config(opts)


def main():
    aoi_learning = PolicyGradientPFRL(config)
    train_mode, test_mode = False, False
    if opts.mode == 'train':
        train_mode = True
    elif opts.mode == 'test':
        test_mode = True

    if train_mode:
        aoi_learning.config.load_model = False
        aoi_learning.config.train = True
        aoi_learning.config.test = False
        VisualPlatform.launch(async_task=aoi_learning.execute, canvas_config=canvas_config)
        # aoi_learning.execute()
    if test_mode:
        aoi_learning.config.load_model = True
        aoi_learning.config.train = False
        aoi_learning.config.test = True
        VisualPlatform.launch(async_task=aoi_learning.execute, canvas_config=canvas_config)


if __name__ == '__main__':
    main()