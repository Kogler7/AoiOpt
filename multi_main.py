import multiprocessing
import warnings

from utils.options import MultiConfig

from algorithm.pfrl_learing.aoi_learning import PolicyGradientPFRL
from algorithm.pfrl_learing.aoi_config import Config
import utils.data_loader as dl

# from visual_plat.platform import VisualPlatform, VisualCanvas
# from visual_plat.proxies.update_proxy import UpdateProxy

warnings.filterwarnings("ignore")


# def canvas_config(vis_canvas: VisualCanvas):
#     # vis_canvas.layer_dict["traj"].agent.auto_read(config.trace_path)
#     # UpdateProxy.reload("weight", dl.load(config.map_path))
#     pass


def train_worker(config):
    aoi_learning = PolicyGradientPFRL(config)
    aoi_learning.config.load_model = False
    aoi_learning.config.debug_main = True
    aoi_learning.config.train = True
    aoi_learning.config.test = False
    aoi_learning.execute()


def test_worker(config):
    aoi_learning = PolicyGradientPFRL(config)
    aoi_learning.config.load_model = True
    aoi_learning.config.debug_main = True
    aoi_learning.config.multi = True
    aoi_learning.config.train = False
    aoi_learning.config.test = True
    aoi_learning.execute()


if __name__ == '__main__':
    options = MultiConfig()
    opts = options.get_config()
    configs = []
    for opt in opts:
        config = Config(opt)
        config.debug_loss, config.debug_time = False, False
        configs.append(config)

    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=1)
    pool.daemon = True

    print(f'current worker is test worker!')  #train
    for config in configs:
        # pool.apply_async(train_worker, (config,))
        pool.apply_async(test_worker, (config,))
    pool.close()
    pool.join()