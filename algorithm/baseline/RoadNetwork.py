import os
import numpy as np
import pandas as pd

from utils.util import ws
from utils.read_data import DataHandler

if __name__=='__main__':
    # read parcels
    h, w = 10, 10
    file = os.path.join(ws,f'data/synthetic_data/{h}_{w}/road_aoi.npy')
    if not os.path.exists(file):
        file.replace('npy','.csv')
        if not os.path.exists(file):
            print('Could not find the road network segmentation!')
            exit()
        else:
            ans = pd.read_csv(file,header=None).values
    else:
        ans = np.load(file)

    print(ans.shape)
    np.save(f'RoadNetwork_{h}_{w}.npy', ans)