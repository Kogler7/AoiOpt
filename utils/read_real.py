import numpy as np
import pandas as pd
import os
from time import time
import warnings

warnings.filterwarnings("ignore")

from utils.util import ws
from utils.read_data import DataHandler


def read_real_data(delta_lon=0.001, delta_lat=0.001, lon_range=[121.4910, 121.4940], lat_range=[31.161, 31.157]):
    file, header = os.path.join(ws, 'data/real_data/trajectory_small.csv'), 0
    print(f'read data {file}')
    dataHandler = DataHandler(filepath=file,header=header, nrows=5e4)
    [h, w], matrix, map, parcle_num, parcels = dataHandler.handle_lonlat(delta_lon=delta_lon, delta_lat=delta_lat, lon_range=lon_range, lat_range=lat_range)
    return [h, w], matrix, map, parcle_num, parcels

if __name__ == '__main__':
    read_start = time()
    [h, w], matrix, map, parcle_num, parcels = read_real_data(delta_lon=0.0001875,
                                                         delta_lat=0.0001739 ,
                                                         lon_range=[121.4910 , 121.4940],
                                                         lat_range=[31.161 , 31.157],
                                                         )
    print(map.shape)
    dir = 'data/real_data/1'
    map_path = os.path.join(ws,dir,'map.npy')
    np.save(map_path, map)
    matrix_path = os.path.join(ws,dir,'matrix.npy')
    np.save(matrix_path,matrix)
    aoi_path = os.path.join(ws,dir,'aoi.npy')
    aoi = (np.arange(h*w)+1).reshape([h,w])
    np.save(aoi_path, aoi)
    read_end = time()
    print(f'read used {read_end - read_start:.2f}s.')
