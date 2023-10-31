import os
import numpy as np
from PIL import Image

def sampling(arr,target_shape):
    img = Image.fromarray(arr)
    img_ans = img.resize(target_shape, resample=Image.Resampling.NEAREST)
    ans = np.array(img_ans)
    return ans

def cut(arr, raw_lon, raw_lat, tar_lon_range:list, tar_lat_range:list, delta_lat,delta_lon):
    assert tar_lat_range[0] >= tar_lat_range[1]
    h_cur, w_cur = np.ceil((raw_lat-tar_lat_range[0])/ delta_lat),np.ceil((tar_lon_range[0]-raw_lon)/delta_lon)
    tar_h, tar_w = np.ceil((tar_lat_range[0] - tar_lat_range[1]) /delta_lat), np.ceil((tar_lon_range[1] - tar_lon_range[0])/delta_lon)
    h_cur,w_cur,tar_h,tar_w = h_cur.astype(np.int16),w_cur.astype(np.int16),tar_h.astype(np.int16),tar_w.astype(np.int16)
    ans = arr[h_cur:h_cur+tar_h,w_cur:w_cur+tar_w]
    return ans

if __name__=='__main__':
    print(os.path.abspath('../'))
    road_path = '../data/real_data/road_raw.npy'
    file = os.path.join(road_path)

    arr = np.load(file)
    print(f'Before cutting:{arr.shape}')
    ans = cut(arr, raw_lon=121.35,raw_lat=31.1488,tar_lon_range=[121.4910, 121.4940],tar_lat_range=[31.161, 31.157],delta_lat=0.0001739,delta_lon=0.0001875)
    print(f'After cutting:{ans.shape}')
    road_file = road_path.replace('road_raw.npy',f'road_aoi.npy')
    np.save(road_file,ans)