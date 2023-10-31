import os
import numpy as np

from utils.util import ws
from utils.read_data import DataHandler

def value(v,C):
    '''
    GCLP value function
    param: v,[int],node.
    param: C,[ndarray 1*n int],nodes in cluster.
    output:w,[float],the value if v divided into C;[0,1].
    '''
    global mode,matrix,tau,h,w
    con = matirx[v,C].sum()
    vh,vw = v//w,v%w
    Ch,Cw = C//w,C%w
    dist_panelty = np.log(tau/np.sqrt((vh-Ch)**2+(vw-Cw)**2).max()+1)
    return con*dist_panelty

if __name__ == '__main__':
    mode = 'synthetic'

    if mode=='synthetic':
        h, w = 6, 6
        tau = 4
        file = os.path.join(ws, f'data/synthetic_data/{h}_{w}/matrix.npy')
        matrix = np.load(file)
        print(matrix.shape)
        matirx = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    else:
        file = os.path.join(ws, 'data/real_data/trajectory.csv')
        print(f'read data {file}')
        dataHandler = DataHandler(filepath=file, header=0)  # , nrows=5e4
        [h, w], matrix, map, parcle_num, parcels = dataHandler.handle_lonlat(delta_lon=0.0001875,
                                                                 delta_lat=0.0001739,
                                                                 lon_range=[121.4910, 121.4940],
                                                                 lat_range=[31.161, 31.157])

    label = np.arange(h*w)

    # GCLP
    max_iter = 1000
    max_continue_nums,continue_nums = 5,0
    for iter in range(max_iter):
        change_flag = False
        num = int(h*w*0.2)
        label1 = label.copy()
        indexes = np.random.choice(h*w,size=num,replace=False)
        for ind in indexes:
            neighbors = np.where(matrix[ind])[0]
            neighbor_clusters = np.unique(label[neighbors],return_counts=False)
            maxx,vert_c = 0.,0
            for nc in neighbor_clusters:
                C = np.where(label==nc)[0]
                v_nc = value(ind,C)
                if v_nc>maxx:
                    maxx = v_nc
                    vert_c = nc

            if label[ind]!=vert_c:
                change_flag = True
                label1[ind] = vert_c

        label = label1
        if not change_flag:
            continue_nums += 1
            if continue_nums>max_continue_nums:
                print(f'did not change again for {max_continue_nums} times, break')
                break

    print(label.reshape([h,w]))
    np.save(f'GCLP_{h}_{w}.npy', label.reshape([h, w]).astype(np.int16))