import os
from math import ceil

import numpy as np
from k_means_constrained import KMeansConstrained

from utils.util import ws
from utils.read_data import DataHandler

def deduplication(coor,label,h,w):
    grid_labels = np.zeros([h,w],dtype=np.int16)
    unique_coordinates = np.unique(coor, axis=0)

    for current_coordinates in unique_coordinates:
        mask = np.all(coor == current_coordinates, axis=1)
        unique_classes, class_counts = np.unique(label[mask], return_counts=True)
        most_common_class = unique_classes[np.argmax(class_counts)]
        grid_labels[current_coordinates[0],current_coordinates[1]] = most_common_class

    return grid_labels

if __name__=='__main__':
    h, w = 10,10
    mode = 'synthetic'
    # read parcels
    if mode=='synthetic':
        parcels = np.load(f'../../data/synthetic_data/{h}_{w}/parcels.npy')
        print(parcels.shape)
        fea = parcels.reshape([-1,2]).astype(np.float32)
    else:
        file = os.path.join(ws, 'data/real_data/trajectory.csv')
        print(f'read data {file}')
        dataHandler = DataHandler(filepath=file, header=0)  # , nrows=5e4
        [h, w], matrix, map, parcle_num, parcels = dataHandler.handle_lonlat(delta_lon=0.0001875,
                                                                             delta_lat=0.0001739,
                                                                             lon_range=[121.4910, 121.4940],
                                                                             lat_range=[31.161, 31.157])
        print(parcels.shape)
        fea = parcels.reshape([-1, 2]).astype(np.float32)

    # CKMeans
    if h==5 and w==5:
        num_clusters = 2
    elif h==6 and w==6:
        num_clusters = 3
    elif h==10 and w==10:
        num_clusters = 4

    if mode=='synthetic':
        clustering = KMeansConstrained(
            n_clusters=num_clusters,
            size_min = 2,
            size_max = ceil(fea.shape[0]/num_clusters),
            random_state= 0
        )
    else:
        num_clusters = (np.sqrt(h * w) / 2).astype(np.int16)
        # calculate eps, which is the bottom 5% distance of samples.
        num_samples = 100 if fea.shape[0]>100 else fea.shape[0]
        selected_indices = np.random.choice(fea.shape[0],size=num_samples,replace=False)
        fea_samples = fea[selected_indices,:]
        dist_1 = np.tile(fea_samples,(num_samples,1)) # N*N*2
        dist_2 = np.expand_dims(fea_samples,axis=1) # N*1*2
        diff_matrix = dist_1 - dist_2
        dist = np.linalg.norm(diff_matrix, axis=2) # N*N

        dist_percent = 0.05
        eps = np.percentile(dist.reshape[-1],[dist_percent])[0]
        clustering = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=eps,
            size_max=ceil(fea.shape[0]/num_clusters),
            random_state=0
        )
    clustering.fit_predict(fea)

    # 根据聚类结果划分区域（区域有多个类的投票选择）
    labels = clustering.labels_
    if mode=='synthetic':
        ans = deduplication(fea.astype(np.int16), labels, h, w)
    else:
        coor = dataHandler.df_ranged[['grid_x', 'grid_y']].values
        ans = deduplication(coor, labels, h, w)

    np.save(f'CKMeans_{h}_{w}.npy', ans.reshape([h,w]).astype(np.int16))