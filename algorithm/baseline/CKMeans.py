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
    # read parcels
    parcels = np.load(f'../../data/synthetic_data/{h}_{w}/parcels.npy')
    print(parcels.shape)
    fea = parcels.reshape([-1,2]).astype(np.float32)

    # CKMeans
    if h==5 and w==5:
        num_clusters = 2
    elif h==6 and w==6:
        num_clusters = 3
    elif h==10 and w==10:
        num_clusters = 4

    clustering = KMeansConstrained(
        n_clusters=num_clusters,
        size_min = 2,
        size_max = ceil(fea.shape[0]/num_clusters),
        random_state= 0
    )
    clustering.fit_predict(fea)

    labels = clustering.labels_
    ans = deduplication(fea.astype(np.int16), labels, h, w)

    np.save(f'CKMeans_{h}_{w}.npy', ans.reshape([h,w]).astype(np.int16))