import os
import numpy as np
from sklearn.cluster import DBSCAN

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
    parcels = parcels[:5,...]
    fea = parcels.reshape([-1,2]).astype(np.float32)

    # DBSCAN
    if h == 5 and w == 5:
        num_clusters = 2
    elif h == 6 and w == 6:
        num_clusters = 3
    elif h == 10 and w == 10:
        num_clusters = 4
    clustering = DBSCAN(eps=1,min_samples=int(0.3*fea.shape[0]/num_clusters)).fit(fea)

    labels = clustering.labels_
    ans = deduplication(fea.astype(np.int16),labels,h,w)

    ans = ans + 1
    print(ans)
    np.save(f'DBSCAN_{h}_{w}.npy', ans.reshape([h,w]).astype(np.int16))