import numpy as np
import pandas as pd

df = pd.read_csv('road_aoi.csv',header=None)
np.save('road_aoi.npy',df.values)

a = np.load('road_aoi.npy')
print(a)
