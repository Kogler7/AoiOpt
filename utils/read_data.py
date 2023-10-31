import math
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class DataHandler:
    def __init__(self, filepath, nrows=None, header=None):
        names = ['courierID','site_id','time','longitude','latitude','accuracy','date']
        df = pd.read_csv(filepath, encoding='gbk', header=header, nrows=nrows)
        df.columns = names
        df.latitude,df.longitude = df.latitude.astype(np.float32),df.longitude.astype(np.float32)

        print('data read over.')
        df = self.change_data(df)
        # df = self.del_remote_points(df)

        self.df = df

    def change_data(self, df):
        # delete duplicate
        df.drop_duplicates(inplace=True) #subset=['longitude','latitude']
        df.reset_index(drop=True)
        # ID->Index
        name, new_name = 'courierID', 'courierIndex'
        id2index = {id: index for index, id in enumerate(df[name].drop_duplicates())}
        df[new_name] = df[name].map(id2index)
        # latitude,longitude -> x,y
        dy, dx = 111.194 * math.cos(df['longitude'].mean() / 180 * 3.1415926), 111.194  # km/du
        df['y'] = (df['latitude'] - df['latitude'].min()) * dy
        df['x'] = (df['longitude'] - df['longitude'].min()) * dx

        # date -> timestamp
        df['time'] = df['time'].apply(lambda x: pd.to_datetime(str(x)))
        # df['user_start_time'] = df['user_start_time'].apply(lambda x: pd.to_datetime(str(x)))
        # df['user_end_time'] = df['user_end_time'].apply(lambda x: pd.to_datetime(str(x)))

        df = df.drop(labels=['courierID'], axis=1)
        return df

    def del_remote_points(self, df):
        print('before del remote points:', df.shape)
        indexes = df[(abs(df.x - df.x.mean()) > 3 * df.x.std()) | (abs(df.y - df.y.mean()) > 3 * df.y.std())].index
        df = df.drop(indexes)
        print('after del remote points:', df.shape)
        return df

    def handle_lonlat(self, delta_lon, delta_lat, lon_range: list, lat_range: list):
        df = self.df

        df_ranged = df[(df.latitude >= lat_range[1]) & (df.latitude < lat_range[0]) &
                       (df.longitude >= lon_range[0]) & (df.longitude < lon_range[1])]
        print(f'scaled data shape:{df_ranged.shape}')
        # lng,lat -> grid_x,grid_y
        print(f'lon_range:{lon_range},lat_range:{lat_range}')
        df_ranged['grid_y'] = ((lat_range[0] - df_ranged['latitude']) / delta_lat).astype(np.int16)
        df_ranged['grid_x'] = ((df_ranged['longitude'] - lon_range[0]) / delta_lon).astype(np.int16)
        self.df_ranged = df_ranged

        h, w = np.ceil((lat_range[0] - lat_range[1]) / delta_lat).astype(np.int32), np.ceil((lon_range[1] - lon_range[0]) / delta_lon).astype(np.int32)
        grid_num = h * w
        matrix = np.zeros([grid_num, grid_num], dtype=np.int16)
        map = np.zeros([h, w, 2], dtype=np.int16)
        parcel_num = np.zeros([h, w], dtype=np.int16)

        # get the trajectory transfer graph
        diag_num,neighbor_num = 0,0
        for _, df1 in df_ranged.groupby(['courierIndex', 'date']):
            df2 = df1.sort_values(['time'])
            x1, y1 = df2['grid_x'].values, df2['grid_y'].values
            # matrix form
            grid_idx = y1 * w + x1
            np.add.at(matrix, (grid_idx[:-1], grid_idx[1:]), 1)
            # # diagonal trajectories
            map_lu = (x1[:-1] > x1[1:]) & (y1[:-1] > y1[1:])
            map_ld = (x1[:-1] > x1[1:]) & (y1[:-1] < y1[1:])
            map_ru = (x1[:-1] < x1[1:]) & (y1[:-1] > y1[1:])
            map_rd = (x1[:-1] < x1[1:]) & (y1[:-1] < y1[1:])
            diag_num += (map_lu|map_ld|map_ru|map_rd).sum()
            # r_begin, r_end = x1[:-1][map_ru | map_rd], x1[1:][map_ru | map_rd]
            # l_begin, l_end = x1[1:][map_lu | map_ld], x1[:-1][map_lu | map_ld]
            # u_begin, u_end = y1[1:][map_lu | map_ru], y1[:-1][map_lu | map_ru]
            # d_begin, d_end = y1[:-1][map_ld | map_rd], y1[1:][map_ld | map_rd]
            # lr_x_lst, lr_y_lst, ud_x_lst, ud_y_lst = [], [], [], []
            #
            # for i in range(r_begin.shape[0]):
            #     rb, re, y = r_begin[i], r_end[i], y1[:-1][map_ru | map_rd][i]
            #     lr_x_lst.append(np.arange(rb, re))
            #     lr_y_lst.append(np.zeros(re-rb,dtype=np.int32) + y)
            # for i in range(l_begin.shape[0]):
            #     lb, le, y = l_begin[i], l_end[i], y1[:-1][map_lu | map_ld][i]
            #     lr_x_lst.append(np.arange(lb, le))
            #     lr_y_lst.append(np.zeros(le-lb,dtype=np.int32) + y)
            # if len(lr_y_lst)!=0:
            #     lr_x_ind, lr_y_ind = np.concatenate(lr_x_lst, axis=0), np.concatenate(lr_y_lst, axis=0)  # N*1
            #     np.add.at(map, (lr_y_ind, lr_x_ind, 0), 1)
            #
            # for i in range(u_begin.shape[0]):
            #     ub, ue, x = u_begin[i], u_end[i], x1[1:][map_lu | map_ru][i]
            #     ud_x_lst.append(np.zeros(ue-ub,dtype=np.int32) + x)
            #     ud_y_lst.append(np.arange(ub, ue))
            # for i in range(d_begin.shape[0]):
            #     db, de, x = d_begin[i], d_end[i], x1[1:][map_ld | map_rd][i]
            #     ud_x_lst.append(np.zeros(de-db,dtype=np.int32) + x)
            #     ud_y_lst.append(np.arange(db, de))
            # if len(ud_x_lst)!=0:
            #     ud_x_ind, ud_y_ind = np.concatenate(ud_x_lst, axis=0), np.concatenate(ud_y_lst, axis=0)  # N*1
            #     np.add.at(map, (ud_y_ind, ud_x_ind, 1), 1)
            # map: right and left
            map_r = (x1[:-1] < x1[1:]) & (y1[:-1] == y1[1:])
            map_l = (x1[:-1] > x1[1:]) & (y1[:-1] == y1[1:])
            np.add.at(map, (y1[:-1][map_r], x1[:-1][map_r], 0), 1)
            np.add.at(map, (y1[1:][map_l], x1[1:][map_l], 0), 1)

            # for i,j in zip(y1[:-1][map_r], x1[:-1][map_r]):
            #     print(f'map add,({i},{j},right)')
            # for i, j in zip(y1[1:][map_l], x1[1:][map_l]):
            #     print(f'map add,({i},{j},right)')
            # # map: up and down
            map_d = (x1[:-1] == x1[1:]) & (y1[:-1] < y1[1:])
            map_u = (x1[:-1] == x1[1:]) & (y1[:-1] > y1[1:])
            np.add.at(map, (y1[1:][map_u], x1[1:][map_u], 1), 1)
            np.add.at(map, (y1[:-1][map_d], x1[:-1][map_d], 1), 1)
            # for i, j in zip(y1[1:][map_u], x1[1:][map_u]):
            #     print(f'map add,({i},{j},down)')
            # for i, j in zip(y1[:-1][map_d], x1[:-1][map_d]):
            #     print(f'map add,({i},{j},down)')

            # parcel_num
            np.add.at(parcel_num, (y1, x1), 1)
            '''
            # cyclic form
            for i in range(0, df2.shape[0] - 1):
                di1, di2 = df2.iloc[i], df2.iloc[i + 1]
                x1, y1 = di1['grid_x'], di1['grid_y']
                x2, y2 = di2['grid_x'], di2['grid_y']
                if y1 == y2 and x1 != x2:
                    map[y1, min(x1, x2), 0] += 1
                    # print(f'map add,({y1},{min(x1,x2)},right)')
                elif x1 == x2 and y1 != y2:
                    map[min(y1, y2), x1, 1] += 1
                    # print(f'map add,({min(y1,y2)},{x1},down)')
                matrix[w * y1 + x1, w * y2 + x2] += 1
            '''
            neighbor_num += map.sum()

        print(f'There ara {diag_num} diagonal trajectories.')
        print(f'There are {neighbor_num} adjacent trajectories.')
        # df_ranged['ust'] = df_ranged['user_start_time'].apply(lambda x: x.hour * 60 + x.minute)
        # df_ranged['uet'] = df_ranged['user_end_time'].apply(lambda x: x.hour * 60 + x.minute)
        parcels = []
        for day, day_df in df_ranged.groupby('date'):
            # parcel_1day = day_df[['ust','uet']].values.astype(np.int16)
            parcel_1day = day_df[['grid_x', 'grid_y']].values.astype(np.int16)
            parcels.append(parcel_1day)

        return [h, w], matrix, map, parcel_num, parcels
