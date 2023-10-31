import os
import folium
import pandas as pd

from utils.util import ws

def on_marker_click(e):
    # callback function for markers
    marker = e.target
    lat, lng = marker.getLatLng()
    print(f"longitude: {lng}, latitiude: {lat}")

if __name__=='__main__':
    data_path = os.path.join(ws,'data/real_data/trajectory_small.csv')
    print(data_path)
    raw = pd.read_csv(data_path, encoding='gbk',nrows=5e4) #

    lng_range=[121.4910 , 121.4940]
    lat_range=[31.1606 , 31.1573]
    data = raw[(raw.lng>=lng_range[0]) & (raw.lng<=lng_range[1]) & \
               (raw.lat>=lat_range[1]) & (raw.lat<=lat_range[0])]
    print(f'scaled data shape:{data.shape}')

    map = folium.Map(location=[data['lat'].mean(), data['lng'].mean()], zoom_start=10,control_scale=True)
    # add markers in the map
    for index, row in data.iterrows():
        marker = folium.Marker([row['lat'], row['lng']], icon=folium.Icon(icon_size=(5, 5)))
        marker.add_to(map)


    map.add_child(folium.LatLngPopup())
    # save the map
    save_path = os.path.join(ws,'data/real_data/traj.html')
    print(save_path)
    map.save(save_path)