# 1 Environment

```
conda create -n AoiOpt python=3.9.0
conda activate AoiOpt
pip install -r requirements.txt
```

# 2 Quick Run

## 2.1 Training script
Training single model.
```
python main.py --mode train --env_type synthetic --name 10_10 --device -1 --max_erg 1 --tra_eps 100 --traj_reward_weight 0.6 --road_reward_weight 0.4 --model_name test
```
Training multiple models.
```
python multi_main.py --tra_eps 100 --max_erg 1 --traj_reward_weight 0.4 0.6 0.8 1.0 --road_reward_weight 0.4 0.6 0.8 1.0
```
Or
```
./bash_run.sh --source ./task.csv
```

## 2.2 Evaluation script
```
python main.py --mode test --env_type synthetic --name 10_10 --device -1 --max_erg 1 --tra_eps 100 --traj_reward_weight 0.6 --road_reward_weight 0.4 --model_name test
```

# 3 Repo Structure
The structure of the code and description of important files are given as follows:  
├────algorithm/  
│    ├───baseline/:  Code of baseline.   
│    └───pfrl_learning/:  Code of pfrl-based RL model.  
├────data/ 
│    ├────real_data/:  Real world dataset. 
│    ├────synthetic_data/:  Synthetic datasets.
│    └────color_set.txt:  Colors for drawing.
├────multi_fig_viewer/:  Figures viewing platform. 
├────output/:  Output of training.
├────result/:  Result of different parts.
├────utils/    
│    ├────ans_performance.py: Implemention of the evaluation metrics. 
│    ├────check_aoi.py: Generate all possible AOIs.    
│    ├────data_creater.py: Generate synthetic dataset.  
│    ├────data_loader.py: Read and convert data in any format.
│    ├────data_looker.py: Plot the map of trajectories in real world.
│    ├────metrics.py: Implemention of different metrics.
│    ├────options.py: Code of parser for main and multi_main.
│    ├────plot_fig.py: Plot reward figures. 
│    ├────read_data.py: Process real world data as a matrix.
│    ├────read_real.py: Read and proccess real world data. 
│    ├────Scaler.py: Scaling reward.
│    └────util.py: Provide the project running path.  
├────visual_plat/: Implemention of visualization  
├────bash_run.sh: Train models in parallel.
├────kill.sh: Kill all training process.
├────main.py: Main function of this project, mainly used for training and evaluating different models. 
├────multi_main.py: Train and test models in parallel.
└────tasks.csv: Configs for multiple models, used in bash_run.sh.

# 4 Dataset

## 4.1 Synthetic data  

### 4.1.1 Introduction
To test the performance of different methods in the ideal case, we first create a benchmark synthetic dataset.
Each data is composed of 5 files in the model, aoi.npy, map.npy, matrix.npy, road_aoi.npy and traj.npy. 
aoi.npy: The initial AOI segmentation. Each grid has a AOI number which is what it belongs to.
road_aoi.npy: The road-network based segmentation.
map.npy: The number of times each points crosses the right and down traces.(Row,Col,2)It is used in visual platform.
matrix.npy: It is similar to map.npy, but its index lists all points.(Row\*Col,Row\*Col)It is used in aoi_venv.
traj.npy: Trajectories generated from parcels. A trajectory is composed of coordinate of all points passed.

### 4.1.2 Generation
1) Prepare label AOI segmentation file, which must be a .csv file, take /data/synthetic_data/5_5/aoi.csv as an example, 
```python
aoi.csv
1,1,2,2,2
1,1,2,2,2
1,1,2,2,2
1,1,2,2,2
1,1,1,2,2
```
And a road-network based segmentation file is also needed, which is similar as aoi.csv.

2) Set the path of the aoi file in `/utils/data_creater.py`, 
```python
grid = GetGrid( '../data/synthetic_data/5_5/aoi.csv')
```
In this case,  `aoi_path = '../data/synthetic_data/5_5/aoi.csv'`

3) Generate data for training
```python
python /utils/data_creater.py
```
After runing the above code, four .npy files are generated:  
map.npy:  The number of times each points crosses the right and down traces.(Row,Col,2)It is used in visual platform.
matrix.npy: It is similar to map.npy, but its index lists all points.(Row\*Col,Row\*Col)It is used in aoi_venv.
parcel.npy: Parcels generated from AOI, which has 3-dim, (Number of package groups, Number of packages generated per group, 2)
traj.npy: Trajectories generated from parcels. A trajectory is composed of coordinate of all points passed.

## 4.2 Real-world data  

### 4.2.1 Introduction
Our research is based on the delivery data provided by one of the world's largest logistics companies, which includes data from Hangzhou and Shanghai, China. The dataset comprises both order information data.csv and trajectory data trajectory.csv. 
data.csv: Recording the parcels information.
| datafield | introduction | format |
| postman_id | couriers' ID | String |
| site_id | site ID the parcel belongs to | String |
| gc_lng | longitude of the parcel | Double(GCJ-02) |
| gc_lat | latitude of the parcel | Double(GCJ-02) |
| create_time | time the parcel is created | String |
| expect_earliest_time | the earliest time the parcel will be sent | String |
| expect_lastest_time | the lastest time the parcel will be sent | String |
| accept_time | the time when the order is sent to a courier | String |
| got_time | the time courier getting this parcel | String |

trajectory.csv: Recording trajectory points.
| datafield | introduction | format |
| postman_id | couriers' ID | String |
| site_id | site ID the parcel belongs to | String |
| gps_time | record time | datetime |
| longitude | longitude of the parcel | String(GCJ-02) |
| latitude | latitude of the parcel | String(GCJ-02) |
| accuracy | position accuracy | Double |

# 5 Platform 

## 5.1 Introduction
We propose a high-performance visualization solution. We have independently designed and developed a visualization platform that can dynamically render AOI grids, parcels, trajectories, and road networks.
When used as standalone software, users can drag and drop their own data into the software for automatic loading and rendering. Users can then zoom and pan the map data and view relevant information by clicking or selecting specific areas, such as the coordinates and values of selected areas or elements.

## 5.2 function
- drag: Use the right mouse button to drag the interface. 
- Scaling: Use the mouse wheel to zoom in or zoom out.
- Onboard data: Select the data (.npy), and drag it to the interface.
- new window: Use *ctrl+N* to create a new window
- Recording: Use *ctrl+R* to record the training process. A record file (.rcd) will be create in the /output/record.
- playback: Select the .rcd file, and drag it into the interface. You can use the space bar to pause, and use the left and right keys to go forward or backward.
